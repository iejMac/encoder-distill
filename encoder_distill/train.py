import os
import wandb
import time
import torch
from PIL import Image
import open_clip
from argparse import Namespace

from collections import OrderedDict
from contextlib import suppress
from tqdm import tqdm
from torch import nn

from data import get_data
from distributed import init_distributed_device, is_master
from params import parse_args
from zero_shot import zero_shot_eval
from scheduler import cosine_lr
from model import create_model_and_transforms


'''
class MLPCLIP(torch.nn.Module):
    def __init__(self, model, img_mlp, txt_mlp, device):
        super().__init__()
        self.model = model
        self.img_mlp = img_mlp
        self.txt_mlp = txt_mlp

        self.dev = device

    def encode_text(self, text):
        temp = self.model.encode_text(text)
        temp = self.txt_mlp(temp)
        return temp
    def encode_image(self, image):
        temp = self.model.encode_image(image)
        temp = self.img_mlp(temp)
        return temp

    def forward(self, img, txt):
        img_feat = self.encode_image(img)
        txt_feat = self.encode_text(txt)
        return img_feat, txt_feat, self.model.logit_scale
'''

def main():
    # Args
    args = parse_args()
    dev = init_distributed_device(args)

    if is_master(args):
        pass
        # wandb.init(project="h14_distillation", entity="iejmac", name=args.name)


    '''
    # Model
    model_l, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion400m_e32')
    model_h, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14') # MAX BS = ~256
    model_h.set_grad_checkpointing()

    d_model_l = 768
    d_model_h = 1024

    img_mlp = nn.Linear(d_model_h, d_model_l, bias=False)
    txt_mlp = nn.Linear(d_model_h, d_model_l, bias=False)

    mlp_model_l = MLPCLIP(model_l, nn.Identity(), nn.Identity(), dev).to(dev)
    mlp_model_h = MLPCLIP(model_h, img_mlp, txt_mlp, dev).to(dev)
    '''
    # Model
    MODALITY = "text"
    teacher_model, preprocess_t = create_model_and_transforms("clip", {"model_name": "ViT-L-14", "pretrained": "laion400m_e32"}, MODALITY, None, dev)
    student_model, preprocess_s = create_model_and_transforms("clip", {"model_name": "ViT-H-14"}, MODALITY, (1024, 768), dev)
    preprocess = preprocess_t # = preprocess_s for now

    # Loss and Opt:
    loss = nn.MSELoss()
    params = student_model.parameters()

    opt = torch.optim.AdamW(
        params=params,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.wd,
    )

    start_step = 0
    args.checkpoint_path = "checkpoints"
    if args.resume is not None and os.path.isfile(args.resume):
        # NOTE: resuming doesn't work with torch >1.11.0 yet (https://github.com/pytorch/pytorch/issues/80809)
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'step' in checkpoint:
            # resuming a train checkpoint w/ step and optimizer state
            start_step = checkpoint["step"]
            sd = checkpoint["state_dict"]
            if next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            student_model.load_state_dict(sd)
            if opt is not None:
                opt.load_state_dict(checkpoint["optimizer"])
            print(f"=> resuming checkpoint '{args.resume}' (step {start_step})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            student_model.load_state_dict(checkpoint)
            print(f"=> loaded checkpoint '{args.resume}' (step {start_step})")


    student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[dev])

    TOTAL_STEPS = 100000
    WARMUP = 1000

    # Scheduler:
    scheduler = cosine_lr(opt, args.lr, WARMUP, TOTAL_STEPS)


    # Data:
    data = get_data(args, (preprocess, preprocess))

    data['train'].set_epoch(0)
    tr_dat = data["train"].dataloader
    step = start_step + 1

    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    student_model.train()

    for batch in tr_dat:
        if step > TOTAL_STEPS:
            break
        t0 = time.perf_counter()
        metrics = {}
        scheduler(step)
        metrics.update({"train/lr": opt.param_groups[0]["lr"]})

        images, texts = batch
        if MODALITY == "image":
            x = images
        elif MODALITY == "text":
            x = texts

        x = x.to(dev, non_blocking=True)

        t0_t_forward = time.perf_counter()
        with torch.no_grad():
            with autocast():
                # l_img_feat, l_txt_feat, l_logit_scale = mlp_model_l(images, texts)
                t_feat = teacher_model(x)
                # l_img_feat = model_l.encode_image(images)
                # l_txt_feat = model_l.encode_text(texts)
        t_t_for = time.perf_counter() - t0_t_forward

        metrics.update({"train/teacher_forward_samples_per_s": x.shape[0]/t_t_for})

        t0_s_forward = time.perf_counter()
        with autocast():
            # h_img_feat, h_txt_feat, h_logit_scale = student_model(images, texts)
            '''
            loss_img = loss(h_img_feat, l_img_feat) + h_logit_scale - h_logit_scale
            loss_txt = loss(h_txt_feat, l_txt_feat) + h_logit_scale - h_logit_scale
            total_loss = loss_img + loss_txt
            '''
            s_feat = student_model(x)
            total_loss = loss(s_feat, t_feat)

        total_loss.backward()
        t_s_for_back = time.perf_counter() - t0_s_forward
        metrics.update({"train/student_forward_backward_samples_per_s": x.shape[0]/t_s_for_back})
        
        opt.step()
        opt.zero_grad()

        # metrics.update({"train/img_loss": loss_img.item(), "train/txt_loss": loss_txt.item()})
        metrics.update({f"train/{MODALITY}_loss": total_loss.item()})

        # Zero-shot eval
        if step % args.zeroshot_frequency == 0:
            student_model.eval()
            zero_shot_metrics = zero_shot_eval(student_model, data, 0, args)
            metrics.update(zero_shot_metrics)

            # Save checkpoint:
            if is_master(args):
                print(f"Saving checkpoint at step {step}...")
                checkpoint_dict = {
                    "step": step,
                    "name": args.name,
                    "state_dict": student_model.state_dict(),
                    "optimizer": opt.state_dict(),
                }        
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"step_{step}.pt"),
                )

            student_model.train()
        # MSE eval
        if step % args.val_frequency == 0:
            student_model.eval()
            val_dat = data['val'].dataloader
            n_batch = 0
            # tot_img_loss, tot_txt_loss = 0.0, 0.0
            tot_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_dat, unit_scale=args.batch_size):
                    if n_batch > 3: # TODO: remove
                        break
                    val_img, val_txt = batch
                    if MODALITY == "image":
                        x_val = val_img
                    elif MODALITY == "test":
                        x_val = val_txt
                        
                    val_x = val_x.to(dev, non_blocking=True)

                    n_batch += 1
                    with autocast():
                        # img_feat, txt_feat, logit_scale = student_model(val_img, val_txt)
                        s_feat = student_model(val_x)
                        # targ_img_feat, targ_txt_feat = model_l.encode_image(val_img), model_l.encode_text(val_txt)
                        # targ_img_feat, targ_txt_feat, l_log_scale = teacher_model(val_img, val_txt)
                        t_feat = teacher_model(val_x)

                        val_loss = loss(s_feat, t_feat)
                        tot_loss += val_loss.item()
                        '''
                        val_loss_img = loss(img_feat, targ_img_feat)
                        val_loss_txt = loss(txt_feat, targ_txt_feat)
                        tot_img_loss += val_loss_img.item()
                        tot_txt_loss += val_loss_txt.item()
                        '''
                '''
                tot_txt_loss /= n_batch
                tot_img_loss /= n_batch
                '''
                tot_loss /= n_batch
            # eval_metrics = {"val/img_loss": tot_img_loss, "val/txt_loss": tot_txt_loss}
            eval_metrics = {f"val/{MODALITY}_loss": tot_loss}
            metrics.update(eval_metrics)

            student_model.train()

        tf = time.perf_counter()
        metrics.update({"train/samples_per_s": x.shape[0]/(tf-t0)})

        if is_master(args):
            for name, val in metrics.items():
                if not (name.startswith("train") or name.startswith("val")):
                    name = "val/" + name # hack for zero-shot stuff
                # wandb.log({name: val}, step=step)
                print(name, val)

        step += 1

if __name__ == "__main__":
    main()
