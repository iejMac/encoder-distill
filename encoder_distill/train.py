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
from distributed import init_distributed_device, is_master, world_info_from_env
from params import parse_args
from zero_shot import zero_shot_eval
from scheduler import cosine_lr
from model import create_model_and_transforms
from evaluate import loss_eval


def main():
    # Args
    args = parse_args()

    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    # fully initialize distributed device environment
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    dev = init_distributed_device(args)

    if is_master(args) and (args.report_to == "wandb"):
        wandb.init(project="h14_distillation", entity="iejmac", name=args.name)

    # Model
    teacher_model, preprocess_t = create_model_and_transforms("clip", {"model_name": "ViT-L-14", "pretrained": "laion400m_e32"}, args.modality, None, dev)
    student_model, preprocess_s = create_model_and_transforms("clip", {"model_name": "ViT-H-14"}, args.modality, (1024, 768), dev)
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
    if is_master(args):
        os.makedirs(args.checkpoint_path, exist_ok=True)

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

    # Scheduler:
    scheduler = cosine_lr(opt, args.lr, args.warmup, args.steps)

    # Data:
    data = get_data(args, (preprocess, preprocess))

    data['train'].set_epoch(0)
    tr_dat = data["train"].dataloader
    step = start_step + 1

    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    student_model.train()

    for batch in tr_dat:
        if step > args.steps:
            break
        t0 = time.perf_counter()
        metrics = {}
        scheduler(step)
        metrics.update({"train/lr": opt.param_groups[0]["lr"]})

        images, texts = batch
        if args.modality == "image":
            x = images
        elif args.modality == "text":
            x = texts

        x = x.to(dev, non_blocking=True)

        t0_t_forward = time.perf_counter()
        with torch.no_grad():
            with autocast():
                t_feat = teacher_model(x)
        t_t_for = time.perf_counter() - t0_t_forward

        metrics.update({"train/teacher_forward_samples_per_s": x.shape[0]/t_t_for})

        t0_s_forward = time.perf_counter()
        with autocast():
            s_feat = student_model(x)
            total_loss = loss(s_feat, t_feat)

        total_loss.backward()
        t_s_for_back = time.perf_counter() - t0_s_forward
        metrics.update({"train/student_forward_backward_samples_per_s": x.shape[0]/t_s_for_back})
        
        opt.step()
        opt.zero_grad()

        metrics.update({f"train/{args.modality}_loss": total_loss.item()})

        # MSE eval
        if step % args.val_frequency == 0:
            eval_metrics = loss_eval(student_model, teacher_model, data, loss, autocast, args)
            metrics.update(eval_metrics)
            student_model.train()

        if is_master(args) and (step % args.save_frequency == 0):
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

        tf = time.perf_counter()
        metrics.update({"train/samples_per_s": x.shape[0]/(tf-t0)})

        if is_master(args):
            for name, val in metrics.items():
                if args.report_to == "wandb":
                    wandb.log({name: val}, step=step)
                elif args.report_to == "stdout":
                    print(name, val)

        step += 1

if __name__ == "__main__":
    main()
