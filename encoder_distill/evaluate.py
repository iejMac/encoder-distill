import torch
import torch.nn.functional as F

from tqdm import tqdm


def loss_eval(student_model, teacher_model, data, loss, autocast, args):
    student_model.eval()
    dev = torch.device(args.device)
    val_dat = data['val'].dataloader
    n_batch = 0
    tot_loss = 0.0

    batch_thresh = args.val_num_samples // args.batch_size + 1

    with torch.no_grad():
        for batch in tqdm(val_dat, unit_scale=args.batch_size):
            if n_batch > batch_thresh:
                break
            val_img, val_txt = batch

            if args.modality == "image":
                val_x = val_img
            elif args.modality == "text":
                val_x = val_txt

            val_x = val_x.to(dev, non_blocking=True)

            n_batch += 1
            with autocast():
                s_feat = student_model(val_x)
                t_feat = teacher_model(val_x)

                val_loss = loss(s_feat, t_feat)
                tot_loss += val_loss.item()
        tot_loss /= n_batch
    eval_metrics = {f"val/{args.modality}_loss": tot_loss}
    return eval_metrics

def dual_loss_eval(student_model, teacher_model, data, autocast, args):
    student_model.eval()
    dev = torch.device(args.device)
    val_dat = data['val'].dataloader
    n_batch = 0
    tot_loss = 0.0

    batch_thresh = args.val_num_samples // args.batch_size + 1

    with torch.no_grad():
        for batch in tqdm(val_dat, unit_scale=args.batch_size):
            if n_batch > batch_thresh:
                break
            val_img, val_txt = batch
            val_img, val_txt = val_img.to(dev, non_blocking=True), val_txt.to(dev, non_blocking=True)

            n_batch += 1
            with autocast():
                si_feat, st_feat, s_ls = student_model(val_img, val_txt)
                ti_feat, tt_feat, _ = teacher_model(val_img, val_txt)
                s_ls = s_ls.mean()

                s_logits_per_image = s_ls * si_feat @ st_feat.T
                s_logits_per_text = s_logits_per_image.T
                # s_logits_per_image = si_feat @ st_feat.T
                # s_logits_per_text = s_logits_per_image.T
                t_logits_per_image = ti_feat @ tt_feat.T # no logit_scale
                t_logits_per_text = t_logits_per_image.T

                t_logits_per_image = t_logits_per_image.softmax(dim=1)
                t_logits_per_text = t_logits_per_text.softmax(dim=1)

                val_loss = (
                        F.cross_entropy(s_logits_per_image, t_logits_per_image) +
                        F.cross_entropy(s_logits_per_text, t_logits_per_text)
                    ) / 2 
                '''
                val_loss = (
                    F.mse_loss(s_logits_per_image, t_logits_per_image) +
                    F.mse_loss(s_logits_per_text, t_logits_per_text)
                ) / 2
                '''
                tot_loss += val_loss.item()
        tot_loss /= n_batch
    eval_metrics = {f"val/similarity_loss": tot_loss}
    return eval_metrics
