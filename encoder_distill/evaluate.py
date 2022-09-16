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

def dual_loss_eval(student_model, teacher_model, data, loss, autocast, args):
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
                si_feat, st_feat = student_model(val_img, val_txt)
                si_feat, st_feat = F.normalize(si_feat, dim=-1), F.normalize(st_feat, dim=-1)
                ti_feat, tt_feat = teacher_model(val_img, val_txt)
                ti_feat, tt_feat = F.normalize(ti_feat, dim=-1), F.normalize(tt_feat, dim=-1)

                s_similarities = si_feat @ st_feat.T
                t_similarities = ti_feat @ tt_feat.T

                val_loss = loss(s_similarities, t_similarities)
                tot_loss += val_loss.item()
        tot_loss /= n_batch
    eval_metrics = {f"val/similarity_loss": tot_loss}
    return eval_metrics
