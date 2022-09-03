import torch

from tqdm import tqdm


def loss_eval(student_model, teacher_model, data, loss, autocast, args):
    student_model.eval()
    dev = torch.device(args.device)
    val_dat = data['val'].dataloader
    n_batch = 0
    tot_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_dat, unit_scale=args.batch_size):
            if n_batch > 3: # TODO: remove
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
