import os
import torch
import open_clip

from data import get_data
from distributed import world_info_from_env
from model import put_together
from params import parse_args
from zero_shot import zero_shot_eval

dev = "cuda" if torch.cuda.is_available() else "cpu"
args = parse_args()
# discover initial world args early so we can log properly
args.distributed = False
args.local_rank, args.rank, args.world_size = world_info_from_env()
args.device = "cuda"

'''
model_type = "clip"
model_kwargs = {"model_name": "ViT-H-14"}
mlp_dims = (1024, 768)

# image_cpt = "checkpoints/H_16384_bs_1e-3_lr/step_16000.pt"
image_cpt = "checkpoints/H_65536_bs_1e-3_lr_40k_steps/step_35300.pt"
text_cpt = "checkpoints/H_16384_bs_5e-4_lr/step_10000.pt"
# text_cpt = "checkpoints/54k_text/step_54000.pt"
'''
'''
model_type = "clip"
model_kwargs = {"model_name": "ViT-B-32"}
mlp_dims = (512, 768)

image_cpt = "checkpoints/B32_16k_bs_1e-3_lr_10k_steps/step_10000.pt"
text_cpt = "checkpoints/B32_16k_bs_1e-3_lr_10k_steps_text/step_10000.pt"


model, preproc = put_together(model_type, model_kwargs, mlp_dims, image_cpt, text_cpt, False, args.device)
model.to(args.device)
'''
model, _, preproc = open_clip.create_model_and_transforms('ViT-H-14', pretrained="/fsx/iejmac/open_clip_norm/open_clip/src/logs/distilled_clip70_50k_bs_lr_5e4/checkpoints/epoch_14.pt")
model.to(args.device)
data = get_data(args, (preproc, preproc))

zero_shot_metrics = zero_shot_eval(model, data, 0, args)

print(zero_shot_metrics)

