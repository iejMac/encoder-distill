import os
import torch

from data import get_data
from distributed import world_info_from_env
from model import put_together
from params import parse_args
from zero_shot import zero_shot_eval

dev = "cuda" if torch.cuda.is_available() else "cpu"

model_type = "clip"
model_kwargs = {"model_name": "ViT-H-14"}
mlp_dims = (1024, 768)

image_cpt = "checkpoints/H_65536_bs_1e-3_lr_10k_GPUh/step_10000.pt"
text_cpt = "checkpoints/54k_text/step_54000.pt"

args = parse_args()
# discover initial world args early so we can log properly
args.distributed = False
args.local_rank, args.rank, args.world_size = world_info_from_env()
args.device = "cuda"

model, preproc = put_together(model_type, model_kwargs, mlp_dims, image_cpt, text_cpt, args.device)
model.to(args.device)
data = get_data(args, (preproc, preproc))

zero_shot_metrics = zero_shot_eval(model, data, 0, args)

print(zero_shot_metrics)

