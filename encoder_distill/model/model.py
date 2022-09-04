import open_clip
import torch

from torch import nn

from .clip import CLIPImage, CLIPText, combine_image_text


class MLPEncoder(nn.Module):
    def __init__(self, model, mlp):
        super().__init__()
        self.model = model
        self.mlp = mlp
    def forward(self, x):
        return self.mlp(self.model(x))


def create_model_and_transforms(model_type, model_kw_args, modality, mlp_dims, device):
    if "clip" in model_type.lower():
        model, _, preprocess = open_clip.create_model_and_transforms(**model_kw_args)
        model.set_grad_checkpointing() # TODO do we always want to do this?
        model.to(device) # TODO: does this double allocate on GPU?

        if modality == "text":
            model = CLIPText(model.token_embedding, model.positional_embedding, model.transformer, model.attn_mask, model.ln_final, model.text_projection)
            preprocess = nn.Identity()
        elif modality == "image":
            model = CLIPImage(model.visual)
            preprocess = preprocess

    mlp = nn.Linear(*mlp_dims, bias=False) if mlp_dims is not None else nn.Identity()

    encoder = MLPEncoder(model, mlp)
    encoder.to(device)
    return encoder, preprocess


def put_together(model_type, model_kw_args, mlp_dims, image_checkpoint, text_checkpoint, device):
    # Create models
    img_model, preprocess = create_model_and_transforms(model_type, model_kw_args, "image", mlp_dims, device)
    txt_model, _ = create_model_and_transforms(model_type, model_kw_args, "text", mlp_dims, device)

    # Load checkpoints
    # TODO fill in

    if "clip" in model_type.lower():
        model = combine_image_text(img_model, txt_model, mlp_dims)
    else:
        model = None

    return model, preprocess
