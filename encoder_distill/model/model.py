import open_clip
import torch

from torch import nn

from .clip import MLPCLIP, CLIPImage, CLIPText, combine_image_text


class MLPEncoder(nn.Module):
    def __init__(self, model, mlp):
        super().__init__()
        self.model = model
        self.mlp = mlp
    def forward(self, x):
        return self.mlp(self.model(x))


def create_model_and_transforms(model_type, model_kwargs, modality, mlp_dims, device):
    if "clip" in model_type.lower():
        model, _, preprocess = open_clip.create_model_and_transforms(**model_kwargs)
        model.set_grad_checkpointing() # TODO do we always want to do this?
        model.to(device) # TODO: does this double allocate on GPU?

        model_image, mlp_image, preprocess_image = None, None, None
        model_text, mlp_text, preprocess_text = None, None, None

        if "text" in modality:
            model_text = CLIPText(model.token_embedding, model.positional_embedding, model.transformer, model.attn_mask, model.ln_final, model.text_projection)
            preprocess_text = nn.Identity()
            mlp_image = nn.Linear(*mlp_dims, bias=False) if mlp_dims is not None else nn.Identity()
        elif "image" in modality:
            model_image = CLIPImage(model.visual)
            preprocess_image = preprocess
            mlp_text = nn.Linear(*mlp_dims, bias=False) if mlp_dims is not None else nn.Identity()


    if model_image is not None and model_text is not None:
        encoder_text = MLPEncoder(model_text, mlp_text)
        encoder_image = MLPEncoder(model_image, mlp_image)
        encoder = MLPCLIP(encoder_image, encoder_text) # TODO: Generalize this into a DualEncoder or something
        preprocess = (preprocess_image, preprocess_text)
    else:
        model, mlp, preprocess = model_image, mlp_image, preprocess_image if model_image is not None else model_text, mlp_text, preprocess_text
        encoder = MLPEncoder(model, mlp)
        encoder.to(device)

    return encoder, preprocess


def load_cpt(model, cpt_path, dev):
	checkpoint = torch.load(cpt_path, map_location=dev)
	sd = checkpoint["state_dict"]
	if next(iter(sd.items()))[0].startswith('module'):
		sd = {k[len('module.'):]: v for k, v in sd.items()}
	model.load_state_dict(sd)
	return model

def put_together(model_type, model_kwargs, mlp_dims, image_checkpoint, text_checkpoint, remove_mlp, device):
    # Create models
    img_model, preprocess = create_model_and_transforms(model_type, model_kwargs, "image", mlp_dims, device)
    txt_model, _ = create_model_and_transforms(model_type, model_kwargs, "text", mlp_dims, device)

    # Load checkpoints
    img_model = load_cpt(img_model, image_checkpoint, device)
    txt_model = load_cpt(txt_model, text_checkpoint, device)

    if "clip" in model_type.lower():
        model = combine_image_text(img_model, txt_model, remove_mlp, model_kwargs)
    else:
        model = None

    return model, preprocess
