import open_clip
import torch
import torch.nn.functional as F

from torch import nn


class CLIPImage(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual
    def forward(self, image):
        return self.visual(image)


class CLIPText(nn.Module):
    def __init__(self, token_embedding, positional_embedding, transformer, attn_mask, ln_final, text_projection):
        super().__init__()
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.transformer = transformer
        self.attn_mask = attn_mask
        self.ln_final = ln_final
        self.text_projection = text_projection
    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


# Useful for evaluation
class MLPCLIP(torch.nn.Module):
    def __init__(self, img_model, txt_model, logit_scale):
        super().__init__()
        self.img_model = img_model
        self.txt_model = txt_model
        self.logit_scale = logit_scale

    def encode_text(self, text):
        return self.txt_model(text)
    def encode_image(self, image):
        return self.img_model(image)

    def forward(self, img, txt):
        img_feat = self.encode_image(img)
        F.normalize(img_feat, dim=-1)
        txt_feat = self.encode_text(txt)
        F.normalize(txt_feat, dim=-1)
        return img_feat, txt_feat, self.logit_scale.exp()

def combine_image_text(image_model, text_model, remove_mlp, model_kwargs):
    if remove_mlp:
        model, _, _ = open_clip.create_model_and_transforms(**model_kwargs)
        # TODO: do this cleaner
        image_model, text_model = image_model.model, text_model.model
        model.visual = image_model.visual
        model.token_embedding = text_model.token_embedding
        model.positional_embedding = text_model.positional_embedding
        model.transformer = text_model.transformer
        model.attn_mask = text_model.attn_mask
        model.ln_final = text_model.ln_final
        model.text_projection = text_model.text_projection       
    else:
        model = MLPCLIP(image_model, text_model)

    return model
