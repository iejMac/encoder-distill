import torch

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


# TODO: make this function take the 2 separate MLP's and put together a working CLIP
def combine_img_text():
    pass
