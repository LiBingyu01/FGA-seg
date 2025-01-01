import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional
from .position_encoding import PositionEmbeddingSine

class ShortCut_CrossAttention(nn.Module):

    def __init__(self, d_model, nhead):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)
        self.norm = nn.LayerNorm(d_model)
        self.activation = F.relu
        self._reset_parameters()
        self.MLP = nn.Linear(d_model, d_model)
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    def forward(self, tgt, memory, flat:str,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,):
        if flat=="text":
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), # text
                                    key=self.with_pos_embed(memory, pos), # vision
                                    value=memory, attn_mask=memory_mask, # vision
                                    key_padding_mask=memory_key_padding_mask)[0]  # 得到的是视觉信息
        elif flat=="vision":
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, pos), # vision
                        key=self.with_pos_embed(memory, query_pos), # text
                        value=memory, attn_mask=memory_mask, # text
                        key_padding_mask=memory_key_padding_mask)[0]  # 得到的是视觉信息

        tgt = tgt + self.MLP(tgt2)
        tgt = self.norm(tgt)
        return tgt
    

class P2Tformer(nn.Module):
    def __init__(self, d_model, nhead, N=1):
        super().__init__()
        self.pe_layer = PositionEmbeddingSine(d_model//2, normalize=True)
        self.cross_atten_text = nn.ModuleList([ShortCut_CrossAttention(d_model = d_model, nhead = nhead) for _ in range(N)])
        self.gamma_text = nn.Parameter(torch.ones(d_model) * 1e-1)
    def forward(self, imgs_feat, text_classifier, ): 
        batch, channel, h, w = imgs_feat.shape
        # text特征
        text_diff = text_classifier.repeat(1,imgs_feat.shape[0],1)
        text_out = text_diff.clone()
        # pos embedding
        pos = self.pe_layer(imgs_feat, None).flatten(2).permute(2, 0, 1)  # hw * b * c
        # img 特征 
        imgs_diff = imgs_feat.flatten(2).permute(2, 0, 1)  # hw * b * c
        # cross attn
        for layer_text in self.cross_atten_text:
            text_out = layer_text(text_out, imgs_diff, flat="text", memory_mask=None, memory_key_padding_mask=None, pos=pos, query_pos=None)
        text_embeddings = text_diff + self.gamma_text * text_out
        return text_embeddings

