# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified by Jian Ding from: https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
# Modified by Heeseong Shin from: https://github.com/dingjiansw101/ZegFormer/blob/main/mask_former/mask_former_model.py
import fvcore.nn.weight_init as weight_init
import torch

from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

from detectron2.config import configurable
from detectron2.layers import Conv2d
# from .Decoder_or import Decoder_or
from .Decoder import Decoder
from .Decoder_fast import Decoder_fast

# from .Decoder_local import Decoder_local
from FGA_seg.third_party import clip
from FGA_seg.third_party import imagenet_templates
from .P2Tformer import P2Tformer
from .projector import Projector
import numpy as np
import open_clip


# Function to visualize and save tensor slices
def visualize_and_save(tensor, name, save_dir=""):

    B = tensor.shape[0]
    # Function to visualize and save tensor slices
    for bb in range(B):
        tensor_bb = tensor[bb].cpu().numpy()
        num_slices = tensor_bb.shape[0]
        for i in range(num_slices):
            fig, axes = plt.subplots(1, 1, figsize=(25, 25))
            fig.suptitle(f"{name} Visualization")
            ax = axes
            ax.imshow(tensor_bb[i].squeeze(), cmap="viridis")
            ax.axis("off")
            # Save the visualization
            save_path = f"{save_dir}/{bb}/{name}_visualization_slice_{i}.png" if save_dir else f"{name}_visualization.png"
            plt.savefig(save_path)
            plt.close()
            print(f"Saved {name} visualization to {save_path}")


class FGA_seg_Predictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        train_class_json: str,
        test_class_json: str,
        clip_pretrained: str,
        prompt_ensemble_type: str,
        text_guidance_dim: int,
        text_guidance_proj_dim: int,
        appearance_guidance_dim: int,
        appearance_guidance_proj_dim: int,
        prompt_depth: int,
        prompt_length: int,
        decoder_dims: list,
        decoder_guidance_dims: list,
        decoder_guidance_proj_dims: list,
        num_heads: int,
        num_layers: tuple,
        hidden_dims: tuple,
        pooling_sizes: tuple,
        feature_resolution: tuple,
        window_sizes: tuple,
        attention_type: str,
        decoder_mod: str,
        fusion_mod: str,
        P2T_layer: int,
        kernel_size: int,
    ):
        """
        Args:
            
        """
        super().__init__()
        
        import json
        # use class_texts in train_forward, and test_class_texts in test_forward
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts
        device = "cuda" if torch.cuda.is_available() else "cpu"
  
        self.tokenizer = None
        if clip_pretrained == "ViT-G" or clip_pretrained == "ViT-H":
            # for OpenCLIP models
            name, pretrain = ('ViT-H-14', 'laion2b_s32b_b79k') if clip_pretrained == 'ViT-H' else ('ViT-bigG-14', 'laion2b_s39b_b160k')
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                name, 
                pretrained=pretrain, 
                device=device, 
                force_image_size=336,)
        
            self.tokenizer = open_clip.get_tokenizer(name)
        else:
            # for OpenAI models
            clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)
    
        self.prompt_ensemble_type = prompt_ensemble_type        

        if self.prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            prompt_templates = ['A photo of a {} in the scene',]
        else:
            raise NotImplementedError
        
        self.prompt_templates = prompt_templates
        
        # Text Projector B:512 L:712
        if clip_pretrained == "ViT-B/16":
            self.DCDT = P2Tformer(d_model=512, nhead=8, N=P2T_layer)
            self.proj = Projector(word_dim=512, vision_dim= 512//2)
        else:
            self.DCDT = P2Tformer(d_model=768, nhead=8, N=P2T_layer)
            self.proj = Projector(word_dim=768, vision_dim= 768//2)

        self.clip_model = clip_model.float()
        self.clip_preprocess = clip_preprocess
        self.decoder_mod = decoder_mod
        if self.decoder_mod == "fast":
            transformer = Decoder_fast(
                text_guidance_dim=text_guidance_dim,
                text_guidance_proj_dim=text_guidance_proj_dim,
                appearance_guidance_dim=appearance_guidance_dim,
                appearance_guidance_proj_dim=appearance_guidance_proj_dim,
                decoder_dims=decoder_dims,
                decoder_guidance_dims=decoder_guidance_dims,
                decoder_guidance_proj_dims=decoder_guidance_proj_dims,
                num_layers=num_layers,
                nheads=num_heads, 
                hidden_dim=hidden_dims,
                pooling_size=pooling_sizes,
                feature_resolution=feature_resolution,
                window_size=window_sizes,
                attention_type=attention_type,
                prompt_channel=len(prompt_templates),
                clip_pretrained=clip_pretrained,
                fusion_mod=fusion_mod,
                kernel_size=kernel_size)

        elif self.decoder_mod == "no_fast":
            transformer = Decoder(
                text_guidance_dim=text_guidance_dim,
                text_guidance_proj_dim=text_guidance_proj_dim,
                appearance_guidance_dim=appearance_guidance_dim,
                appearance_guidance_proj_dim=appearance_guidance_proj_dim,
                decoder_dims=decoder_dims,
                decoder_guidance_dims=decoder_guidance_dims,
                decoder_guidance_proj_dims=decoder_guidance_proj_dims,
                num_layers=num_layers,
                nheads=num_heads, 
                hidden_dim=hidden_dims,
                pooling_size=pooling_sizes,
                feature_resolution=feature_resolution,
                window_size=window_sizes,
                attention_type=attention_type,
                prompt_channel=len(prompt_templates),
                clip_pretrained=clip_pretrained,
                fusion_mod=fusion_mod,
                kernel_size=kernel_size
                )
        else :
            transformer = Decoder_or(
                text_guidance_dim=text_guidance_dim,
                text_guidance_proj_dim=text_guidance_proj_dim,
                appearance_guidance_dim=appearance_guidance_dim,
                appearance_guidance_proj_dim=appearance_guidance_proj_dim,
                decoder_dims=decoder_dims,
                decoder_guidance_dims=decoder_guidance_dims,
                decoder_guidance_proj_dims=decoder_guidance_proj_dims,
                num_layers=num_layers,
                nheads=num_heads, 
                hidden_dim=hidden_dims,
                pooling_size=pooling_sizes,
                feature_resolution=feature_resolution,
                window_size=window_sizes,
                attention_type=attention_type,
                prompt_channel=len(prompt_templates),
                clip_pretrained=clip_pretrained,
                fusion_mod=fusion_mod,
                kernel_size=kernel_size
                )
        self.transformer = transformer
        
        self.tokens = None
        self.cache = None

    @classmethod
    def from_config(cls, cfg):#, in_channels, mask_classification):
        ret = {}

        ret["train_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON
        ret["test_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON
        ret["clip_pretrained"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
        ret["prompt_ensemble_type"] = cfg.MODEL.PROMPT_ENSEMBLE_TYPE

        # Decoder parameters:
        ret["text_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM
        ret["text_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM
        ret["appearance_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM
        ret["appearance_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM

        ret["decoder_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS
        ret["decoder_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS
        ret["decoder_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS

        ret["prompt_depth"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH
        ret["prompt_length"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH

        ret["num_layers"] = cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS
        ret["num_heads"] = cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS
        ret["hidden_dims"] = cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS
        ret["pooling_sizes"] = cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES
        ret["feature_resolution"] = cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION
        ret["window_sizes"] = cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES
        ret["attention_type"] = cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE
        ret["decoder_mod"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_MOD
        ret["fusion_mod"] = cfg.MODEL.SEM_SEG_HEAD.FUSION_MOD
        ret["P2T_layer"] = cfg.MODEL.SEM_SEG_HEAD.P2T_LAYER
        ret["kernel_size"] = cfg.MODEL.SEM_SEG_HEAD.KERNEL_SIZE
        return ret

    def forward(self, imgs_feats, vis_guidance, prompt=None, gt_cls=None):
        # ------ preparing vis_guidance
        vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]
        # ------ preparing text embedding
        text = self.class_texts if self.training else self.test_class_texts
        text = [text[c] for c in gt_cls] if gt_cls is not None else text
        text = self.get_text_embeds(text, self.prompt_templates, self.clip_model, prompt)
        # ------ PTformer 
        out_text = self.DCDT(imgs_feats, text)
        align_out = self.proj(imgs_feats, out_text)         # 0
        print("align_out_align_out.shape",align_out.shape)
        # align_out = F.interpolate(align_out, size=[24,24], mode='bilinear', align_corners=False, )
        print("align_out_align_out.shape",align_out.shape)
        visualize_and_save(align_out, "align_out", "/data01/lby/FGA-Seg/visulization/align_out") 

        # ------ aggregation 
        text_feats = out_text.unsqueeze(0).transpose(2,0) 
        # text_feats = text.repeat(imgs_feats.shape[0], 1, 1, 1)

        if self.training:
            out, mask_aux0, mask_aux1, mask_aux2  = self.transformer(imgs_feats, text_feats, vis)
            # align_out = self.proj(imgs_feats, out_text)         # 0
            return out, mask_aux0, mask_aux1, mask_aux2, align_out
        else:
            out, _, _, _ = self.transformer(imgs_feats, text_feats, vis)
            # align_out = self.proj(imgs_feats, out_text)  
            
            return out

    # @torch.no_grad()
    # def class_embeddings(self, classnames, templates, clip_model):
    #     zeroshot_weights = []
    #     for classname in classnames:
    #         if ', ' in classname:
    #             classname_splits = classname.split(', ')
    #             texts = []
    #             for template in templates:
    #                 for cls_split in classname_splits:
    #                     texts.append(template.format(cls_split))
    #         else:
    #             texts = [template.format(classname) for template in templates]  # format with class
    #         if self.tokenizer is not None:
    #             texts = self.tokenizer(texts).cuda()
    #         else: 
    #             texts = clip.tokenize(texts).cuda()
    #         class_embeddings = clip_model.encode_text(texts)
    #         class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
    #         if len(templates) != class_embeddings.shape[0]:
    #             class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
    #             class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
    #         class_embedding = class_embeddings
    #         zeroshot_weights.append(class_embedding)
    #     zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    #     return zeroshot_weights
    
    def get_text_embeds(self, classnames, templates, clip_model, prompt=None):
        if self.cache is not None and not self.training:
            return self.cache
        
        if self.tokens is None or prompt is not None:
            tokens = []
            for classname in classnames:
                if ', ' in classname:
                    classname_splits = classname.split(', ')
                    texts = [template.format(classname_splits[0]) for template in templates]
                else:
                    texts = [template.format(classname) for template in templates]  # format with class
                if self.tokenizer is not None:
                    texts = self.tokenizer(texts).cuda()
                else: 
                    texts = clip.tokenize(texts).cuda()
                tokens.append(texts)
            tokens = torch.stack(tokens, dim=0).squeeze(1)
            if prompt is None:
                self.tokens = tokens
        elif self.tokens is not None and prompt is None:
            tokens = self.tokens

        class_embeddings = clip_model.encode_text(tokens, prompt)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        
        class_embeddings = class_embeddings.unsqueeze(1)
        
        if not self.training:
            self.cache = class_embeddings
            
        return class_embeddings