# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torchvision

class DinoV2(torch.nn.Module):
    AVAILABLE_MODELS = [
        'dinov2_vits14',
        'dinov2_vitb14',
        'dinov2_vitl14',
        'dinov2_vitg14'
    ]
    
    def __init__(
        self,
        backbone_name="dinov2_vitb14",
        unfreeze_n_blocks=2,
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.unfreeze_n_blocks = unfreeze_n_blocks

        
        # make sure the backbone_name is in the available models
        if self.backbone_name not in self.AVAILABLE_MODELS:
            print(f"Backbone {self.backbone_name} is not recognized!, using dinov2_vitb14")
            self.backbone_name = "dinov2_vitb14"                             
                
        self.dino = torch.hub.load('facebookresearch/dinov2', self.backbone_name)
        
        # freeze all parameters
        for param in self.dino.parameters():
            param.requires_grad = False
        
        if unfreeze_n_blocks > 0:
            # unfreeze the last few blocks
            for block in self.dino.blocks[ -unfreeze_n_blocks : ]:
                for param in block.parameters():
                    param.requires_grad = True
        
        self.out_channels = self.dino.embed_dim
        
    @property
    def patch_size(self):
        return self.dino.patch_embed.patch_size[0]  # Assuming square patches
    
    def forward(self, x):
        B, _, H, W = x.shape
        if self.unfreeze_n_blocks > 0:
            # No need to compute gradients for frozen layers
            with torch.no_grad():
                x = self.dino.prepare_tokens_with_masks(x)
                for blk in self.dino.blocks[ : -self.unfreeze_n_blocks]:
                    x = blk(x)

            # Last blocks are trained
            for blk in self.dino.blocks[-self.unfreeze_n_blocks : ]:
                x = blk(x)
        else:
            x = self.dino.prepare_tokens_with_masks(x)
            for blk in self.dino.blocks:
                x = blk(x)

        cls = x[:, 0]  # [B, C]  
        x = x[:, 1:] # remove the [CLS] token
        
        return x, cls  # return the features and the [CLS] token
    
class DinoV3(torch.nn.Module):
    AVAILABLE_MODELS = [
        'dinov3_vits16',
        'dinov3_vitb16',
        'dinov3_vitl16',
        'dinov3_vitg16'
    ]
    REPO_DIR = "/home/dragon_llm/Backbone/DINOV3/dinov3"
    WEIGHT_DIR = "/home/dragon_llm/Backbone/DINOV3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

    def __init__(
        self,
        backbone_name="dinov3_vitb16",
        unfreeze_n_blocks=2,
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.unfreeze_n_blocks = unfreeze_n_blocks
        
        # make sure the backbone_name is in the available models
        if self.backbone_name not in self.AVAILABLE_MODELS:
            print(f"Backbone {self.backbone_name} is not recognized!, using dinov3_vitb16")
            self.backbone_name = "dinov3_vitb16"

        self.dino = torch.hub.load('facebookresearch/dinov3', self.backbone_name)

        # freeze all parameters
        for param in self.dino.parameters():
            param.requires_grad = False
        
        if unfreeze_n_blocks > 0:
            # unfreeze the last few blocks
            for block in self.dino.blocks[ -unfreeze_n_blocks : ]:
                for param in block.parameters():
                    param.requires_grad = True
        
        self.out_channels = self.dino.embed_dim
        
    @property
    def patch_size(self):
        return self.dino.patch_embed.patch_size[0]  # Assuming square patches
    
    def forward(self, x):
        # No need to compute gradients for frozen layers
        x, rope = self.dino.prepare_tokens_with_masks(x)
        H, W = rope
        if self.unfreeze_n_blocks > 0:
            with torch.no_grad():
                for blk in self.dino.blocks[ : -self.unfreeze_n_blocks]:
                    rope_sincos = self.dino.rope_embed(H=H, W=W) 
                    x = blk(x, rope_sincos)

            # Last blocks are trained
            for blk in self.dino.blocks[-self.unfreeze_n_blocks : ]:
                rope_sincos = self.dino.rope_embed(H=H, W=W)
                x = blk(x, rope_sincos)
        else:
            for blk in self.dino.blocks:
                rope_sincos = self.dino.rope_embed(H=H, W=W)
                x = blk(x, rope_sincos)
    
        cls = x[:, 0]  # [B, C]  
        x = x[:, 5:] # remove the [CLS] token
        
        # features = self.dino.forward_features(x)[0]
        # cls = features["x_norm_clstoken"]  # [B, C]
        # x = features["x_norm_patchtokens"]  # [B, N, C]

        return x, cls

    
class ResNet(nn.Module):
    AVAILABLE_MODELS = {
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34,
        "resnet50": torchvision.models.resnet50,
        "resnet101": torchvision.models.resnet101,
        "resnet152": torchvision.models.resnet152,
        "resnext50": torchvision.models.resnext50_32x4d,
    }

    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        unfreeze_n_blocks=1,
        crop_last_block=True,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.crop_last_block = crop_last_block

        if backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Backbone {backbone_name} is not recognized!" 
                             f"Supported backbones are: {list(self.AVAILABLE_MODELS.keys())}")

        # Load the model
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = self.AVAILABLE_MODELS[backbone_name](weights=weights)

        # Create backbone with only the necessary layers
        self.net = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            *([] if crop_last_block else [resnet.layer4]),
        )

        # Handle trainable/frozen layers
        nb_layers = len(self.net)
        assert (
            isinstance(unfreeze_n_blocks, int) and 0 <= unfreeze_n_blocks <= nb_layers
        ), f"unfreeze_n_blocks must be an integer between 0 and {nb_layers} (inclusive)"

        if pretrained:
            # Freeze required layers
            for layer in self.net[:nb_layers - unfreeze_n_blocks]:
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            if self.unfreeze_n_blocks > 0:
                print("Warning: unfreeze_n_blocks is ignored when pretrained=False. Setting it to 0.")
                self.unfreeze_n_blocks = 0

        # Output channels
        if backbone_name in ["resnet18", "resnet34"]:
            self.out_channels = resnet.layer3[-1].conv2.out_channels
        else:
            self.out_channels = resnet.layer3[-1].conv3.out_channels

    def forward(self, x):
        return self.net(x)