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
    


if __name__ == "__main__":
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PIL import Image
    import torch.nn.functional as F
    # 配置
    image_path = "/home/dragon_llm/daikin/daikin_ws/src/boq/data/train/gsv-cities/Images/Bangkok/Bangkok_0000001_2019_08_218_13.7154238856932_100.4848777732716_alIIqPQFyy4iZ0UEaeTEVw.jpg"  # 修改为你的图片路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = DinoV3(backbone_name="dinov3_vitb16", unfreeze_n_blocks=0)
    model.to(device)
    model.eval()
    
    # 加载和预处理图片
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (224, 224))
    
    # 转换为tensor并归一化
    image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1) / 255.0
    # 标准化 (ImageNet标准)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # 前向传播
    with torch.no_grad():
        x, cls = model(image_tensor)
    
    # 计算余弦相似度
    # x: [1, N, C], cls: [1, C]
    cls_expanded = cls.unsqueeze(1)  # [1, 1, C]
    
    # 计算余弦相似度
    similarity = F.cosine_similarity(x, cls_expanded, dim=2)  # [1, N]
    similarity = similarity.squeeze(0).cpu().numpy()  # [N]
    
    # 将相似度重新整形为空间维度 (14x14 for 224x224 input with patch_size=16)
    patch_size = model.patch_size
    h_patches = w_patches = 224 // patch_size
    similarity_map = similarity.reshape(h_patches, w_patches)
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    # 原图 - 添加网格分割
    plt.subplot(1, 2, 1)
    plt.imshow(image_resized)
    
    # 添加网格线显示patch分割
    for i in range(1, h_patches):
        plt.axhline(y=i*patch_size-0.5, color='white', linestyle='-', linewidth=1, alpha=0.7)
    for j in range(1, w_patches):
        plt.axvline(x=j*patch_size-0.5, color='white', linestyle='-', linewidth=1, alpha=0.7)
    
    plt.title(f"Original Image (224x224) with {patch_size}x{patch_size} Patches")
    plt.axis('off')
    
    # 热力图
    plt.subplot(1, 2, 2)
    sns.heatmap(similarity_map, cmap='viridis', annot=False, cbar=True)
    plt.title("CLS-Patch Cosine Similarity Heatmap")
    
    plt.tight_layout()
    plt.show()
    
    print(f"Image resized to: 224x224")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Number of patches: {h_patches}x{w_patches}")
    print(f"Similarity map shape: {similarity_map.shape}")
    print(f"Similarity range: [{similarity_map.min():.3f}, {similarity_map.max():.3f}]")
