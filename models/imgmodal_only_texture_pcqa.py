import torch
import torch.nn as nn
from models.backbones import resnet50
from models.transformer_img_only_texture import (
    TransformerEncoderLayer_CMA_cls,
)
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from torchsummary import summary  # xm: for the summary of the model
import os
import cv2
import numpy as np
from torchvision import transforms, utils, models
from tqdm import tqdm
from utils.data_process import preprocess_img, postprocess_img
from PIL import Image
import numpy as np


def visualize_feature_map(intermediate_outputs, model_with_intermediate):
    # Visualize the intermediate outputs (feature maps)
    fig, axes = plt.subplots(1, len(intermediate_outputs) + 1, figsize=(15, 3))

    for i, output in enumerate(intermediate_outputs):
        output_array = output.squeeze().detach().cpu().numpy()
        axes[i].imshow(
            output_array[0], cmap="viridis"
        )  # Assuming you have a batch size of 1
        axes[i].set_title(f"Layer {i + 1}")
        axes[i].axis("off")

    # Visualize the final output, it is a feature map not a weight matrix
    final_output_array = model_with_intermediate.final_fc.weight.detach().cpu().numpy()
    axes[-1].imshow(
        final_output_array[0].reshape(1, -1), cmap="viridis"
    )  # Assuming you have a batch size of 1
    axes[-1].set_title("Final Output (FC)")
    axes[-1].axis("off")

    plt.show()


class CMA_fusion(nn.Module):
    def __init__(self, img_inplanes, pc_inplanes, cma_planes=1024, use_local=1):
        super(CMA_fusion, self).__init__()
        # added by xuemei
        self.cls_token = nn.Parameter(torch.randn(1, 1, cma_planes))
        self.global_encoder = TransformerEncoderLayer_CMA_cls(
            d_model=cma_planes,
            nhead=8,
            img_inplanes=img_inplanes,
            pc_inplanes=pc_inplanes,
            cma_planes=cma_planes,
            dim_feedforward=2048,
            dropout=0.1,
        )
        self.use_local = use_local
        self.linear1 = nn.Linear(img_inplanes, cma_planes)
        # xm: do the batch normalization for the input of the cross modal attention: shape = [B, pc_projection or pc_patch_number, cma_planes]
        self.img_bn = nn.BatchNorm1d(cma_planes)

    def forward(
        self,
        texture_img,
    ):
        # linear mapping and batch normalization
        texture_img = self.linear1(texture_img)
        # change the shape of the input of the cross modal attention img
        texture_img = texture_img.permute(0, 2, 1)
        texture_img = self.img_bn(
            texture_img
        )  # xm: shape = [B, pc_projection, cma_planes]

        # change img and pc back to the original shape
        texture_img = texture_img.permute(0, 2, 1)

        cls_tokens = self.cls_token.expand(texture_img.shape[0], -1, -1)  # [b, 1, 1024]
        texture_img_token = torch.cat((cls_tokens, texture_img), dim=1)  # [b, 7,1024]

        (
            tex_img_global,
            tex2D_global_attention,
        ) = self.global_encoder(
            texture_img_token,
        )

        output_local = torch.cat(
            (texture_img,),
            dim=1,
        ).mean(
            dim=1
        )  #  keepdim=True # xm: shape = [B, cma_planes] after the mean operation, the shape of the output is [B, cma_planes]

        output_global = torch.stack(  # xm: stack the tensors in a new dimension
            (
                tex_img_global,
                tex2D_global_attention,
            ),
            dim=-1,
        ).mean(dim=-1)

        # output_global = output_global.squeeze(0)
        if self.use_local:
            output = output_local + output_global
        else:
            output = output_global
        # output = torch.cat((output_local, output_global), dim=1)

        return output


class QualityRegression(nn.Module):
    def __init__(self, cma_planes=(1024 + 256)):
        super(QualityRegression, self).__init__()
        self.activation = nn.ReLU()
        self.quality1 = nn.Linear(cma_planes, cma_planes // 2)
        self.quality2 = nn.Linear(cma_planes // 2, 1)
        self.adapool = nn.AdaptiveAvgPool2d(1)

    def forward(self, fusion_output, vs_output):
        vs_output = vs_output.mean(1)
        vs_output = self.adapool(vs_output)  # xm: shape = [B, 256, 1, 1]
        vs_output = vs_output.squeeze(3).squeeze(2)  # xm: shape = [B, 256]
        total_output = torch.cat((fusion_output, vs_output), dim=1)
        # mos regression # add the relu activation function
        regression_output = self.activation(self.quality1(total_output))
        regression_output = self.quality2(regression_output)
        return regression_output


class DistortionClassification(nn.Module):
    def __init__(self, cma_planes=1024, num_classes=None, dropout_prob=0.5):
        super(DistortionClassification, self).__init__()

        self.classifier1 = nn.Linear(cma_planes, cma_planes // 2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(
            p=dropout_prob
        )  # Dropout layer with the specified dropout probability
        self.classifier2 = nn.Linear(cma_planes // 2, cma_planes // 4)
        self.classifier3 = nn.Linear(cma_planes // 4, num_classes)

        # self.classifier21 = nn.Linear(cma_planes, cma_planes // 2)
        # self.classifier22 = nn.Linear(cma_planes // 2, cma_planes // 4)
        # self.classifier23 = nn.Linear(cma_planes // 4, num_classes)

    def forward(self, fusion_output):
        classification_output = self.activation(self.classifier1(fusion_output))
        classification_output = self.activation(self.classifier2(classification_output))
        classification_output = self.dropout(
            classification_output
        )  # Applying dropout PQA-net add a batchnormal
        classification_output = self.classifier3(classification_output)

        # classification_output = self.activation(self.classifier21(fusion_output))
        # classification_output = self.activation(self.classifier22(classification_output))
        # classification_output = self.dropout(
        #     classification_output
        # )  # Applying dropout PQA-net add a batchnormal
        # classification_output = self.classifier23(classification_output)
        return classification_output


class ResNet50WithIntermediate(nn.Module):
    def __init__(self, model):
        super(ResNet50WithIntermediate, self).__init__()
        self.features = nn.ModuleList(
            [
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            ]
        )
        self.avgpool = model.avgpool  # Add avgpool layer
        self.intermediate_outputs = []

    def forward(self, x):
        self.intermediate_outputs = []
        for layer in self.features:
            x = layer(x)
            # save only the output of layer1
            if layer == self.features[4]:
                self.intermediate_outputs.append(x.clone())

        x = self.avgpool(
            x
        )  # Apply average pooling to the output of the last convolutional block
        self.intermediate_outputs.append(x.clone())  # Capture the final feature map
        # return the self.intermediate_outputs
        return self.intermediate_outputs[0], x
        # return x


def save_tensor_to_numpy(tensor, dtype=np.uint8):
    texture_img_vs_ = tensor.cpu().numpy()
    texture_img_vs_ = (texture_img_vs_ - np.min(texture_img_vs_)) / (
        np.max(texture_img_vs_) - np.min(texture_img_vs_)
    )
    texture_img_vs_ = (texture_img_vs_ * 255).astype(np.uint8)
    texture_img_vs_ = np.transpose(texture_img_vs_, (1, 2, 0))
    texture_img_vs_ = Image.fromarray(texture_img_vs_)
    return texture_img_vs_


class MM_PCQAnet(nn.Module):
    def __init__(self, num_classes, args):
        super(
            MM_PCQAnet,
            self,
        ).__init__()  # inherits all the functionalities and attributes of the nn.Module
        self.img_inplanes = args.img_inplanes
        self.pc_inplanes = args.pc_inplanes
        self.cma_planes = args.cma_planes
        self.use_local = args.use_local

        self.img_backbone = resnet50(pretrained=True)
        self.depth_backbone = resnet50(pretrained=True)
        self.SharedFusion = CMA_fusion(
            self.img_inplanes, self.pc_inplanes, self.cma_planes, self.use_local
        )  # xm: img_inplanes = 2048 image feature hidden embedding, pc_inplanes = 1024 pc feature hidden embedding, cma_planes = 1024 cross modal attention hidden embedding
        self.MosRegression = QualityRegression()
        self.DistortionClassify = DistortionClassification(num_classes=num_classes)

    def forward(
        self,
        texture_img,
        depth_img,
        vs_img,
    ):
        # Create an instance of ResNet50WithIntermediate
        # print(vs_img.shape)
        # covert the depth_img to a grayscale image
        depth_img_for_vs = torch.mean(depth_img, dim=2, keepdim=False)
        # using the depth_img to mask the vs_img
        vs_img = torch.mul(
            vs_img, depth_img_for_vs
        )  # xm: vs_img shape = [B, 1, 224, 224] depth_img shape = [B, N, 224, 224]
        # extract features from the projections
        img_size = texture_img.shape  # [B, N, C, Height, Width]
        texture = texture_img.view(-1, img_size[2], img_size[3], img_size[4])
        # texture = self.img_backbone(texture)
        model_with_intermediate = ResNet50WithIntermediate(self.img_backbone)
        # print(model_with_intermediate)
        # print(self.img_backbone)
        # summary(model_with_intermediate, (3, 224, 224))
        # summary(self.img_backbone, (3, 224, 224))
        output_of_layer1, texture = model_with_intermediate(texture)
        # reshape the output_layer1 to the same shape as the texture_img
        (
            B,
            featuremap_Channels,
            featuremap_Height,
            featuremap_Width,
        ) = output_of_layer1.shape

        output_of_layer1 = output_of_layer1.view(
            img_size[0],
            img_size[1],
            featuremap_Channels,
            featuremap_Height,
            featuremap_Width,
        )

        # # summary(self.img_backbone, (3, 224, 224))
        vs_img = transforms.functional.resize(
            vs_img, (featuremap_Height, featuremap_Width)
        )

        # Reshape A to [2, 6, 1, 56, 56] to make it compatible with B for broadcasting
        vs_img_reshaped = vs_img.unsqueeze(2)
        # mutiply the texture_img_for_vs with vs_img
        texture_with_vs = torch.mul(
            vs_img_reshaped, output_of_layer1
        )  # Element-wise multiplication
        # print("Result Shape:", texture_with_vs.shape)
        # # visualize_feature_map(intermediate_outputs, model_with_intermediate)

        # # average the projection features (xm: why first flatten and then view??)
        # layer5_feature = torch.flatten(layer5_feature, 1)  # xm shape: [B*N, HiddenImg]
        # texture_img = layer5_feature.view(
        #     img_size[0], img_size[1], self.img_inplanes
        # )  # xm: [B, N, HiddenImg]
        texture = torch.flatten(texture, 1)
        texture = texture.view(img_size[0], img_size[1], self.img_inplanes)

        # shape: [B, HiddenImg] HiddenImg = 2048
        # extract features from depths
        depth = depth_img.view(-1, img_size[2], img_size[3], img_size[4])
        depth = self.depth_backbone(depth)
        depth = torch.flatten(depth, 1)
        depth = depth.view(img_size[0], img_size[1], self.img_inplanes)

        # TODO Before put them into CMA, we need to aline the shape of img_global and geometry_global
        # attention, fusion, and regression and classification
        fusion_output_local_global = self.SharedFusion(texture)
        output_regression = self.MosRegression(
            fusion_output_local_global, texture_with_vs
        )
        output_classification = self.DistortionClassify(fusion_output_local_global)
        return output_regression, output_classification
