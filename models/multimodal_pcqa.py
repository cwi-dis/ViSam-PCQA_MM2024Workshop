import torch
import torch.nn as nn
from models.backbones import pointnet2, resnet50
from models.transformer import TransformerEncoderLayer_CMA
import torch.nn.functional as F

# from models.Gdn import Gdn  # use this as a activation function
import math


class CMA_fusion(nn.Module):
    def __init__(self, img_inplanes, pc_inplanes, cma_planes=1024, use_local=1):
        super(CMA_fusion, self).__init__()
        self.global_local_encoder = TransformerEncoderLayer_CMA(
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
        self.linear2 = nn.Linear(pc_inplanes, cma_planes)
        # xm: do the batch normalization for the input of the cross modal attention: shape = [B, pc_projection or pc_patch_number, cma_planes]
        self.img_bn = nn.BatchNorm1d(cma_planes)
        self.pc_bn = nn.BatchNorm1d(cma_planes)

    def forward(self, texture_img, geometry_img, texture_pc, geometry_pc):
        # linear mapping and batch normalization
        texture_img = self.linear1(texture_img)
        # change the shape of the input of the cross modal attention img
        texture_img = texture_img.permute(0, 2, 1)
        texture_img = self.img_bn(
            texture_img
        )  # xm: shape = [B, pc_projection, cma_planes]

        # linear mapping and batch normalization for geometry map
        geometry_img = self.linear1(geometry_img)
        geometry_img = geometry_img.permute(0, 2, 1)
        geometry_img = self.img_bn(geometry_img)

        # linear mapping and batch normalization for pc texture
        texture_pc = self.linear2(texture_pc)
        # change the shape of the input of the cross modal attention pc for batch normalization
        texture_pc = texture_pc.permute(0, 2, 1)
        texture_pc = self.pc_bn(texture_pc)

        # linear mapping and batch normalization for pc geometry
        geometry_pc = self.linear2(geometry_pc)
        geometry_pc = geometry_pc.permute(0, 2, 1)
        geometry_pc = self.pc_bn(geometry_pc)

        # change img and pc back to the original shape
        texture_img = texture_img.permute(0, 2, 1)
        geometry_img = geometry_img.permute(0, 2, 1)
        texture_pc = texture_pc.permute(0, 2, 1)
        geometry_pc = geometry_pc.permute(0, 2, 1)

        (
            tex_img_global,
            tex_pc_global,
            geometry_img_global,
            geometry_pc_global,
            tex2D_geo2D_attention,
            tex2D_geo2D_global_attention,
            geo2D_tex2D_attention,
            geo2D_tex2D_global_attention,
            tex3D_geo3D_attention,
            tex3D_geo3D_global_attention,
            geo3D_tex3D_attention,
            geo3D_tex3D_global_attention,
            tex2D_tex3D_attention,
            tex2D_tex3D_global_attention,
            tex3D_tex2D_attention,
            tex3D_tex2D_global_attention,
            tex2D_geo3D_attention,
            tex2D_geo3D_global_attention,
            geo3D_tex2D_attention,
            geo3D_tex2D_global_attention,
            geo2D_tex3D_attention,
            geo2D_tex3D_global_attention,
            tex3D_geo2D_attention,
            tex3D_geo2D_global_attention,
            geo2D_geo3D_attention,
            geo2D_geo3D_global_attention,
            geo3D_geo2D_attention,
            geo3D_geo2D_global_attention,
        ) = self.global_local_encoder(
            texture_img, geometry_img, texture_pc, geometry_pc
        )

        output_local = torch.cat(
            (
                texture_img,
                geometry_img,
                texture_pc,
                geometry_pc,  # orginal
                tex2D_geo2D_attention,
                geo2D_tex2D_attention,
                tex3D_geo3D_attention,
                geo3D_tex3D_attention,
                tex2D_tex3D_attention,
                tex3D_tex2D_attention,
                tex2D_geo3D_attention,
                geo3D_tex2D_attention,
                geo2D_tex3D_attention,
                tex3D_geo2D_attention,
                geo2D_geo3D_attention,
                geo3D_geo2D_attention,
            ),
            dim=1,
        ).mean(
            dim=1
        )  #  keepdim=True # xm: shape = [B, cma_planes] after the mean operation, the shape of the output is [B, cma_planes]

        output_global = torch.stack(  # xm: stack the tensors in a new dimension
            (
                tex_img_global,
                tex_pc_global,
                geometry_img_global,
                geometry_pc_global,
                tex2D_geo2D_global_attention,
                geo2D_tex2D_global_attention,
                tex3D_geo3D_global_attention,
                geo3D_tex3D_global_attention,
                tex2D_tex3D_global_attention,
                tex3D_tex2D_global_attention,
                tex2D_geo3D_global_attention,
                geo3D_tex2D_global_attention,
                geo2D_tex3D_global_attention,
                tex3D_geo2D_global_attention,
                geo2D_geo3D_global_attention,
                geo3D_geo2D_global_attention,
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
    def __init__(self, cma_planes=1024):
        super(QualityRegression, self).__init__()
        self.activation = nn.ReLU()
        self.quality1 = nn.Linear(cma_planes, cma_planes // 2)
        self.quality2 = nn.Linear(cma_planes // 2, 1)

    def forward(self, fusion_output):
        # mos regression # add the relu activation function
        regression_output = self.activation(self.quality1(fusion_output))
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

    def forward(self, fusion_output):
        classification_output = self.activation(self.classifier1(fusion_output))
        classification_output = self.activation(self.classifier2(classification_output))
        classification_output = self.dropout(
            classification_output
        )  # Applying dropout PQA-net add a batchnormal
        classification_output = self.classifier3(classification_output)
        return classification_output


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
        self.depth_backbone = resnet50(
            pretrained=True
        )  # xm: should I still set pretrained = True?
        self.normal_backbone = resnet50(pretrained=True)
        self.pc_position_backbone = pointnet2()
        self.pc_normal_backbone = pointnet2()
        self.pc_texture_backbone = pointnet2()
        self.SharedFusion = CMA_fusion(
            self.img_inplanes, self.pc_inplanes, self.cma_planes, self.use_local
        )  # xm: img_inplanes = 2048 image feature hidden embedding, pc_inplanes = 1024 pc feature hidden embedding, cma_planes = 1024 cross modal attention hidden embedding
        self.MosRegression = QualityRegression()
        self.DistortionClassify = DistortionClassification(num_classes=num_classes)

    def forward(
        self, texture_img, depth_img, normal_img, texture_pc, normal_pc, position_pc
    ):
        # extract features from the projections
        img_size = texture_img.shape  # [B, N, C, Height, Width]

        texture_img = texture_img.view(
            -1, img_size[2], img_size[3], img_size[4]
        )  # xm: reshaping input data to match the expected input format of a neural network model. -1 means the size of that dimension is inferred from the size of the tensor and the remaining dimensions
        # xm shape: [B*N, C, Height, Width]
        texture_img = self.img_backbone(texture_img)  # xm shape: [B*N, HiddenImg, 1, 1]
        texture_img = torch.flatten(texture_img, 1)  # xm shape: [B*N, HiddenImg]
        # average the projection features (xm: why first flatten and then view??)
        texture_img = texture_img.view(
            img_size[0], img_size[1], self.img_inplanes
        )  # xm: [B, N, HiddenImg]

        # shape: [B, HiddenImg] HiddenImg = 2048
        # extract features from depths
        depth = depth_img.view(-1, img_size[2], img_size[3], img_size[4])
        depth = self.depth_backbone(depth)
        depth = torch.flatten(depth, 1)
        depth = depth.view(img_size[0], img_size[1], self.img_inplanes)

        # extract features from normals
        normal = normal_img.view(-1, img_size[2], img_size[3], img_size[4])
        normal = self.normal_backbone(normal)
        normal = torch.flatten(normal, 1)
        normal = normal.view(img_size[0], img_size[1], self.img_inplanes)

        texture_pc_size = (
            texture_pc.shape
        )  # [B, sub-models, Coords + normal channel number?，K]
        texture_pc = texture_pc.view(
            -1, texture_pc_size[2], texture_pc_size[3]
        )  # [B*sub-models (BxM), Coords，K]
        texture_pc = self.pc_texture_backbone(
            texture_pc
        )  # [B*M, HiddenPC] HiddenPC = 1024
        # average the patch features
        texture_pc = texture_pc.view(
            texture_pc_size[0], texture_pc_size[1], self.pc_inplanes
        )  # [B, M, HiddenPC]

        normal_pc_size = normal_pc.shape
        normal_pc = normal_pc.view(-1, normal_pc_size[2], normal_pc_size[3])
        normal_pc = self.pc_normal_backbone(normal_pc)
        normal_pc = normal_pc.view(
            normal_pc_size[0], normal_pc_size[1], self.pc_inplanes
        )

        position_pc_size = position_pc.shape
        position_pc = position_pc.view(-1, position_pc_size[2], position_pc_size[3])
        position_pc = self.pc_position_backbone(position_pc)
        position_pc = position_pc.view(
            position_pc_size[0], position_pc_size[1], self.pc_inplanes
        )
        geometry_img = depth + normal
        geometry_pc = normal_pc + position_pc

        # TODO Before put them into CMA, we need to aline the shape of img_global and geometry_global
        # attention, fusion, and regression and classification
        fusion_output_local_global = self.SharedFusion(
            texture_img,
            geometry_img,
            texture_pc,
            geometry_pc,
        )

        output_regression = self.MosRegression(fusion_output_local_global)
        output_classification = self.DistortionClassify(fusion_output_local_global)
        return output_regression, output_classification
