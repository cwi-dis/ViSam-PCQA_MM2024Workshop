import copy
from typing import Optional, Any
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm


class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoderLayer_CMA(Module):
    r"""Co-attention Module"""

    def __init__(
        self,
        d_model,
        nhead,
        img_inplanes,
        pc_inplanes,
        cma_planes,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        super(TransformerEncoderLayer_CMA, self).__init__()
        self.img_inplanes = img_inplanes
        self.pc_inplanes = pc_inplanes
        self.cma_planes = cma_planes

        self.linear1 = nn.Linear(img_inplanes, cma_planes)
        self.linear2 = nn.Linear(pc_inplanes, cma_planes)
        # xm: do the batch normalization for the input of the cross modal attention: shape = [B, pc_projection or pc_patch_number, cma_planes]
        self.img_bn = nn.BatchNorm1d(cma_planes)
        self.pc_bn = nn.BatchNorm1d(cma_planes)
        self.pc2img_ca = TransformerEncoderLayer_GA(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.img2pc_ca = TransformerEncoderLayer_GA(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(
        self,
        texture_img: Tensor,
        geometry_img: Tensor,
        texture_pc: Tensor,
        geometry_pc: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer."""

        tex2D_tex3D_attention = self.pc2img_ca(
            texture_img, texture_pc
        )  # xm: img is the src, pc is the guide (target)
        # xm: pc2img_ca and img2pc_ca are the TransformerEncoderLayer_GA. The returned feature is the output of the last layer of the TransformerEncoderLayer_GA
        # hence is feaure map already incldues the attention mechanism (attention first and then pooling or pooling first and then attention??)
        tex3D_tex2D_attention = self.img2pc_ca(
            texture_pc, texture_img
        )  # xm: pc is the src, img is the guide (target)

        tex2D_geo2D_attention = self.pc2img_ca(texture_img, geometry_img)
        geo2D_tex2D_attention = self.img2pc_ca(geometry_img, texture_img)

        tex3D_geo3D_attention = self.pc2img_ca(texture_pc, geometry_pc)
        geo3D_tex3D_attention = self.img2pc_ca(geometry_pc, texture_pc)

        tex2D_geo3D_attention = self.pc2img_ca(texture_img, geometry_pc)
        geo3D_tex2D_attention = self.img2pc_ca(geometry_pc, texture_img)

        tex2D_tex3D_attention = self.pc2img_ca(texture_img, texture_pc)
        tex3D_tex2D_attention = self.img2pc_ca(texture_pc, texture_img)

        geo2D_tex3D_attention = self.pc2img_ca(geometry_img, texture_pc)
        tex3D_geo2D_attention = self.img2pc_ca(texture_pc, geometry_img)

        geo2D_geo3D_attention = self.pc2img_ca(geometry_img, geometry_pc)
        geo3D_geo2D_attention = self.img2pc_ca(geometry_pc, geometry_img)

        tex_img_global = torch.mean(texture_img, dim=1)
        tex_pc_global = torch.mean(texture_pc, dim=1)
        geometry_img_global = torch.mean(geometry_img, dim=1)
        geometry_pc_global = torch.mean(geometry_pc, dim=1)

        # xm is this correct?? directly use the mean of the attentioned feature map as the img_global_attention and pc_global_attention
        tex2D_tex3D_global_attention = torch.mean(tex2D_tex3D_attention, dim=1)
        tex3D_tex2D_global_attention = torch.mean(tex3D_tex2D_attention, dim=1)
        tex2D_geo2D_global_attention = torch.mean(tex2D_geo2D_attention, dim=1)
        geo2D_tex2D_global_attention = torch.mean(geo2D_tex2D_attention, dim=1)
        tex3D_geo3D_global_attention = torch.mean(tex3D_geo3D_attention, dim=1)
        geo3D_tex3D_global_attention = torch.mean(geo3D_tex3D_attention, dim=1)
        tex2D_geo3D_global_attention = torch.mean(tex2D_geo3D_attention, dim=1)
        geo3D_tex2D_global_attention = torch.mean(geo3D_tex2D_attention, dim=1)
        geo2D_tex3D_global_attention = torch.mean(geo2D_tex3D_attention, dim=1)
        tex3D_geo2D_global_attention = torch.mean(tex3D_geo2D_attention, dim=1)
        geo2D_geo3D_global_attention = torch.mean(geo2D_geo3D_attention, dim=1)
        geo3D_geo2D_global_attention = torch.mean(geo3D_geo2D_attention, dim=1)

        return (
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
        )


class TransformerEncoderLayer_GA(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super(TransformerEncoderLayer_GA, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerEncoderLayer_GA, self).__setstate__(state)

    def forward(
        self,
        src: Tensor,
        guide: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        #  src: B,4,1024 image feature
        # guide: B,6,1024 # xm: need jiahua to check again! point cloud feature
        src2 = self.self_attn(
            guide, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        """
        # xm: tensor: B,6,1024 ((N, L, E) L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the embedding dimension ``embed_dim``.)
        # xm: weight matrix: B, 6,4 ((N, L, S) N` is the batch size, `L` is the target sequence length, and:math:`S` is the source sequence length)
        # xm: attention weights averaged across heads of shape
        In here we can make a refeernce: Since L is 6 and L is the target sequence length, so point cloud is the target sequence, and image is the source sequence.
        """

        guide = guide + self.dropout1(src2)
        guide = self.norm1(guide)

        _guide = self.linear2(self.dropout(self.activation(self.linear1(guide))))

        guide = guide + self.dropout2(_guide)
        guide = self.norm2(guide)
        return guide

        # Original Implementation
        # src = src + self.dropout1(src2)
        # src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        # return src


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
