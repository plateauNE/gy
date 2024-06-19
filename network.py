"""
(c) Research Group CAMMA, University of Strasbourg, IHU Strasbourg, France
Website: http://camma.u-strasbg.fr
"""

import os
import numpy as np
from typing import Tuple
from collections import OrderedDict
from einops import repeat, rearrange

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as basemodels
from triplet_text_label import triplet_text_label
from clip import clip
import math
OUT_HEIGHT = 8
OUT_WIDTH = 14
# %% Rendezvous-in-time (RiT)


class RiT(nn.Module):
    def __init__(self, clip_model, basename="resnet18", num_tool=6, num_verb=10, num_target=15, num_triplet=100, layer_size=8, num_heads=4, d_model=128, hr_output=False, use_ln=False, m=3):
        super(RiT, self).__init__()
        self.encoder = Encoder(basename, num_tool, num_verb,
                               num_target, num_triplet, hr_output=hr_output, m=m)
        self.decoder = Decoder(layer_size, d_model,
                               num_heads, num_triplet, use_ln=use_ln, m=m)
        self.gy = gy_clip(clip_model, m=m)
        self.fc_joint_a = nn.Linear(200, 100)

    def forward(self, inputs):
        enc_i, enc_v, enc_t, enc_ivt, high_x = self.encoder(inputs)
        dec_ivt, cam_ivt = self.decoder(enc_i, enc_v, enc_t, enc_ivt)
        gy_ivt = self.gy(cam_ivt, high_x)
        out_ivt = self.fc_joint_a(torch.cat((dec_ivt, gy_ivt), dim=1))
        return enc_i, enc_v, enc_t, out_ivt


class gy_clip(nn.Module):  # 只是得到promte的word embedding
    def __init__(self, clip_model, m):
        super().__init__()
        self.prompt_learner = PromptLearner(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.dim = 512
        self.fc1 = nn.Linear(112, self.dim)
        self.attn = nn.MultiheadAttention(self.dim, 8)
        self.attn1 = nn.MultiheadAttention(self.dim, 8)
        # 创建一个transformer encoder结构，去生成step。先建EncoderLayer再建Encoder
        self.pos_embed = nn.Parameter(torch.zeros(1, 112, 512))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # 初始化位置编码
        enc_layer = nn.TransformerEncoderLayer(self.dim, nhead=8)
        self.transformer_enc = nn.TransformerEncoder(
            enc_layer, num_layers=4, norm=nn.LayerNorm(self.dim))
        self.fc2 = nn.Linear(512 * 112, 100 * 100)
        self.soft = nn.Softmax(dim=1)
        self.conv1 = nn.Conv2d(in_channels=m+1, out_channels=1, kernel_size=1)

        self.amp = nn.AdaptiveAvgPool1d(1)
        self.gcn1 = GraphConvolution(self.dim, 2048)
        self.gcn2 = GraphConvolution(2048, 2048)
        self.gcn3 = GraphConvolution(2048, self.dim)
        self.relu = nn.LeakyReLU(0.2)
        self.mlp = nn.Linear(100, 100)
        self.m = m

    def forward(self, x, high_x):
        bt, c, h, w = x.shape
        b = int(bt/(self.m + 1))
        prompts = self.prompt_learner()  # 学习到的promte
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features, relation = self.attn(text_features, text_features, text_features)
        # relation = relation.unsqueeze(0).repeat(bt, 1, 1)  # [b,100,100]

        # 处理图像特征得到手术阶段
        bh, ch, hh, wh = high_x.shape
        high_x = high_x.permute(2, 3, 0, 1).view(-1, bh, ch)
        high_x = self.transformer_enc(high_x)
        high_x = high_x.permute(1, 2, 0).view(-1, self.m+1, ch, hh*wh)
        high_x = self.conv1(high_x).squeeze(1)  # 用1x1conv沿时间维度合并
        high_x = high_x.view(b, -1)
        high_x = self.fc2(high_x)
        high_x = high_x.view(b, 100, 100)

        # 根据high_x得到动态relation
        relation = relation.unsqueeze(0).repeat(b, 1, 1)  # [b,100,100]
        attention = torch.matmul(high_x, relation.transpose(-2, -1)) / \
            torch.sqrt(torch.tensor(100, dtype=torch.float32))
        attention = self.soft(attention)
        relation = torch.matmul(attention, relation)
        relation = relation.unsqueeze(1).repeat(1, self.m + 1, 1, 1)
        relation = relation.view(-1, 100, 100)

        x = x.view(bt, c, -1)
        x = self.fc1(x)
        identity = x

        x = self.gcn1(x, relation)
        x = self.relu(x)
        x = self.gcn2(x, relation)
        x = self.relu(x)
        x = self.gcn3(x, relation)
        x += identity
        x = self.amp(x).squeeze(-1)
        x = self.mlp(x)
        return x


class PromptLearner(nn.Module):  # 只是得到promte的word embedding
    def __init__(self, clip_model):
        super().__init__()
        triplet_label = triplet_text_label
        n_cls = len(triplet_label.keys())
        # n_ctx = cfg.n_ctx
        # ctx_init = cfg.ctx_init.replace("_", " ")  # a photo of a
        ctx_init = "a photo of a"
        n_ctx = 4
        # 如果n_ctx不等于ctx_init字符数量，则报错
        assert n_ctx == len(ctx_init.split(" "))
        dtype = clip_model.dtype

        # 将初始promte：a photo of a 给embedding，变成可学习参数ctx
        prompt = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
        prompt_prefix = ctx_init

        # 将ctx_vectors变为可学习的参数传递给ctx
        self.ctx = nn.Parameter(ctx_vectors)  # 可学习,初始值仍为a photo of a
        # self.ctx = ctx_vectors  # 不学习，仍然是a photo of a

        # 生成最终的句子，格式为[prompt_prefix] [三联label].
        prompts = [prompt_prefix + " " + triplet_label[id] + "." for id in triplet_label.keys()]
        # token化，然后喂入CLIP 做embedding
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # 使用缓存register_buffer将参数保存，但不参与更新
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        # token_suffix是包含CLS即[class]的，这是不可学习的
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        # 没有用到，其实就是a photo of a只不过参数化后变成ctx了，后续用到的是ctx
        self.register_buffer("token_middle", embedding[:, 1: (1 + n_ctx), :])

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.n_cls = n_cls

    def forward(self):
        ctx = self.ctx
        # 当batch_size为1时，仍然将其扩展为3维以上。
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix

        # ctx是可学习的参数，prefix和suffix是不可学习的参数，是SOS,CLS,SOS即开头和结尾符号
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],  # type: ignore
            dim=1,
        )
        return prompts


class TextEncoder(nn.Module):  # 将promte后的word embedding转化为text encoder

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # x是word embedding
        x = prompts + self.positional_embedding.type(self.dtype)
        # 维度顺序改变，便于transformer计算
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        # 计算完再改回来
        x = x.permute(1, 0, 2)  # LND -> NLD
        # 归一化
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # 得到最终的文本特征，也就是text encoder的结果。
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )

        return x


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 可学习的权重矩阵，维度为in_features乘out_features
        self.weight = nn.parameter.Parameter(torch.Tensor(in_features, out_features))
        # 可学习的偏置项
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    # 初始化权重参数
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        # 也可以采用凯明初始化
        # nn.init.kaiming_uniform_(self.weight)

    def forward(self, input, adj):
        # 先计算HW，即text与W乘
        support = torch.matmul(input, self.weight)
        # 再计算A*HW，即临近矩阵乘HW
        # 顺序有待商榷，GCN正常是先AH，再乘W
        output = torch.matmul(adj, support)
        # 加偏置项
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    # 可以使用print打印模型，输出模型的输入和输出特征维度，方便调试
    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


# %% Triplet Components Feature Encoder


class Encoder(nn.Module):
    def __init__(self, basename='resnet18', num_tool=6,  num_verb=10, num_target=15, num_triplet=100, enctype='cagam', is_bottleneck=True, hr_output=False, m=3):
        super(Encoder, self).__init__()
        depth = 64 if basename == 'resnet18' else 128
        def Ignore(x): return None
        self.basemodel = BaseModel(basename, hr_output)
        self.wsl = WSL(num_tool, depth)
        self.cagam = CAG(num_tool, num_verb, num_target) if enctype == 'cag' else CAGAM(
            num_tool, num_verb, num_target, m=m)
        self.bottleneck = Bottleneck(num_triplet) if is_bottleneck else Ignore

    def forward(self, x):
        high_x, low_x = self.basemodel(x)
        enc_i = self.wsl(high_x)
        enc_v, enc_t = self.cagam(high_x, enc_i[0])
        enc_ivt = self.bottleneck(low_x)
        return enc_i, enc_v, enc_t, enc_ivt, high_x

# %% MultiHead Attention Decoder


class Decoder(nn.Module):
    def __init__(self, layer_size, d_model, num_heads, num_class=100, use_ln=False, m=3):
        super(Decoder, self).__init__()
        self.projection = nn.ModuleList(
            [Projection(num_triplet=num_class, out_depth=d_model) for i in range(layer_size)])
        self.mhma = nn.ModuleList([MHMA(num_class=num_class, depth=d_model,
                                  num_heads=num_heads, use_ln=use_ln) for i in range(layer_size)])
        self.ffnet = nn.ModuleList(
            [FFN(k=layer_size-i-1, num_class=num_class, use_ln=use_ln) for i in range(layer_size)])
        self.classifier = Classifier(num_class, m=m)

    def forward(self, enc_i, enc_v, enc_t, enc_ivt):
        X = enc_ivt.clone()
        for P, M, F in zip(self.projection, self.mhma, self.ffnet):
            X = P(enc_i[0], enc_v[0], enc_t[0], X)
            X = M(X)
            X = F(X)
        logits = self.classifier(X)
        return logits, X

# %% Feature extraction backbone


class BaseModel(nn.Module):
    def __init__(self, basename='resnet18', hr_output=False, *args):
        super(BaseModel, self).__init__(*args)
        self.output_feature = {}
        if basename == 'resnet18':
            self.basemodel = basemodels.resnet18(pretrained=True)
            if hr_output:
                self.increase_resolution()
            self.basemodel.layer1[1].bn2.register_forward_hook(
                self.get_activation('low_level_feature'))
            self.basemodel.layer4[1].bn2.register_forward_hook(
                self.get_activation('high_level_feature'))
        if basename == 'resnet50':
            self.basemodel = basemodels.resnet50(pretrained=True)
            # print(self.basemodel)
            self.basemodel.layer1[2].bn2.register_forward_hook(
                self.get_activation('low_level_feature'))
            self.basemodel.layer4[2].bn2.register_forward_hook(
                self.get_activation('high_level_feature'))

    def increase_resolution(self):
        global OUT_HEIGHT, OUT_WIDTH
        self.basemodel.layer3[0].conv1.stride = (1, 1)
        self.basemodel.layer3[0].downsample[0].stride = (1, 1)
        self.basemodel.layer4[0].conv1.stride = (1, 1)
        self.basemodel.layer4[0].downsample[0].stride = (1, 1)
        OUT_HEIGHT *= 4
        OUT_WIDTH *= 4
        print("using high resolution output ({}x{})".format(OUT_HEIGHT, OUT_WIDTH))

    def get_activation(self, layer_name):
        def hook(module, input: Tuple[torch.Tensor], output: torch.Tensor):
            self.output_feature[layer_name] = output
        return hook

    def forward(self, x):
        _ = self.basemodel(x)
        return self.output_feature['high_level_feature'], self.output_feature['low_level_feature']

# %% Weakly-Supervised localization


class WSL(nn.Module):
    def __init__(self, num_class, depth=64):
        super(WSL, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=512, out_channels=depth, kernel_size=3, padding=1)
        self.cam = nn.Conv2d(
            in_channels=depth, out_channels=num_class, kernel_size=1)
        self.elu = nn.ELU()
        self.bn = nn.BatchNorm2d(depth)
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        feature = self.conv1(x)
        feature = self.bn(feature)
        feature = self.elu(feature)
        cam = self.cam(feature)
        logits = self.gmp(cam).squeeze(-1).squeeze(-1)
        return cam, logits

# %% Unfiltered Bottleneck layer


class Bottleneck(nn.Module):
    def __init__(self, num_class):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=64, out_channels=256, stride=(2, 2), kernel_size=3)
        # self.conv2 = nn.Conv2d(in_channels=256, out_channels=num_class, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=256, out_channels=num_class, kernel_size=1)
        self.elu = nn.ELU()
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(num_class)

    def forward(self, x):
        feature = self.conv1(x)
        feature = self.bn1(feature)
        feature = self.elu(feature)
        feature = self.conv2(feature)
        feature = self.bn2(feature)
        feature = self.elu(feature)
        return feature


# %% Class Activation Guided Temporal Attention Mechanism (CAGTAM)
# TODO: change name to CAGTAM
class CAGAM(nn.Module):
    def __init__(self, num_tool, num_verb, num_target, in_depth=512, m=3):
        super(CAGAM, self).__init__()
        out_depth = num_tool
        self.verb_context = nn.Conv2d(
            in_channels=in_depth, out_channels=out_depth, kernel_size=3, padding=1)
        self.verb_query = nn.Conv2d(
            in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.verb_tool_query = nn.Conv2d(
            in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.verb_key = nn.Conv2d(
            in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.verb_tool_key = nn.Conv2d(
            in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.verb_cmap = nn.Conv2d(
            in_channels=out_depth, out_channels=num_verb, kernel_size=1)
        self.target_context = nn.Conv2d(
            in_channels=in_depth, out_channels=out_depth, kernel_size=3, padding=1)
        self.target_query = nn.Conv2d(
            in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.target_tool_query = nn.Conv2d(
            in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.target_key = nn.Conv2d(
            in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.target_tool_key = nn.Conv2d(
            in_channels=out_depth, out_channels=out_depth, kernel_size=1)
        self.target_cmap = nn.Conv2d(
            in_channels=out_depth, out_channels=num_target, kernel_size=1)
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.elu = nn.ELU()
        self.soft = nn.Softmax(dim=1)
        self.flat = nn.Flatten(2, 3)
        self.bn1 = nn.BatchNorm2d(out_depth)
        self.bn2 = nn.BatchNorm2d(out_depth)
        self.bn3 = nn.BatchNorm2d(out_depth)
        self.bn4 = nn.BatchNorm2d(out_depth)
        self.bn5 = nn.BatchNorm2d(out_depth)
        self.bn6 = nn.BatchNorm2d(out_depth)
        self.bn7 = nn.BatchNorm2d(out_depth)
        self.bn8 = nn.BatchNorm2d(out_depth)
        self.bn9 = nn.BatchNorm2d(out_depth)
        self.bn10 = nn.BatchNorm2d(out_depth)
        self.bn11 = nn.BatchNorm2d(out_depth)
        self.bn12 = nn.BatchNorm2d(out_depth)
        self.encoder_cagam_verb_beta = torch.nn.Parameter(torch.randn(1))
        self.encoder_cagam_target_beta = torch.nn.Parameter(torch.randn(1))
        self.m = m
        self.bn13 = nn.BatchNorm2d(self.m+1)
        self.gate = nn.Conv2d(self.m+1, self.m+1,
                              kernel_size=1) if self.m > 0 else nn.Identity()
        self.fc1 = nn.Linear(num_verb, 1)
        self.fc2 = nn.Linear(num_target, 1)

        # NOTE: Temporal Attention Module (TAM)
        self.cmap_attn4 = TAM(in_channels=num_verb, m=self.m, k=1)
        self.cmap_attn5 = TAM(in_channels=num_verb, m=self.m, k=1)
        self.cbn4 = nn.BatchNorm2d(num_verb)

    def get_verb(self, raw, cam):
        x = self.elu(self.bn1(self.verb_context(raw)))

        z = x.clone()
        sh = list(z.shape)
        sh[0] = -1
        q1 = self.elu(self.bn2(self.verb_query(x)))
        k1 = self.elu(self.bn3(self.verb_key(x)))

        w1 = self.flat(k1).matmul(self.flat(q1).transpose(-1, -2))

        q2 = self.elu(self.bn4(self.verb_tool_query(cam)))
        k2 = self.elu(self.bn5(self.verb_tool_key(cam)))
        w2 = self.flat(k2).matmul(self.flat(q2).transpose(-1, -2))

        attention = (w1 * w2) / \
            torch.sqrt(torch.tensor(sh[-1], dtype=torch.float32))
        attention = self.soft(attention)

        v = self.flat(z)
        e = (attention.matmul(v) * self.encoder_cagam_verb_beta).reshape(sh)
        e = self.bn6(e + z)

        cmap = self.verb_cmap(e)

        # NOTE: TAM in Late Fusion mode
        cmap = self.cmap_attn4(cmap)
        cmap = self.cbn4(cmap)
        cmap = self.elu(cmap)

        cmap = self.cmap_attn5(cmap)

        y = self.gmp(cmap).squeeze(-1).squeeze(-1)

        return cmap, y

    def get_target(self, raw, cam):
        x = self.elu(self.bn7(self.target_context(raw)))
        z = x.clone()
        sh = list(z.shape)
        sh[0] = -1
        q1 = self.elu(self.bn8(self.target_query(x)))
        k1 = self.elu(self.bn9(self.target_key(x)))
        w1 = self.flat(k1).transpose(-1, -2).matmul(self.flat(q1))

        q2 = self.elu(self.bn10(self.target_tool_query(cam)))
        k2 = self.elu(self.bn11(self.target_tool_key(cam)))
        w2 = self.flat(k2).transpose(-1, -2).matmul(self.flat(q2))
        attention = (w1 * w2) / \
            torch.sqrt(torch.tensor(sh[-1], dtype=torch.float32))
        attention = self.soft(attention)
        v = self.flat(z)
        e = (v.matmul(attention) * self.encoder_cagam_target_beta).reshape(sh)
        e = self.bn12(e + z)
        cmap = self.target_cmap(e)

        y = self.gmp(cmap).squeeze(-1).squeeze(-1)
        return cmap, y

    def forward(self, x, cam):
        cam_v, logit_v = self.get_verb(x, cam)
        cam_t, logit_t = self.get_target(x, cam)
        return (cam_v, logit_v), (cam_t, logit_t)


class TAM(nn.Module):
    """
    Constructs proposed Temporal Attention Module (TAM)
    """

    def __init__(self, in_channels=10, m=3, k=1):
        super(TAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=k, bias=False)
        self.act = nn.Sigmoid()
        self.m = m
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x).squeeze()

        # reshape to include time dimension
        B, c = y.shape  # 16x10
        b = int(B/(self.m + 1))  # 4
        y = y.reshape([b, self.m+1, c])  # 4 x 4 x 10

        # apply conv to 1 dim value
        y = self.bn1(self.conv(y.transpose(-1, -2))).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.act(y)

        B, c, h, w = x.shape
        x = x.reshape([b, self.m+1, c, h, w])
        past_cmap, curr_cmap = torch.split(x, [self.m, 1], 1)
        x = x * y.unsqueeze(-1).unsqueeze(-1)
        x = x.sum(dim=1).unsqueeze(1)
        x = torch.cat([past_cmap, x], dim=1).view(b*(self.m+1), c, h, w)

        return x

# %% Projection function


class Projection(nn.Module):
    def __init__(self, num_tool=6, num_verb=10, num_target=15, num_triplet=100, out_depth=128):
        super(Projection, self).__init__()
        self.ivt_value = nn.Conv2d(
            in_channels=num_triplet, out_channels=out_depth, kernel_size=1)
        self.i_value = nn.Conv2d(in_channels=num_tool,
                                 out_channels=out_depth, kernel_size=1)
        self.v_value = nn.Conv2d(in_channels=num_verb,
                                 out_channels=out_depth, kernel_size=1)
        self.t_value = nn.Conv2d(
            in_channels=num_target, out_channels=out_depth, kernel_size=1)
        self.ivt_query = nn.Linear(
            in_features=num_triplet, out_features=out_depth)
        self.dropout = nn.Dropout(p=0.3)
        self.ivt_key = nn.Linear(
            in_features=num_triplet, out_features=out_depth)
        self.i_key = nn.Linear(in_features=num_tool, out_features=out_depth)
        self.v_key = nn.Linear(in_features=num_verb, out_features=out_depth)
        self.t_key = nn.Linear(in_features=num_target, out_features=out_depth)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.elu = nn.ELU()
        self.bn1 = nn.BatchNorm1d(out_depth)
        self.bn2 = nn.BatchNorm1d(out_depth)
        self.bn3 = nn.BatchNorm2d(out_depth)
        self.bn4 = nn.BatchNorm1d(out_depth)
        self.bn5 = nn.BatchNorm2d(out_depth)
        self.bn6 = nn.BatchNorm1d(out_depth)
        self.bn7 = nn.BatchNorm2d(out_depth)
        self.bn8 = nn.BatchNorm1d(out_depth)
        self.bn9 = nn.BatchNorm2d(out_depth)

    def forward(self, cam_i, cam_v, cam_t, X):
        q = self.elu(self.bn1(self.ivt_query(
            self.dropout(self.gap(X).squeeze(-1).squeeze(-1)))))
        k = self.elu(self.bn2(self.ivt_key(
            self.gap(X).squeeze(-1).squeeze(-1))))
        v = self.bn3(self.ivt_value(X))
        k1 = self.elu(
            self.bn4(self.i_key(self.gap(cam_i).squeeze(-1).squeeze(-1))))
        v1 = self.elu(self.bn5(self.i_value(cam_i)))
        k2 = self.elu(
            self.bn6(self.v_key(self.gap(cam_v).squeeze(-1).squeeze(-1))))
        v2 = self.elu(self.bn7(self.v_value(cam_v)))
        k3 = self.elu(
            self.bn8(self.t_key(self.gap(cam_t).squeeze(-1).squeeze(-1))))
        v3 = self.elu(self.bn9(self.t_value(cam_t)))
        sh = list(v1.shape)
        v = self.elu(F.interpolate(v, (sh[2], sh[3])))
        X = self.elu(F.interpolate(X, (sh[2], sh[3])))
        return (X, (k1, v1), (k2, v2), (k3, v3), (q, k, v))


# %% Multi-head of self and cross attention
class MHMA(nn.Module):
    def __init__(self, depth, num_class=100, num_heads=4, use_ln=False):
        super(MHMA, self).__init__()
        self.concat = nn.Conv2d(
            in_channels=depth*num_heads, out_channels=num_class, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_class)
        self.ln = nn.LayerNorm(
            [num_class, OUT_HEIGHT, OUT_WIDTH]) if use_ln else nn.BatchNorm2d(num_class)
        self.elu = nn.ELU()
        self.soft = nn.Softmax(dim=1)
        self.heads = num_heads

    def scale_dot_product(self, key, value, query):
        dk = torch.sqrt(torch.tensor(list(key.shape)[-2], dtype=torch.float32))
        affinity = key.matmul(query.transpose(-1, -2))
        attn_w = affinity / dk
        attn_w = self.soft(attn_w)
        attention = attn_w.matmul(value)
        return attention

    def forward(self, inputs):
        (X, (k1, v1), (k2, v2), (k3, v3), (q, k, v)) = inputs
        query = torch.stack([q]*self.heads, dim=1)  # [B,Head,D]
        query = query.unsqueeze(dim=-1)  # [B,Head,D,1]
        key = torch.stack([k, k1, k2, k3], dim=1)  # [B,Head,D]
        key = key.unsqueeze(dim=-1)  # [B,Head,D,1]
        value = torch.stack([v, v1, v2, v3], dim=1)  # [B,Head,D,H,W]
        dims = list(value.shape)  # [B,Head,D,H,W]
        value = value.reshape(
            [-1, dims[1], dims[2], dims[3]*dims[4]])  # [B,Head,D,HW]
        attn = self.scale_dot_product(key, value, query)  # [B,Head,D,HW]
        attn = attn.reshape(
            [-1, dims[1]*dims[2], dims[3], dims[4]])  # [B,DHead,H,W]
        mha = self.elu(self.bn(self.concat(attn)))
        mha = self.ln(mha + X.clone())
        return mha

# %% Feed-forward layer


class FFN(nn.Module):
    def __init__(self, k, num_class=100, use_ln=False):
        super(FFN, self).__init__()
        def Ignore(x): return x
        self.conv1 = nn.Conv2d(in_channels=num_class,
                               out_channels=num_class, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_class,
                               out_channels=num_class, kernel_size=1)
        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU() if k > 0 else Ignore
        self.bn1 = nn.BatchNorm2d(num_class)
        self.bn2 = nn.BatchNorm2d(num_class)
        self.ln = nn.LayerNorm(
            [num_class, OUT_HEIGHT, OUT_WIDTH]) if use_ln else nn.BatchNorm2d(num_class)

    def forward(self, inputs,):
        x = self.elu1(self.bn1(self.conv1(inputs)))
        x = self.elu2(self.bn2(self.conv2(x)))
        x = self.ln(x + inputs.clone())
        return x

# %% Classification layer


class Classifier(nn.Module):
    def __init__(self, layer_size, num_class=100, m=3):
        super(Classifier, self).__init__()
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.mlp = nn.Linear(in_features=num_class, out_features=num_class)

    def forward(self, inputs):
        x = self.gmp(inputs).squeeze(-1).squeeze(-1)
        y = self.mlp(x)
        return y
