import copy

import numpy as np
import torch
import torch.nn as nn
import brevitas.nn as qnn


n_bits = 6
scale_factor = 1.0


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_flatten = feat.reshape(N, C, -1)
    feat_mean = (feat_flatten.sum(dim=2) / (size[2] * size[3])).reshape(N, C, 1, 1)
    feat_var = ((feat - feat_mean.expand(size)) ** 2).reshape(N, C, -1)
    feat_var = feat_var.sum(dim=2) / (size[2] * size[3] - 1) + eps
    feat_std = feat_var.pow(0.5).reshape(N, C, 1, 1)
    return feat_mean, feat_std


class CustomUpsampling(nn.Module):
    def __init__(self, s):
        super(CustomUpsampling, self).__init__()
        self.s = s

    def forward(self, x):
        n, c, h, w = x.shape
        out = x.reshape(-1, c, h, 1, w, 1)
        out = out.cat([out] * self.s, dim=-3)
        out = out.cat([out] * self.s, dim=-1)
        out = out.reshape(-1, c, h * self.s, w * self.s)
        return out


class CustomAvgPool2dTo1X1(nn.Module):
    def __init__(self):
        super(CustomAvgPool2dTo1X1, self).__init__()

    def forward(self, x):
        n, c, h, w = x.shape
        out = x.reshape(-1, c, h * w)
        out = out.sum(dim=2) / (h * w)
        return out.reshape(-1, c, 1, 1)


class Reflection1xPad2d(nn.Module):
    def __init__(self):
        super(Reflection1xPad2d, self).__init__()
        self.padding = 1

    def forward(self, x):
        x_pad_left = x[:, :, :, [1]]
        x_pad_right = x[:, :, :, [-2]]
        x_pad = torch.cat([x_pad_left, x, x_pad_right], dim=3)

        x_pad_top = x_pad[:, :, [1], :]
        x_pad_bottom = x_pad[:, :, [-2], :]
        x_pad = torch.cat([x_pad_top, x_pad, x_pad_bottom], dim=2)

        return x_pad


class Reflection4xPad2d(nn.Module):
    def __init__(self):
        super(Reflection4xPad2d, self).__init__()
        self.padding = 4

    def forward(self, x):
        x_pad_left = x[:, :, :, [4, 3, 2, 1]]
        x_pad_right = x[:, :, :, [-2, -3, -4, -5]]
        x_pad = torch.cat([x_pad_left, x, x_pad_right], dim=3)

        x_pad_top = x_pad[:, :, [4, 3, 2, 1], :]
        x_pad_bottom = x_pad[:, :, [-2, -3, -4, -5], :]
        x_pad = torch.cat([x_pad_top, x_pad, x_pad_bottom], dim=2)

        return x_pad


class QuantConv2dReflection(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, weight_bit_width, bias_quant, return_quant_tensor):
        super(QuantConv2dReflection, self).__init__()
        if padding == 0:
            self.reflection = nn.Identity()
        elif padding == 1:
            self.reflection = Reflection1xPad2d()
        elif padding == 4:
            self.reflection = Reflection4xPad2d()
        else:
            raise ValueError(f"Unsupported padding: {padding}")

        self.quant_ident = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.conv = qnn.QuantConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, groups=groups,
                                    weight_bit_width=weight_bit_width, bias_quant=bias_quant,
                                    return_quant_tensor=return_quant_tensor)

    def forward(self, x):
        x = self.reflection(x)
        x = self.quant_ident(x)
        return self.conv(x)


class ResidualLayer(nn.Module):
    def __init__(self, channels=128, kernel_size=3, groupnum=1):
        super(ResidualLayer, self).__init__()
        self.input_quant_inp = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False)
        self.weight_quant_inp = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.x1_quant_inp = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.x4_quant_inp = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.add_quant_inp_1 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.add_quant_inp_2 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)

        self.bias_quant_inp = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False)
        self.conv1 = QuantConv2dReflection(channels, channels, kernel_size, stride=1, padding=kernel_size // 2,
                                     groups=groupnum, weight_bit_width=n_bits, bias_quant=None,
                                     return_quant_tensor=True)
        self.relu = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=False)
        self.conv2 = QuantConv2dReflection(channels, channels, kernel_size, stride=1, padding=kernel_size // 2,
                                     groups=groupnum, weight_bit_width=n_bits, bias_quant=None,
                                     return_quant_tensor=True)
        self.out_quant_inp = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)

    def forward(self, x, weight=None, bias=None, filterMod=False):
        quant_x = self.input_quant_inp(x)
        if filterMod:
            quant_weight = self.weight_quant_inp(weight)
            quant_bias = self.bias_quant_inp(bias)
            x1 = self.x1_quant_inp(self.conv1(quant_x))
            x2 = self.add_quant_inp_1(quant_weight * x1) + self.add_quant_inp_1(quant_bias * quant_x)

            x3 = self.relu(x2)
            x4 = self.x4_quant_inp(self.conv2(x3))
            x5 = self.add_quant_inp_2(quant_weight * x4) + self.add_quant_inp_2(quant_bias * x3)
            return self.out_quant_inp(x) + self.out_quant_inp(x5)
        else:
            return self.out_quant_inp(x) + self.out_quant_inp(self.conv2(self.relu(self.conv1(quant_x))))


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.input_quant_inp = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False)
        self.enc1 = nn.Sequential(
            qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False),
            QuantConv2dReflection(3, int(16 * scale_factor), kernel_size=9, stride=1, padding=9 // 2,
                            groups=1, weight_bit_width=n_bits, bias_quant=None,
                            return_quant_tensor=True),
            qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=False),
            QuantConv2dReflection(int(16 * scale_factor), int(32 * scale_factor), kernel_size=3, stride=2, padding=3 // 2,
                            groups=int(16 * scale_factor), weight_bit_width=n_bits, bias_quant=None,
                            return_quant_tensor=True),
            qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=False),
            QuantConv2dReflection(int(32 * scale_factor), int(32 * scale_factor), kernel_size=1, stride=1, padding=1 // 2,
                            groups=1, weight_bit_width=n_bits, bias_quant=None,
                            return_quant_tensor=True),
            qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=False),
            QuantConv2dReflection(int(32 * scale_factor), int(64 * scale_factor), kernel_size=3, stride=2, padding=3 // 2,
                            groups=int(32 * scale_factor), weight_bit_width=n_bits, bias_quant=None,
                            return_quant_tensor=True),
            qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=False),
            QuantConv2dReflection(int(64 * scale_factor), int(64 * scale_factor), kernel_size=1, stride=1, padding=1 // 2,
                            groups=1, weight_bit_width=n_bits, bias_quant=None,
                            return_quant_tensor=True),
            qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True),
            ResidualLayer(int(64 * scale_factor), kernel_size=3),
            qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=False),
        )
        self.enc2 = nn.Sequential(
            ResidualLayer(int(64 * scale_factor), kernel_size=3),
            qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=False),
        )

    def forward(self, x):
        x1 = self.enc1(self.input_quant_inp(x))
        x2 = self.enc2(x1)
        out = [x1, x2]
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.content_quant_inp_1 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False)
        self.style_quant_inp_1 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False)
        self.weight_1_quant_inp = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False)
        self.bias_1_quant_inp = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False)
        self.x2_quant_inp = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False)
        self.style_quant_inp_2 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False)
        self.weight_0_quant_inp = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False)
        self.bias_0_quant_inp = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False)

        self.sub_quant_inp_1 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False)
        self.sub_quant_inp_2 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False)

        self.dec1 = ResidualLayer(int(64 * scale_factor), kernel_size=3)
        self.dec2 = ResidualLayer(int(64 * scale_factor), kernel_size=3)
        self.dec3 = nn.Sequential(
            qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True),
            CustomUpsampling(s=2),
            qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False),
            QuantConv2dReflection(int(64 * scale_factor), int(32 * scale_factor), kernel_size=3, stride=1, padding=3 // 2,
                            groups=int(32 * scale_factor), weight_bit_width=n_bits, bias_quant=None,
                            return_quant_tensor=True),
            qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=False),
            QuantConv2dReflection(int(32 * scale_factor), int(32 * scale_factor), kernel_size=1, stride=1, padding=1 // 2,
                            groups=1, weight_bit_width=n_bits, bias_quant=None, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True),
            qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True),
            CustomUpsampling(s=2),
            qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False),
            QuantConv2dReflection(int(32 * scale_factor), int(16 * scale_factor), kernel_size=3, stride=1, padding=3 // 2,
                            groups=int(16 * scale_factor), weight_bit_width=n_bits, bias_quant=None,
                            return_quant_tensor=True),
            qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=False),
            QuantConv2dReflection(int(16 * scale_factor), int(16 * scale_factor), kernel_size=1, stride=1, padding=1 // 2,
                            groups=1, weight_bit_width=n_bits, bias_quant=None, return_quant_tensor=True),
            qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=False),
            QuantConv2dReflection(int(16 * scale_factor), 3, kernel_size=9, stride=1, padding=9 // 2,
                            groups=1, weight_bit_width=n_bits, bias_quant=None, return_quant_tensor=True),
            qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False),
        )

    def forward(self, contents, styles, weights, biases):
        # 1.0: x1 = featMod(self.x1_quant_inp(x[1]), self.s1_quant_inp(s[1]))
        quant_content_1 = self.content_quant_inp_1(contents[1])
        quant_style_1 = self.style_quant_inp_1(styles[1])

        # 1.1: calculate mean and std of s1
        n, c, h, w = quant_style_1.size()
        quant_style_1_reshape = quant_style_1.reshape(-1, c, h * w)
        quant_style_1_mean = (quant_style_1_reshape.sum(dim=2) / (h * w)).reshape(-1, c, 1, 1)

        # 1.2: calculate mean and std of x1
        n, c, h, w = quant_content_1.size()
        quant_content_1_reshape = quant_content_1.reshape(n, c, -1)
        quant_content_1_mean = (quant_content_1_reshape.sum(dim=2) / (h * w)).reshape(n, c, 1, 1)

        # 1.3: normalize quant_x1 with quant_s1
        x1 = self.sub_quant_inp_1(quant_content_1 - quant_content_1_mean)
        x1 = x1 + quant_style_1_mean

        # 2.0: x2 = self.dec1(x1, self.w1_quant_inp(weights[1]), self.b1_quant_inp(biases[1]), filterMod=True)
        quant_weight_1 = self.weight_1_quant_inp(weights[1])
        quant_bias_1 = self.bias_1_quant_inp(biases[1])
        x2 = self.dec1(x1, quant_weight_1, quant_bias_1, filterMod=True)

        # 3.0: x3 = featMod(x2, self.s0_quant_inp(s[0]))
        quant_x2 = self.x2_quant_inp(x2)
        quant_style_0 = self.style_quant_inp_2(styles[0])

        # 3.1: calculate mean and std of s0
        n, c, h, w = quant_style_0.size()
        quant_style_0_reshape = quant_style_0.reshape(-1, c, h * w)
        quant_style_0_mean = (quant_style_0_reshape.sum(dim=2) / (h * w)).reshape(-1, c, 1, 1)

        # 3.2: calculate mean and std of quant_x2
        n, c, h, w = quant_x2.size()
        quant_x2_reshape = quant_x2.reshape(-1, c, h * w)
        quant_x2_mean = (quant_x2_reshape.sum(dim=2) / (h * w)).reshape(-1, c, 1, 1)

        # 3.3 normalize x2 with s0
        x3 = self.sub_quant_inp_2(quant_x2 - quant_x2_mean)
        x3 = x3 + quant_style_0_mean

        # 4.0: x4 = self.dec2(x3, self.w0_quant_inp(weights[0]), self.b0_quant_inp(biases[0]), filterMod=True)
        quant_w0 = self.weight_0_quant_inp(weights[0])
        quant_b0 = self.bias_0_quant_inp(biases[0])
        x4 = self.dec2(x3, quant_w0, quant_b0, filterMod=True)

        # 5.0: out = self.dec3(x4)
        out = self.dec3(x4)
        return out


class Modulator(nn.Module):
    def __init__(self):
        super(Modulator, self).__init__()
        self.weight1 = nn.Sequential(
            QuantConv2dReflection(int(64 * scale_factor), int(64 * scale_factor), kernel_size=3, stride=1, padding=3 // 2,
                            groups=int(64 * scale_factor), weight_bit_width=n_bits, bias_quant=None,
                            return_quant_tensor=True),
            qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False),
            CustomAvgPool2dTo1X1(),
            qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False),
        )
        self.bias1 = nn.Sequential(
            QuantConv2dReflection(int(64 * scale_factor), int(64 * scale_factor), kernel_size=3, stride=1, padding=3 // 2,
                            groups=int(64 * scale_factor), weight_bit_width=n_bits, bias_quant=None,
                            return_quant_tensor=True),
            qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False),
            CustomAvgPool2dTo1X1(),
            qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False),
        )
        self.weight2 = nn.Sequential(
            QuantConv2dReflection(int(64 * scale_factor), int(64 * scale_factor), kernel_size=3, stride=1, padding=3 // 2,
                            groups=int(64 * scale_factor), weight_bit_width=n_bits, bias_quant=None,
                            return_quant_tensor=True),
            qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False),
            CustomAvgPool2dTo1X1(),
            qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False),
        )
        self.bias2 = nn.Sequential(
            QuantConv2dReflection(int(64 * scale_factor), int(64 * scale_factor), kernel_size=3, stride=1, padding=3 // 2,
                            groups=int(64 * scale_factor), weight_bit_width=n_bits, bias_quant=None,
                            return_quant_tensor=True),
            qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False),
            CustomAvgPool2dTo1X1(),
            qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=False),
        )

    def forward(self, x):
        w1 = self.weight1(x[0])
        b1 = self.bias1(x[0])

        w2 = self.weight2(x[1])
        b2 = self.bias2(x[1])

        return [w1, w2], [b1, b2]


vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),

    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class Net(nn.Module):
    def __init__(self, vgg, content_encoder, style_encoder, modulator, decoder):
        super(Net, self).__init__()
        vgg_enc_layers = list(vgg.children())
        self.vgg_enc_1 = nn.Sequential(*vgg_enc_layers[:4])  # input -> relu1_1
        self.vgg_enc_2 = nn.Sequential(*vgg_enc_layers[4:11])  # relu1_1 -> relu2_1
        self.vgg_enc_3 = nn.Sequential(*vgg_enc_layers[11:18])  # relu2_1 -> relu3_1
        self.vgg_enc_4 = nn.Sequential(*vgg_enc_layers[18:31])  # relu3_1 -> relu4_1

        self.content_encoder = content_encoder
        self.style_encoder = style_encoder
        self.modulator = modulator
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        # freeze the encoder
        for name in ['vgg_enc_1', 'vgg_enc_2', 'vgg_enc_3', 'vgg_enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_vgg_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'vgg_enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode_vgg_content(self, input):
        for i in range(4):
            input = getattr(self, 'vgg_enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
            self.mse_loss(input_std, target_std)

    def forward(self, content, style):
        # extract style modulation signals
        style_feats = self.style_encoder(style)
        filter_weights, filter_biases = self.modulator(style_feats)

        # extract content features
        content_feats = self.content_encoder(content)

        # generate results
        res = self.decoder(content_feats, style_feats, filter_weights, filter_biases)

        # vgg content and style loss
        res_feats_vgg = self.encode_with_vgg_intermediate(res)

        style_feats_vgg = self.encode_with_vgg_intermediate(style)
        content_feats_vgg = self.encode_vgg_content(content)

        loss_c = self.calc_content_loss(res_feats_vgg[-1], content_feats_vgg)
        loss_s = self.calc_style_loss(res_feats_vgg[0], style_feats_vgg[0])
        for i in range(1, 4):
            loss_s = loss_s + self.calc_style_loss(res_feats_vgg[i], style_feats_vgg[i])

        res_style_feats = self.style_encoder(res)
        res_filter_weights, res_filter_biases = self.modulator(res_style_feats)

        # style signal contrastive loss
        loss_contrastive = 0.
        for i in range(int(style.size(0))):
            pos_loss = 0.
            neg_loss = 0.

            for j in range(int(style.size(0))):
                if j == i:
                    FeatMod_loss = self.calc_style_loss(res_style_feats[0][i].unsqueeze(0),
                                                        style_feats[0][j].unsqueeze(0)) + \
                                   self.calc_style_loss(res_style_feats[1][i].unsqueeze(0),
                                                        style_feats[1][j].unsqueeze(0))
                    FilterMod_loss = self.calc_content_loss(res_filter_weights[0][i],
                                                            filter_weights[0][j]) + \
                                     self.calc_content_loss(res_filter_weights[1][i],
                                                            filter_weights[1][j]) + \
                                     self.calc_content_loss(res_filter_biases[0][i], filter_biases[0][j]) + \
                                     self.calc_content_loss(res_filter_biases[1][i], filter_biases[1][j])
                    pos_loss = FeatMod_loss + FilterMod_loss
                else:
                    FeatMod_loss = self.calc_style_loss(res_style_feats[0][i].unsqueeze(0),
                                                        res_style_feats[0][j].unsqueeze(0)) + \
                                   self.calc_style_loss(res_style_feats[1][i].unsqueeze(0),
                                                        style_feats[1][j].unsqueeze(0))
                    FilterMod_loss = self.calc_content_loss(res_filter_weights[0][i],
                                                            filter_weights[0][j]) + \
                                     self.calc_content_loss(res_filter_weights[1][i],
                                                            filter_weights[1][j]) + \
                                     self.calc_content_loss(res_filter_biases[0][i], filter_biases[0][j]) + \
                                     self.calc_content_loss(res_filter_biases[1][i], filter_biases[1][j])
                    neg_loss = neg_loss + FeatMod_loss + FilterMod_loss

            loss_contrastive = loss_contrastive + pos_loss / neg_loss

        return res, loss_c, loss_s, loss_contrastive


class TestNet(nn.Module):
    def __init__(self, content_encoder, style_encoder, modulator, decoder):
        super(TestNet, self).__init__()

        self.content_encoder = content_encoder
        self.style_encoder = style_encoder
        self.modulator = modulator
        self.decoder = decoder

    def forward(self, content, style):
        style_feats = self.style_encoder(style)
        filter_weights, filter_biases = self.modulator(style_feats)

        content_feats = self.content_encoder(content)

        res = self.decoder(content_feats, style_feats, filter_weights, filter_biases)
        return res


if __name__ == '__main__':
    content = torch.randn(2, 3, 128, 128)
    style = torch.randn(2, 3, 128, 128)
    network = Net(vgg, Encoder(), Encoder(), Modulator(), Decoder())
    out = network(content, style)
    print(out)
