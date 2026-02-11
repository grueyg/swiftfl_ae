import torch
from torch import nn
from einops import rearrange
from federatedscope.spec.predictor.model.module import ConvSC, gInception_ST

class Predictor_Model(nn.Module):

    def __init__(self, 
                 grad_shape, 
                 s_hid_dim=16, 
                 t_hid_dim=256, 
                 s_block_num=4, 
                 t_block_num=4,
                 enc_kernel_size=3,
                 dec_kernel_size=3,
                 **kwargs):
        super(Predictor_Model, self).__init__()
        history_length, g_hid_dim, = grad_shape[0:2]
        self.history_length = history_length

        self.spatial_enc = SpatialEncoder(g_hid_dim, s_hid_dim, s_block_num, enc_kernel_size)
        self.temporal_enc = TemporalEncoder(history_length * s_hid_dim, t_hid_dim, t_block_num)
        self.decoder = Decoder(s_hid_dim, g_hid_dim, s_block_num, dec_kernel_size)

    def reshape(self, feat, reverse=False):
        if reverse:
            return rearrange(feat, '(b t) c h w -> b t c h w', t=self.history_length)
        else:
            return rearrange(feat, 'b t c h w -> (b t) c h w')

    def forward(self, grad):
        grad = self.reshape(grad)

        spatial_feat, history_info = self.spatial_enc(grad)
        spatial_feat = self.reshape(spatial_feat, reverse=True)

        spatial_temporal_feat = self.temporal_enc(spatial_feat)
        spatial_temporal_feat = self.reshape(spatial_temporal_feat)

        predicted_grad = self.decoder(spatial_temporal_feat, history_info)
        predicted_grad = self.reshape(predicted_grad, reverse=True)

        return predicted_grad


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


class SpatialEncoder(nn.Module):

    def __init__(self, g_hid_dim, s_hid_dim, s_block_num, enc_kernel_size):
        samplings = sampling_generator(s_block_num)
        super(SpatialEncoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC(g_hid_dim, s_hid_dim, enc_kernel_size, downsampling=samplings[0]),
            *[ConvSC(s_hid_dim, s_hid_dim, enc_kernel_size, downsampling=s) for s in samplings[1:]]
        )

    def forward(self, spatial_feat):
        history_info = self.enc[0](spatial_feat)
        latent = history_info
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, history_info


class Decoder(nn.Module):

    def __init__(self, s_hid_dim, g_hid_dim, s_block_num, dec_kernel_size):
        samplings = sampling_generator(s_block_num, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(s_hid_dim, s_hid_dim, dec_kernel_size, upsampling=s) for s in samplings[:-1]],
              ConvSC(s_hid_dim, s_hid_dim, dec_kernel_size, upsampling=samplings[-1])
        )
        self.readout = nn.Conv2d(s_hid_dim, g_hid_dim, 1)

    def forward(self, spatial_temporal_feat, history_info=None):
        for i in range(0, len(self.dec)-1):
            spatial_temporal_feat = self.dec[i](spatial_temporal_feat)
        predicted_grad = self.dec[-1](spatial_temporal_feat + history_info)
        predicted_grad = self.readout(predicted_grad)
        return predicted_grad


class TemporalEncoder(nn.Module):

    def __init__(self, channel_in, channel_hid, N2, incep_ker=[3,5,7,11], groups=8, **kwargs):
        super(TemporalEncoder, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        enc_layers = [gInception_ST(
            channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1,N2-1):
            enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers = [
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups)]
        for i in range(1,N2-1):
            dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_in,
                              incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2-1:
                skips.append(z)
        # decoder
        z = self.dec[0](z)
        for i in range(1,self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1) )

        y = z.reshape(B, T, C, H, W)
        return y


if __name__ == '__main__':
    model_args= {"s_hid_dim": 32,
                  "t_hid_dim": 256,
                  "s_block_num": 16,
                  "t_block_num": 16,
                  "model_type": 'incepu'}
    model = Predictor_Model(grad_shape=(10, 3, 32, 32), **model_args)
    grad = torch.randn(10, 3, 32, 32)
    # compute model parameter size
    total_para_size = 0
    for name, para in model.named_parameters():
        total_para_size += para.numel()
    # convert to M
    total_para_size = total_para_size / 1e6
    print(f"Total parameter size: {total_para_size}")