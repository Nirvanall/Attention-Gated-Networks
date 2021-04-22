import torch.nn as nn
import torch
from .utils import UnetConv3, UnetConv3_He, UnetUp3_He, UnetGridGatingSignal3, UnetDsv3
import torch.nn.functional as F
from models.layers.grid_attention_layer import GridAttentionBlock3D
from models.networks_other import init_weights

class unet_grid_attention_dsv_3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3,
                 nonlocal_mode='concatenation', attention_dsample=(2,2,2), is_batchnorm=True):
        super(unet_grid_attention_dsv_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3_He(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2,2,2))

        self.conv2 = UnetConv3_He(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2,2,2))

        self.conv3 = UnetConv3_He(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2,2,2))

        self.conv4 = UnetConv3_He(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2,2,2))

        self.center = UnetConv3_He(filters[3], filters[4], self.is_batchnorm)
        self.gating = UnetGridGatingSignal3(filters[4], filters[3], kernel_size=(1, 1, 1), is_batchnorm=self.is_batchnorm)

        # attention blocks
        self.attentionblock2 = GridAttentionBlock3D(in_channels=filters[1], gating_channels=filters[3],
                                                    inter_channels=filters[1], sub_sample_factor=attention_dsample, mode=nonlocal_mode)
        self.attentionblock3 = GridAttentionBlock3D(in_channels=filters[2], gating_channels=filters[3],
                                                    inter_channels=filters[2], sub_sample_factor=attention_dsample, mode=nonlocal_mode)
        self.attentionblock4 = GridAttentionBlock3D(in_channels=filters[3], gating_channels=filters[3],
                                                    inter_channels=filters[3], sub_sample_factor=attention_dsample, mode=nonlocal_mode)

        # upsampling
        self.up_concat4 = UnetUp3_He(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = UnetUp3_He(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_concat2 = UnetUp3_He(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = UnetUp3_He(filters[1], filters[0], self.is_deconv, self.is_batchnorm)
        
        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=8)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=4)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = nn.Conv3d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)

        # final conv (without any concat)
        self.final = nn.Conv3d(n_classes*4, n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up_concat4(g_conv4, center)
        g_conv3, att3 = self.attentionblock3(conv3, gating)
        up3 = self.up_concat3(g_conv3, up4)
        g_conv2, att2 = self.attentionblock2(conv2, gating)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        
        # Deep supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        final = self.final(torch.cat([dsv1,dsv2,dsv3,dsv4], dim=1))


        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p













