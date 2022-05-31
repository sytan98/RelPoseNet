import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class RelPoseNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone, self.concat_layer = self._get_backbone()
        self.net_q_fc = nn.Linear(self.concat_layer.in_features, 4)
        self.net_t_fc = nn.Linear(self.concat_layer.in_features, 3)
        self.dropout = nn.Dropout(0.3)

    def _get_backbone(self):
        backbone, concat_layer = None, None
        if self.cfg.backbone_net == 'resnet34':
            backbone = models.resnet34(pretrained=True)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            concat_layer = nn.Linear(2 * in_features, 2 * in_features)
        return backbone, concat_layer

    def _forward_one(self, x):
        x = self.backbone(x)
        x = x.view(x.size()[0], -1)
        return x

    def forward(self, x1, x2):
        feat1 = self._forward_one(x1)
        feat2 = self._forward_one(x2)

        feat = torch.cat((feat1, feat2), 1)
        q_est = self.net_q_fc(self.dropout(self.concat_layer(feat)))
        t_est = self.net_t_fc(self.dropout(self.concat_layer(feat)))
        return q_est, t_est

class RelPoseNetWithSingleOutput(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone, self.concat_layer = self._get_backbone()
        self.fc_layer2 = nn.Linear(self.concat_layer.in_features, self.concat_layer.in_features)
        self.fc_layer3 = nn.Linear(self.concat_layer.in_features, self.concat_layer.in_features)
        self.net_t_fc = nn.Linear(self.concat_layer.in_features, 1)
        self.relu_activation = nn.ReLU()

    def _get_backbone(self):
        backbone, concat_layer = None, None
        if self.cfg.backbone_net == 'resnet34':
            backbone = models.resnet34(pretrained=True)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            concat_layer = nn.Linear(2 * in_features, 2 * in_features)
        elif self.cfg.backbone_net == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            concat_layer = nn.Linear(2 * in_features, 2 * in_features)
        return backbone, concat_layer

    def _forward_one(self, x):
        x = self.backbone(x)
        x = x.view(x.size()[0], -1)
        return x

    def forward(self, x1, x2):
        feat1 = self._forward_one(x1)
        feat2 = self._forward_one(x2)

        feat = torch.cat((feat1, feat2), 1)
        fc_output1 = self.concat_layer(self.relu_activation(feat))
        fc_output2 = self.fc_layer2(self.relu_activation(fc_output1))
        fc_output3 = self.fc_layer3(self.relu_activation(fc_output2))
        t_est = self.net_t_fc(self.relu_activation(fc_output3))
        return t_est

class RelPoseNetLarger(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone, self.concat_layer = self._get_backbone()
        self.fc_layer2 = nn.Linear(self.concat_layer.in_features, self.concat_layer.in_features)
        self.fc_layer3 = nn.Linear(self.concat_layer.in_features, self.concat_layer.in_features)
        self.net_q_fc = nn.Linear(self.concat_layer.in_features, 4)
        self.net_t_fc = nn.Linear(self.concat_layer.in_features, 3)
        self.dropout = nn.Dropout(0.3)

    def _get_backbone(self):
        backbone, concat_layer = None, None
        if self.cfg.backbone_net == 'resnet34':
            backbone = models.resnet34(pretrained=True)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            concat_layer = nn.Linear(2 * in_features, 2 * in_features)
        return backbone, concat_layer

    def _forward_one(self, x):
        x = self.backbone(x)
        x = x.view(x.size()[0], -1)
        return x

    def forward(self, x1, x2):
        feat1 = self._forward_one(x1)
        feat2 = self._forward_one(x2)

        feat = torch.cat((feat1, feat2), 1)
        fc_output1 = self.dropout(self.concat_layer(feat))
        fc_output2 = self.dropout(self.fc_layer2(fc_output1))
        fc_output3 = self.dropout(self.fc_layer3(fc_output2))
        q_est = self.net_q_fc(fc_output3)
        t_est = self.net_t_fc(fc_output3)
        return q_est, t_est

class RelPoseNetWithAccel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.imu_features = 7

        self.backbone, self.concat_layer = self._get_backbone()
        self.net_imu_fc = nn.Linear(self.imu_features, 10)
        self.concat_img_imu_layer = nn.Linear(self.net_imu_fc.out_features + self.concat_layer.out_features, 
                                              self.net_imu_fc.out_features + self.concat_layer.out_features)

        self.net_q_fc = nn.Linear(self.concat_layer.in_features, 4)
        self.net_t_fc = nn.Linear(self.concat_layer.in_features, 3)
        self.net_t_imu_fc = nn.Linear(self.concat_img_imu_layer.out_features, 3)
        self.dropout = nn.Dropout(0.3)

    def _get_backbone(self):
        backbone, concat_layer = None, None
        if self.cfg.backbone_net == 'resnet34':
            backbone = models.resnet34(pretrained=True)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            concat_layer = nn.Linear(2 * in_features, 2 * in_features)
        return backbone, concat_layer

    def _forward_one(self, x):
        x = self.backbone(x)
        x = x.view(x.size()[0], -1)
        return x

    def forward(self, x1, x2, x3):
        feat1 = self._forward_one(x1)
        feat2 = self._forward_one(x2)
        feat3 = self.net_imu_fc(x3)

        feat = torch.cat((feat1, feat2), 1)
        intermediate_fc_output = self.concat_layer(feat)
        q_est = self.net_q_fc(self.dropout(intermediate_fc_output))
        t_est = self.net_t_fc(self.dropout(intermediate_fc_output))

        feat_with_accel = torch.cat((feat3, intermediate_fc_output), 1)
        t_imu_est = self.net_t_imu_fc(self.dropout(feat_with_accel))
        return q_est, t_est, t_imu_est

class RelPoseNetWithIMU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.imu_features = 10

        self.backbone, self.concat_layer = self._get_backbone()
        self.net_imu_fc = nn.Linear(self.imu_features, 10)
        self.concat_img_imu_layer = nn.Linear(self.net_imu_fc.out_features + self.concat_layer.out_features, 
                                              self.net_imu_fc.out_features + self.concat_layer.out_features)

        self.net_q_fc = nn.Linear(self.concat_layer.in_features, 4)
        self.net_t_fc = nn.Linear(self.concat_layer.in_features, 3)
        self.net_q_imu_fc = nn.Linear(self.concat_img_imu_layer.out_features, 4)
        self.net_t_imu_fc = nn.Linear(self.concat_img_imu_layer.out_features, 3)
        self.dropout = nn.Dropout(0.3)

    def _get_backbone(self):
        backbone, concat_layer = None, None
        if self.cfg.backbone_net == 'resnet34':
            backbone = models.resnet34(pretrained=True)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            concat_layer = nn.Linear(2 * in_features, 2 * in_features)
        return backbone, concat_layer

    def _forward_one(self, x):
        x = self.backbone(x)
        x = x.view(x.size()[0], -1)
        return x

    def forward(self, x1, x2, x3):
        feat1 = self._forward_one(x1)
        feat2 = self._forward_one(x2)
        feat3 = self.net_imu_fc(x3)

        feat = torch.cat((feat1, feat2), 1)
        intermediate_fc_output = self.concat_layer(feat)
        q_est = self.net_q_fc(self.dropout(intermediate_fc_output))
        t_est = self.net_t_fc(self.dropout(intermediate_fc_output))

        feat_with_accel = torch.cat((feat3, intermediate_fc_output), 1)
        t_imu_est = self.net_t_imu_fc(self.dropout(feat_with_accel))
        q_imu_est = self.net_q_imu_fc(self.dropout(feat_with_accel))
        return q_est, t_est, q_imu_est, t_imu_est


def get_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
):
    return (
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.Dropout(0.2),
        nn.ReLU(),
    )


class MeNet(nn.Module):
    def __init__(self, outputs) -> None:
        super().__init__()
        self.output_count = outputs
        self.conv_encoder = nn.Sequential(
            *get_conv_block(6, 16, 7, 2, 3),
            *get_conv_block(16, 32, 5, 2, 2),
            *get_conv_block(32, 64, 3, 2, 1),
            *get_conv_block(64, 64, 3, 1, 1),
            *get_conv_block(64, 128, 3, 2, 1),
            *get_conv_block(128, 128, 3, 1, 1),
            *get_conv_block(128, 256, 3, 2, 1),
            *get_conv_block(256, 256, 3, 1, 1),
            *get_conv_block(256, 512, 3, 2, 1),
        )

        self.linear_encoder = nn.Sequential(
            nn.Linear(8192, 4092),
            nn.ReLU(),
            nn.Linear(4092, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, outputs),
        )

    def forward(self, x1, x2):
        """
        Input shape should be [Batch x 2 x Channels x Width x Height]
        """
        input = torch.cat((x1, x2), 1)
        output = self.conv_encoder(input)
        output = output.view(-1, output.size()[1] * output.size()[2] * output.size()[3])
        output = self.linear_encoder(output)

        if self.output_count < 7:
            return output
        else:
            return output[:, :4], output[:, 4:]
