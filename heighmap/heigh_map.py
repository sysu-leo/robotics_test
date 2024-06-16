import torch
import torch.nn as nn
import torch.nn.functional as F

from res_net import ResNet_34
from unet import UNet


class mlp(torch.nn.Module):

    def __init__(self, num_i, num_h, num_o, hidden_layers = 1):
        super(mlp, self).__init__()

        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()
        self.hidden_layers = nn.ModuleList()
        for i in range(hidden_layers):
            self.hidden_layers.append(torch.nn.Linear(num_h, num_h))
            self.hidden_layers.append(torch.nn.ReLU())
        self.linear3 = torch.nn.Linear(num_h, num_o)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        for idx in range(self.hidden_layers):
            x = self.hidden_layers[idx * 2](x)
            x = self.hidden_layers[idx * 2 + 1](x)
        x = self.linear3(x)
        return x

class HeighMap(nn.Module):
    def __init__(self, depth_image_channel, robot_state_input_dim, robot_state_output_dim, lstm_layers = 2, depth_img_size = [224, 224], height_map_size = [30, 20]):
        super(HeighMap, self).__init__()
        self.depth_image_channel = depth_image_channel
        self.robot_state_input_dim = robot_state_input_dim
        self.robot_state_output_dim = robot_state_output_dim
        self.lstm_layers = lstm_layers
        self.depth_img_size = depth_img_size
        self.height_map_size = height_map_size

        self.depth_feature_size = depth_img_size[0] * depth_img_size[1] / 64
        self.resnet = ResNet_34()
        self.unet = UNet(1,1)
        self.lstm = nn.LSTM(input_size=self.robot_state_dim + self.depth_feature_size, hidden_size=height_map_size[0] * height_map_size[1], num_layers=self.lstm_layers)
        self.linear = nn.Linear(self.robot_state_dim + self.depth_feature_size, height_map_size[0] * height_map_size[1])
        self.mlp = mlp(self.robot_state_input_dim, self.robot_state_output_dim, self.lstm_layers)

    def init_hidden(self):
        # 一开始并没有隐藏状态所以我们要先初始化一个
        # 关于维度为什么这么设计请参考Pytoch相关文档
        # 各个维度的含义是 (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def getOutput(self):
        return self.lstm_out

    def get_hidden(self):
        return self.hidden

    def forward(self, depth_img,hidden_feature,  robot_state):
        depth_feature = self.resnet(depth_img)
        depth_feature_flat = depth_feature.view(depth_feature.size(0), -1)
        robot_state_feature = self.mlp(robot_state)
        robot_state_feature_flat = robot_state_feature.view(robot_state_feature.size(0), -1)
        fused_feature = torch.cat((depth_feature_flat, robot_state_feature_flat), 1).unsqueeze(0)
        self.lstm_out, self.hidden = self.lstm(fused_feature, hidden_feature)
        raw_hight_map = torch.reshape(self.linear(self.lstm_out), (depth_img.size(0), self.height_map_size[0], self.height_map_size[1]))

        refine_hight_map = self.unet(raw_hight_map)
        return refine_hight_map



