import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import sqrt

from policy.policies.base import Net
from policy.policies.actor import Actor, FF_Actor, LSTM_Actor

class blind_policy(Actor):
    def __init__(self, state_dim, action_dim, commond_dim, clock_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.command_dim = commond_dim
        self.clock_dim = clock_dim

        self.input_dim = state_dim + commond_dim + clock_dim
        self.output_dim = action_dim

        self.actor = LSTM_Actor(self.input_dim,self.output_dim)

    def forward(self, x, deterministic=True):
        self.action = self.actor(x, deterministic)
        return self.action

    def get_action(self):
            return self.action


class vision_module(Actor):
    def __init__(self, state_dim, action_dim, commond_dim, clock_dim, hidden_layer_dim, hight_map_dim, clock_action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.command_dim = commond_dim
        self.clock_dim = clock_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.hight_map_dim = hight_map_dim

        self.input_dim = state_dim + commond_dim + clock_dim
        self.motor_action_dim = action_dim
        self.clock_action_dim = clock_action_dim

        self.mlp_1 = FF_Actor(self.input_dim, self.hidden_layer_dim)
        self.mlp_2 = FF_Actor(self.hight_map_dim, self.hidden_layer_dim)

        tmp_dim = self.hidden_layer_dim * 2 + self.motor_action_dim
        self.lstm_actor = LSTM_Actor(tmp_dim, self.motor_action_dim + self.clock_action_dim)

    def forward(self, x, hight_map, blind_action, deterministic=True):
        hidden_state_1 = self.mlp_1(x)
        hidden_state_2 = self.mlp_2(hidden_state_1)

        hidden_state = torch.cat((hidden_state_1, hidden_state_2, blind_action), 1)
        self.action = self.lstm_actor(hidden_state, deterministic)

        return self.action

    def get_action(self):
        return self.action


class full_policy(Actor):
    def __init__(self, state_dim, action_dim, commond_dim, clock_dim,  hidden_layer_dim, hight_map_dim, clock_action_dim):
        super().__init__()
        self.blind_module = blind_policy(state_dim, action_dim, commond_dim, clock_dim)
        self.visual_module = vision_module(state_dim, action_dim, commond_dim, clock_dim, hidden_layer_dim, hight_map_dim, clock_action_dim)

    def forward(self, x, hight_map, deterministic=True):
        blind_action = self.blind_module(x)
        visual_action = self.visual_module(x, hight_map, blind_action)

        self.motor_action = visual_action[:, :self.motor_action_dim]
        self.clock_action = visual_action[:, self.motor_action_dim:]
        return self.visual_action

    def get_action(self):
        return self.motor_action

    def get_clock_action(self):
        return self.clock_action


