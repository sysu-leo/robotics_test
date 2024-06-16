import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import sqrt

from policy.policies.base import Net

LOG_STD_HI = -1.5
LOG_STD_LO = -20

class Actor(Net):
    def __init__(self):
        super(Actor, self).__init__()

    def forward(self):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError

class Linear_Actor(Actor):
    def __init__(self, state_dim, action_dim, hidden_size=32):
        super(Linear_Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, action_dim)

        self.action_dim = action_dim

        for p in self.parameters():
            p.data = torch.zeros(p.shape)

    def forward(self, state):
        a = self.l1(state)
        a = self.l2(a)
        self.action = a
        return a

    def get_action(self):
        return self.action

class FF_Actor(Actor):
    def __init__(self, state_dim, action_dim, layers=(256, 256), env_name=None, nonlinearity=F.relu, max_action=1):
        super(FF_Actor, self).__init__()

        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.Linear(state_dim, layers[0])]
        for i in range(len(layers)-1):
            self.actor_layers += [nn.Linear(layers[i], layers[i+1])]
        self.network_out = nn.Linear(layers[-1], action_dim)

        self.action = None
        self.action_dim = action_dim
        self.env_name = env_name
        self.nonlinearity = nonlinearity

        self.initialize_parameters()

        self.max_action = max_action

    def forward(self, state, deterministic=True):
        x = state
        for idx, layer in enumerate(self.actor_layers):
            x = self.nonlinearity(layer(x))

        self.action = torch.tanh(self.network_out(x))
        return self.action * self.max_action

    def get_action(self):
        return self.action


class LSTM_Actor(Actor):
    def __init__(self, state_dim, action_dim, layers=(128, 128), env_name=None, nonlinearity=torch.tanh, max_action=1):
        super(LSTM_Actor, self).__init__()

        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.LSTMCell(state_dim, layers[0])]
        for i in range(len(layers)-1):
            self.actor_layers += [nn.LSTMCell(layers[i], layers[i+1])]
        self.network_out = nn.Linear(layers[i-1], action_dim)

        self.action = None
        self.action_dim = action_dim
        self.init_hidden_state()
        self.env_name = env_name
        self.nonlinearity = nonlinearity

        self.is_recurrent = True

        self.max_action = max_action

    def get_hidden_state(self):
        return self.hidden, self.cells

    def set_hidden_state(self, data):
        if len(data) != 2:
            print("Got invalid hidden state data.")
            exit(1)

        self.hidden, self.cells = data

    def init_hidden_state(self, batch_size=1):
        self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]
        self.cells = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]

    def forward(self, x, deterministic=True):
        dims = len(x.size())

        if dims == 3:  # if we get a batch of trajectories
            self.init_hidden_state(batch_size=x.size(1))
            y = []
            for t, x_t in enumerate(x):
                for idx, layer in enumerate(self.actor_layers):
                    c, h = self.cells[idx], self.hidden[idx]
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
                    x_t = self.hidden[idx]
                y.append(x_t)
            x = torch.stack([x_t for x_t in y])

        else:
            if dims == 1:  # if we get a single timestep (if not, assume we got a batch of single timesteps)
                x = x.view(1, -1)

            for idx, layer in enumerate(self.actor_layers):
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))
                x = self.hidden[idx]
            x = self.nonlinearity(self.network_out(x))

            if dims == 1:
                x = x.view(-1)

        self.action = self.network_out(x)
        return self.action

    def get_action(self):
        return self.action

def normc_fn(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)
