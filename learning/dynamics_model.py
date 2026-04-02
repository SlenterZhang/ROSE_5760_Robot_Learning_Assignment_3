import gymnasium as gym
import numpy as np
import torch

import learning.base_model as base_model
import learning.nets.net_builder as net_builder
import util.torch_util as torch_util


class DynamicsModel(base_model.BaseModel):
    def __init__(self, config, env):
        super().__init__(config, env)
        self._build_nets(config, env)
        return

    def forward(self, obs, norm_action):
        x = torch.cat([obs, norm_action], dim=-1)
        h = self._layers(x)
        delta = self._delta_head(h)
        reward = self._reward_head(h).squeeze(-1)
        return delta, reward

    def _build_nets(self, config, env):
        obs_space = env.get_obs_space()
        act_space = env.get_action_space()
        act_dim = int(np.prod(act_space.shape))
        in_dim = obs_space.shape[0] + act_dim
        input_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(in_dim,), dtype=obs_space.dtype)
        self._layers, _ = net_builder.build_net(config["dyn_net"], {"obs": input_space}, activation=self._activation)
        out_size = torch_util.calc_layers_out_size(self._layers)
        obs_dim = int(np.prod(obs_space.shape))
        self._delta_head = torch.nn.Linear(out_size, obs_dim)
        self._reward_head = torch.nn.Linear(out_size, 1)
        torch.nn.init.zeros_(self._delta_head.bias)
        torch.nn.init.zeros_(self._reward_head.bias)
        return
