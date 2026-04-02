import torch

import learning.base_model as base_model
import learning.nets.net_builder as net_builder
import util.torch_util as torch_util


class PPOModel(base_model.BaseModel):
    def __init__(self, config, env):
        super().__init__(config, env)
        self._build_nets(config, env)
        return

    def eval_actor(self, obs):
        h = self._actor_layers(obs)
        return self._action_dist(h)

    def eval_critic(self, obs):
        h = self._critic_layers(obs)
        return self._critic_out(h)

    def _build_nets(self, config, env):
        self._build_actor(config, env)
        self._build_critic(config, env)
        return

    def _build_actor(self, config, env):
        net_name = config["actor_net"]
        input_dict = {"obs": env.get_obs_space()}
        self._actor_layers, _ = net_builder.build_net(net_name, input_dict, activation=self._activation)
        self._action_dist = self._build_action_distribution(config, env, self._actor_layers)
        return

    def _build_critic(self, config, env):
        net_name = config["critic_net"]
        input_dict = {"obs": env.get_obs_space()}
        self._critic_layers, _ = net_builder.build_net(net_name, input_dict, activation=self._activation)
        out_size = torch_util.calc_layers_out_size(self._critic_layers)
        self._critic_out = torch.nn.Linear(out_size, 1)
        torch.nn.init.zeros_(self._critic_out.bias)
        return
