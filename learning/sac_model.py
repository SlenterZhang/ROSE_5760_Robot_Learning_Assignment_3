import torch
import torch.distributions as D

import learning.base_model as base_model
import learning.nets.net_builder as net_builder
import util.torch_util as torch_util


class SACModel(base_model.BaseModel):
    def __init__(self, config, env):
        super().__init__(config, env)
        self._logstd_min = config.get("logstd_min", -5.0)
        self._logstd_max = config.get("logstd_max", 2.0)
        self._build_nets(config, env)
        return

    def actor_forward(self, obs):
        h = self._actor_layers(obs)
        mean = self._actor_mean(h)
        logstd = self._actor_logstd(h)
        logstd = torch.clamp(logstd, self._logstd_min, self._logstd_max)
        return mean, logstd

    def sample_action(self, obs, deterministic=False):
        mean, logstd = self.actor_forward(obs)
        std = torch.exp(logstd)
        # TODO:
        # mean, log_std
        # sample using reparameterization, if NOT deterministic, else, deterministic output
        # tanh squash    
        dist = D.Normal(mean, std)
        if deterministic:
            z = mean
        else:
            z = dist.rsample()
        norm_action = torch.tanh(z)

        logp = dist.log_prob(z).sum(dim=-1)
        logp -= torch.sum(torch.log(1 - norm_action.pow(2) + 1e-6), dim=-1)
        
        return norm_action, logp, torch.tanh(mean)

    def eval_q1(self, obs, norm_action):
        x = torch.cat([obs, norm_action], dim=-1)
        h = self._q1_layers(x)
        return self._q1_out(h)

    def eval_q2(self, obs, norm_action):
        x = torch.cat([obs, norm_action], dim=-1)
        h = self._q2_layers(x)
        return self._q2_out(h)

    def _build_nets(self, config, env):
        obs_space = env.get_obs_space()
        act_space = env.get_action_space()

        self._actor_layers, _ = net_builder.build_net(config["actor_net"], {"obs": obs_space}, activation=self._activation)
        actor_out = torch_util.calc_layers_out_size(self._actor_layers)
        act_dim = int(torch.tensor(act_space.shape).prod().item())
        init_scale = config.get("actor_init_output_scale", 0.01)
        self._actor_mean = torch.nn.Linear(actor_out, act_dim)
        self._actor_logstd = torch.nn.Linear(actor_out, act_dim)
        torch.nn.init.uniform_(self._actor_mean.weight, -init_scale, init_scale)
        torch.nn.init.zeros_(self._actor_mean.bias)
        torch.nn.init.uniform_(self._actor_logstd.weight, -init_scale, init_scale)
        torch.nn.init.zeros_(self._actor_logstd.bias)

        q_input = obs_space.shape[0] + act_dim
        q_space = type(obs_space)(low=-float("inf"), high=float("inf"), shape=(q_input,), dtype=obs_space.dtype)
        self._q1_layers, _ = net_builder.build_net(config["critic_net"], {"obs": q_space}, activation=self._activation)
        self._q2_layers, _ = net_builder.build_net(config["critic_net"], {"obs": q_space}, activation=self._activation)
        q1_out = torch_util.calc_layers_out_size(self._q1_layers)
        q2_out = torch_util.calc_layers_out_size(self._q2_layers)
        self._q1_out = torch.nn.Linear(q1_out, 1)
        self._q2_out = torch.nn.Linear(q2_out, 1)
        torch.nn.init.zeros_(self._q1_out.bias)
        torch.nn.init.zeros_(self._q2_out.bias)
        return
