import numpy as np
import torch

import envs.base_env as base_env
import learning.base_agent as base_agent
import learning.mp_optimizer as mp_optimizer
import learning.sac_model as sac_model
import util.mp_util as mp_util
import util.torch_util as torch_util


class SACAgent(base_agent.BaseAgent):
    NAME = "SAC"

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        self._hard_sync_target()
        return

    def _load_params(self, config):
        super()._load_params(config)
        num_procs = mp_util.get_num_procs()
        self._exp_buffer_length = int(np.ceil(config["exp_buffer_size"] / num_procs))
        self._exp_buffer_length = max(self._exp_buffer_length, self._steps_per_iter)
        self._batch_size = int(np.ceil(config["batch_size"] / num_procs))
        self._init_samples = int(np.ceil(config["init_samples"] / num_procs))
        self._updates_per_iter = config["updates_per_iter"]
        self._actor_update_interval = config.get("actor_update_interval", 1)
        self._target_update_interval = config.get("target_update_interval", 1)
        self._tau = config.get("tau", 0.005)
        self._alpha = config.get("alpha", 0.2)
        return

    def _build_model(self, config):
        model_cfg = config["model"]
        self._model = sac_model.SACModel(model_cfg, self._env)
        self._tar_model = sac_model.SACModel(model_cfg, self._env)
        for p in self._tar_model.parameters():
            p.requires_grad = False
        return

    def _build_optimizer(self, config):
        actor_params = list(self._model._actor_layers.parameters()) + list(self._model._actor_mean.parameters()) + list(self._model._actor_logstd.parameters())
        actor_params = [p for p in actor_params if p.requires_grad]
        self._actor_optimizer = mp_optimizer.MPOptimizer(config["actor_optimizer"], actor_params)

        critic_params = list(self._model._q1_layers.parameters()) + list(self._model._q1_out.parameters())                       + list(self._model._q2_layers.parameters()) + list(self._model._q2_out.parameters())
        critic_params = [p for p in critic_params if p.requires_grad]
        self._critic_optimizer = mp_optimizer.MPOptimizer(config["critic_optimizer"], critic_params)
        return

    def _get_exp_buffer_length(self):
        return self._exp_buffer_length

    def _init_train(self):
        super()._init_train()
        self.eval()
        self.set_mode(base_agent.AgentMode.TRAIN)
        self._rollout_train(self._init_samples)
        return

    def _decide_action(self, obs, info):
        # Use uniform random exploration before the replay buffer has enough data.
        if self._mode == base_agent.AgentMode.TRAIN and self._exp_buffer.get_total_samples() < self._init_samples:
            a_space = self._env.get_action_space()
            low = torch.as_tensor(a_space.low, device=self._device, dtype=obs.dtype)
            high = torch.as_tensor(a_space.high, device=self._device, dtype=obs.dtype)
            a = low + torch.rand_like(low) * (high - low)
            return a, {}

        norm_obs = self._obs_norm.normalize(obs)
        if self._mode == base_agent.AgentMode.TRAIN:
            norm_a, _, _ = self._model.sample_action(norm_obs, deterministic=False)
        else:
            norm_a, _, _ = self._model.sample_action(norm_obs, deterministic=True)

        a = self._a_norm.unnormalize(norm_a.detach())
        a = torch.clamp(a, min=torch.as_tensor(self._env.get_action_space().low, device=self._device),
                           max=torch.as_tensor(self._env.get_action_space().high, device=self._device))
        return a, {}

    def _update_model(self):
        self.train()
        info = {}
        sample_count = self._exp_buffer.get_sample_count()
        if sample_count < self._batch_size:
            return info

        for i in range(self._updates_per_iter):
            batch = self._exp_buffer.sample(self._batch_size)
            critic_info = self._update_critic(batch)
            torch_util.add_torch_dict(critic_info, info)

            if i % self._actor_update_interval == 0:
                actor_info = self._update_actor(batch)
                torch_util.add_torch_dict(actor_info, info)

            if i % self._target_update_interval == 0:
                self._soft_sync_target()

        torch_util.scale_torch_dict(1.0 / self._updates_per_iter, info)
        return info

    def _update_critic(self, batch):
        loss_info = self._compute_q_loss(batch)
        self._critic_optimizer.step(loss_info["critic_loss"])
        return loss_info

    def _update_actor(self, batch):
        loss_info = self._compute_actor_loss(batch)
        self._actor_optimizer.step(loss_info["actor_loss"])
        return loss_info

    def _compute_q_loss(self, batch):
        norm_obs = self._obs_norm.normalize(batch["obs"])
        norm_next_obs = self._obs_norm.normalize(batch["next_obs"])
        norm_action = self._a_norm.normalize(batch["action"])
        reward = batch["reward"].squeeze(-1) if batch["reward"].ndim > 1 else batch["reward"]
        done = batch["done"]

        # TODO:
        # mse(Q1, target)
        # mse(Q2, target)

        return {
            "critic_loss": q1_loss + q2_loss,
            "q1_loss": q1_loss.detach(),
            "q2_loss": q2_loss.detach(),
            "target_q": torch.mean(target).detach(),
            "pred_q1": torch.mean(pred_q1).detach(),
            "pred_q2": torch.mean(pred_q2).detach(),
        }

    def _compute_actor_loss(self, batch):
        # TODO:
        # maximize Q - alpha * entropy

        return {
            "actor_loss": actor_loss,
            "policy_logp": torch.mean(logp).detach(),
            "policy_q": torch.mean(q).detach(),
        }

    def _hard_sync_target(self):
        for src, tar in zip(self._model.parameters(), self._tar_model.parameters()):
            tar.data.copy_(src.data)
        return

    def _soft_sync_target(self):
        # TODO:
        # target = tau * online + (1-tau) * target        
        return
