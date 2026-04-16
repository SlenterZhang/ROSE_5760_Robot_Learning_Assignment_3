import numpy as np
import torch

import envs.base_env as base_env
import learning.base_agent as base_agent
import learning.mp_optimizer as mp_optimizer
import learning.ppo_model as ppo_model
import util.mp_util as mp_util
import util.torch_util as torch_util


class PPOAgent(base_agent.BaseAgent):
    NAME = "PPO"

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        return

    def _load_params(self, config):
        super()._load_params(config)

        num_procs = mp_util.get_num_procs()
        self._batch_size = int(np.ceil(config["batch_size"] / num_procs))
        self._update_epochs = config["update_epochs"]
        self._critic_coef = config.get("critic_coef", 0.5)
        self._entropy_coef = config.get("entropy_coef", 0.0)
        self._clip_eps = config.get("clip_eps", 0.2)
        self._gae_lambda = config.get("gae_lambda", 0.95)
        self._norm_adv_clip = config.get("norm_adv_clip", 5.0)
        self._action_bound_weight = config.get("action_bound_weight", 0.0)
        return

    def _build_model(self, config):
        self._model = ppo_model.PPOModel(config["model"], self._env)
        return

    def _build_optimizer(self, config):
        actor_params = list(self._model._actor_layers.parameters()) + list(self._model._action_dist.parameters())
        actor_params = [p for p in actor_params if p.requires_grad]
        self._actor_optimizer = mp_optimizer.MPOptimizer(config["actor_optimizer"], actor_params)

        critic_params = list(self._model._critic_layers.parameters()) + list(self._model._critic_out.parameters())
        critic_params = [p for p in critic_params if p.requires_grad]
        self._critic_optimizer = mp_optimizer.MPOptimizer(config["critic_optimizer"], critic_params)
        return

    def _get_exp_buffer_length(self):
        return self._steps_per_iter

    def _build_exp_buffer(self, config):
        super()._build_exp_buffer(config)
        n = self._get_exp_buffer_length()
        self._exp_buffer.add_buffer("norm_obs", torch.zeros_like(self._exp_buffer.get_data("obs")))
        self._exp_buffer.add_buffer("norm_action", torch.zeros_like(self._exp_buffer.get_data("action")))
        self._exp_buffer.add_buffer("ret", torch.zeros([n], device=self._device, dtype=torch.float32))
        self._exp_buffer.add_buffer("adv", torch.zeros([n], device=self._device, dtype=torch.float32))
        self._exp_buffer.add_buffer("old_logp", torch.zeros([n], device=self._device, dtype=torch.float32))
        self._exp_buffer.add_buffer("old_val", torch.zeros([n], device=self._device, dtype=torch.float32))
        return

    def _init_iter(self):
        super()._init_iter()
        self._exp_buffer.reset()
        return

    def _decide_action(self, obs, info):
        norm_obs = self._obs_norm.normalize(obs)
        a_dist = self._model.eval_actor(norm_obs)

        if self._mode == base_agent.AgentMode.TRAIN:
            norm_a = a_dist.sample()
        else:
            norm_a = a_dist.mode

        a = self._a_norm.unnormalize(norm_a.detach())
        return a, {}

    def _build_train_data(self):
        self.eval()
        obs = self._exp_buffer.get_data("obs")
        next_obs = self._exp_buffer.get_data("next_obs")
        action = self._exp_buffer.get_data("action")
        reward = self._exp_buffer.get_data("reward")
        done = self._exp_buffer.get_data("done")

        norm_obs = self._obs_norm.normalize(obs)
        norm_next_obs = self._obs_norm.normalize(next_obs)
        norm_action = self._a_norm.normalize(action)

        with torch.no_grad():
            old_dist = self._model.eval_actor(norm_obs)
            old_logp = old_dist.log_prob(norm_action)
            old_val = self._model.eval_critic(norm_obs).squeeze(-1)
            next_val = self._model.eval_critic(norm_next_obs).squeeze(-1)

        adv, ret = self._calc_gae(reward, done, old_val, next_val)
        adv_std, adv_mean = torch.std_mean(adv)
        adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-5)
        adv = torch.clamp(adv, -self._norm_adv_clip, self._norm_adv_clip)

        self._exp_buffer.set_data("norm_obs", norm_obs)
        self._exp_buffer.set_data("norm_action", norm_action)
        self._exp_buffer.set_data("old_logp", old_logp)
        self._exp_buffer.set_data("old_val", old_val)
        self._exp_buffer.set_data("adv", adv)
        self._exp_buffer.set_data("ret", ret)

        return {"adv_mean": adv_mean, "adv_std": adv_std}


    def _update_model(self):
        self.train()
        num_samples = self._exp_buffer.get_sample_count()
        batch_size = self._batch_size
        num_batches = int(np.ceil(float(num_samples) / batch_size))
        train_info = {}

        for _ in range(self._update_epochs):
            for _ in range(num_batches):
                batch = self._exp_buffer.sample(batch_size)
                critic_info = self._update_critic(batch)
                actor_info = self._update_actor(batch)
                torch_util.add_torch_dict(critic_info, train_info)
                torch_util.add_torch_dict(actor_info, train_info)

        torch_util.scale_torch_dict(1.0 / (self._update_epochs * num_batches), train_info)
        return train_info

    def _update_critic(self, batch):
        loss = self._calc_critic_loss(batch["norm_obs"], batch["ret"])
        self._critic_optimizer.step(loss)
        return {"critic_loss": loss}

    def _update_actor(self, batch):
        loss_info = self._calc_actor_loss(batch["norm_obs"], batch["norm_action"], batch["old_logp"], batch["adv"])
        loss = loss_info["actor_loss"]

        if self._action_bound_weight != 0:
            a_dist = self._model.eval_actor(batch["norm_obs"])
            action_bound_loss = self._compute_action_bound_loss(a_dist)
            if action_bound_loss is not None:
                action_bound_loss = torch.mean(action_bound_loss)
                loss = loss + self._action_bound_weight * action_bound_loss
                loss_info["action_bound_loss"] = action_bound_loss.detach()

        self._actor_optimizer.step(loss)
        return loss_info

    def _calc_gae(self, reward, done, value, next_value):
        T = reward.shape[0]
        adv = torch.zeros_like(reward)
        ret = torch.zeros_like(reward)
        last_gae = torch.zeros([], device=self._device)

        # TODO: implement GAE
        for t in reversed(range(T)):
            not_done = 1.0 - done[t].float()
            delta = reward[t] + self._discount * next_value[t] * not_done - value[t]
            last_gae = delta + self._discount * self._gae_lambda * not_done * last_gae
            adv[t] = last_gae
            ret[t] = adv[t] + value[t]

        return adv, ret

    def _calc_critic_loss(self, norm_obs, ret):
        # TODO: MSE 
        value = self._model.eval_critic(norm_obs).squeeze(-1)
        critic_loss = torch.mean((value - ret) ** 2)

        return critic_loss

    def _calc_actor_loss(self, norm_obs, norm_action, old_logp, adv):
        # TODO:
        # ratio = exp(logp - old_logp)
        # clipped ratio
        # min(...)
        a_dist = self._model.eval_actor(norm_obs)
        logp = a_dist.log_prob(norm_action)
        ratio = torch.exp(logp - old_logp)
        clipped_ratio = torch.clamp(ratio, 1.0 - self._clip_eps, 1.0 + self._clip_eps)

        surr1 = ratio * adv
        surr2 = clipped_ratio * adv
        actor_loss = -torch.mean(torch.min(surr1, surr2))

        entropy = torch.mean(a_dist.entropy())
        loss = actor_loss - self._entropy_coef * entropy

        return {
            "actor_loss": loss,
            "ppo_loss": actor_loss.detach(),
            "entropy": entropy.detach(),
            "clip_frac": torch.mean((torch.abs(ratio - 1.0) > self._clip_eps).float()).detach(),
        }
