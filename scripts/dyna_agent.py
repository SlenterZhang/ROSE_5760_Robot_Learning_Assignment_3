import numpy as np
import torch

import envs.base_env as base_env
import learning.base_agent as base_agent
import learning.experience_buffer as experience_buffer
import learning.mp_optimizer as mp_optimizer
import learning.dynamics_model as dynamics_model
import learning.sac_model as sac_model
import util.mp_util as mp_util
import util.torch_util as torch_util


class DYNAAgent(base_agent.BaseAgent):
    NAME = "DYNA"

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        self._hard_sync_target()
        return

    def _load_params(self, config):
        super()._load_params(config)
        num_procs = mp_util.get_num_procs()
        self._exp_buffer_length = int(np.ceil(config["exp_buffer_size"] / num_procs))
        self._exp_buffer_length = max(self._exp_buffer_length, self._steps_per_iter)
        self._model_buffer_length = int(np.ceil(config["model_buffer_size"] / num_procs))
        self._batch_size = int(np.ceil(config["batch_size"] / num_procs))
        self._init_samples = int(np.ceil(config["init_samples"] / num_procs))
        self._updates_per_iter = config["updates_per_iter"]
        self._dyn_updates_per_iter = config.get("dyn_updates_per_iter", self._updates_per_iter)
        self._rollout_horizon = config.get("rollout_horizon", 1)
        self._model_rollouts_per_iter = config.get("model_rollouts_per_iter", 64)
        self._real_ratio = config.get("real_ratio", 0.8)
        self._tau = config.get("tau", 0.005)
        self._alpha = config.get("alpha", 0.1)
        return

    def _build_model(self, config):
        self._model = sac_model.SACModel(config["model"], self._env)
        self._tar_model = sac_model.SACModel(config["model"], self._env)
        for p in self._tar_model.parameters():
            p.requires_grad = False
        self._dyn_model = dynamics_model.DynamicsModel(config["dynamics_model"], self._env)
        return

    def _build_optimizer(self, config):
        actor_params = list(self._model._actor_layers.parameters()) + list(self._model._actor_mean.parameters()) + list(self._model._actor_logstd.parameters())
        actor_params = [p for p in actor_params if p.requires_grad]
        self._actor_optimizer = mp_optimizer.MPOptimizer(config["actor_optimizer"], actor_params)

        critic_params = list(self._model._q1_layers.parameters()) + list(self._model._q1_out.parameters()) \
                      + list(self._model._q2_layers.parameters()) + list(self._model._q2_out.parameters())
        critic_params = [p for p in critic_params if p.requires_grad]
        self._critic_optimizer = mp_optimizer.MPOptimizer(config["critic_optimizer"], critic_params)

        dyn_params = [p for p in self._dyn_model.parameters() if p.requires_grad]
        self._dyn_optimizer = mp_optimizer.MPOptimizer(config["dynamics_optimizer"], dyn_params)
        return

    def _get_exp_buffer_length(self):
        return self._exp_buffer_length

    def _build_exp_buffer(self, config):
        super()._build_exp_buffer(config)
        self._model_buffer = experience_buffer.ExperienceBuffer(self._model_buffer_length, self._device)
        self._model_buffer.add_buffer("obs", torch.zeros([self._model_buffer_length] + list(self._exp_buffer.get_data("obs").shape[1:]), device=self._device, dtype=self._exp_buffer.get_data("obs").dtype))
        self._model_buffer.add_buffer("next_obs", torch.zeros([self._model_buffer_length] + list(self._exp_buffer.get_data("next_obs").shape[1:]), device=self._device, dtype=self._exp_buffer.get_data("next_obs").dtype))
        self._model_buffer.add_buffer("action", torch.zeros([self._model_buffer_length] + list(self._exp_buffer.get_data("action").shape[1:]), device=self._device, dtype=self._exp_buffer.get_data("action").dtype))
        self._model_buffer.add_buffer("reward", torch.zeros([self._model_buffer_length], device=self._device, dtype=self._exp_buffer.get_data("reward").dtype))
        self._model_buffer.add_buffer("done", torch.zeros([self._model_buffer_length], device=self._device, dtype=self._exp_buffer.get_data("done").dtype))
        return

    def _init_train(self):
        super()._init_train()
        self.eval()
        self.set_mode(base_agent.AgentMode.TRAIN)
        self._rollout_train(self._init_samples)
        return

    def _decide_action(self, obs, info):
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
        a = torch.clamp(a,
                        min=torch.as_tensor(self._env.get_action_space().low, device=self._device),
                        max=torch.as_tensor(self._env.get_action_space().high, device=self._device))
        return a, {}

    def _update_model(self):
        self.train()
        info = {}
        sample_count = self._exp_buffer.get_sample_count()
        if sample_count < self._batch_size:
            return info

        for _ in range(self._dyn_updates_per_iter):
            batch = self._exp_buffer.sample(self._batch_size)
            dyn_info = self._update_dynamics(batch)
            torch_util.add_torch_dict(dyn_info, info)

        self._generate_model_rollouts()

        for _ in range(self._updates_per_iter):
            critic_batch = self._sample_mixed_batch(self._batch_size)
            critic_info = self._update_critic(critic_batch)
            torch_util.add_torch_dict(critic_info, info)

            # Update the actor only on real states to reduce model-bias feedback.
            actor_batch = self._exp_buffer.sample(self._batch_size)
            actor_info = self._update_actor(actor_batch)
            torch_util.add_torch_dict(actor_info, info)
            self._soft_sync_target()

        denom = self._dyn_updates_per_iter + self._updates_per_iter
        torch_util.scale_torch_dict(1.0 / max(1, denom), info)
        return info

    def _update_dynamics(self, batch):
        norm_obs = self._obs_norm.normalize(batch["obs"])
        norm_action = self._a_norm.normalize(batch["action"])
        
        # TODO:
        # loss = mse(predicted next state, true next state)
        
        dyn_loss = delta_loss + reward_loss
        self._dyn_optimizer.step(dyn_loss)
        return {
            "dyn_loss": dyn_loss.detach(),
            "dyn_delta_loss": delta_loss.detach(),
            "dyn_reward_loss": reward_loss.detach(),
        }

    def _generate_model_rollouts(self):
        if self._exp_buffer.get_sample_count() == 0:
            return

        n = min(self._model_rollouts_per_iter, self._exp_buffer.get_sample_count())
        seed_batch = self._exp_buffer.sample(n)
        obs = seed_batch["obs"]
        a_low = torch.as_tensor(self._env.get_action_space().low, device=self._device, dtype=obs.dtype)
        a_high = torch.as_tensor(self._env.get_action_space().high, device=self._device, dtype=obs.dtype)

        for _ in range(self._rollout_horizon):
            norm_obs = self._obs_norm.normalize(obs)
            with torch.no_grad():
                # TODO:
                # sample real states
                # rollout using learned model
                # store synthetic transitions
                done = torch.full([obs.shape[0]], base_env.DoneFlags.NULL.value, device=self._device, dtype=torch.int)

            for i in range(obs.shape[0]):
                self._model_buffer.record("obs", obs[i])
                self._model_buffer.record("action", action[i])
                self._model_buffer.record("next_obs", next_obs[i])
                self._model_buffer.record("reward", pred_reward[i])
                self._model_buffer.record("done", done[i])
                self._model_buffer.inc()

            obs = next_obs.detach()
        return

    def _sample_mixed_batch(self, batch_size):
        real_n = int(round(batch_size * self._real_ratio))
        real_n = min(max(1, real_n), batch_size)
        model_n = batch_size - real_n

        real_batch = self._exp_buffer.sample(real_n)
        if self._model_buffer.get_sample_count() == 0 or model_n <= 0:
            return real_batch

        model_n = min(model_n, self._model_buffer.get_sample_count())
        model_batch = self._model_buffer.sample(model_n)
        batch = {} 
        # TODO: mix real + model data
        return batch

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
        # TODO: actor loss 

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
