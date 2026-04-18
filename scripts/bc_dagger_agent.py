import numpy as np
import torch

import learning.base_agent as base_agent
import learning.mp_optimizer as mp_optimizer
import learning.ppo_model as ppo_model
import scripts.ppo_agent as ppo_agent
import util.mp_util as mp_util
import util.torch_util as torch_util


class BCDAggerAgent(base_agent.BaseAgent):
    NAME = "BC_DAGGER"

    def __init__(self, config, env, device):
        self._teacher_agent_config = config["teacher_agent_config"]
        self._teacher_model_file = config["teacher_model_file"]
        super().__init__(config, env, device)
        self._build_teacher()
        return

    def _load_params(self, config):
        super()._load_params(config)
        num_procs = mp_util.get_num_procs()
        self._exp_buffer_length = int(np.ceil(config["exp_buffer_size"] / num_procs))
        self._exp_buffer_length = max(self._exp_buffer_length, self._steps_per_iter)
        self._batch_size = int(np.ceil(config["batch_size"] / num_procs))
        self._update_epochs = config.get("update_epochs", 10)
        self._bc_pretrain_iters = config.get("bc_pretrain_iters", 5)
        self._dagger_teacher_prob = config.get("dagger_teacher_prob", 0.0)
        self._dagger_teacher_prob_decay = config.get("dagger_teacher_prob_decay", 1.0)
        self._entropy_coef = config.get("entropy_coef", 0.0)
        self._action_bound_weight = config.get("action_bound_weight", 0.0)
        return

    def _build_model(self, config):
        self._model = ppo_model.PPOModel(config["model"], self._env)
        return

    def _build_teacher(self):
        self._teacher = ppo_agent.PPOAgent(config=self._teacher_agent_config, env=self._env, device=self._device)
        self._teacher.load(self._teacher_model_file)
        self._teacher.eval()
        self._teacher.set_mode(base_agent.AgentMode.TEST)
        for p in self._teacher.parameters():
            p.requires_grad = False
        return

    def _build_optimizer(self, config):
        actor_params = list(self._model._actor_layers.parameters()) + list(self._model._action_dist.parameters())
        actor_params = [p for p in actor_params if p.requires_grad]
        self._optimizer = mp_optimizer.MPOptimizer(config["optimizer"], actor_params)
        return

    def _get_exp_buffer_length(self):
        return self._exp_buffer_length

    def _build_exp_buffer(self, config):
        super()._build_exp_buffer(config)
        n = self._get_exp_buffer_length()
        self._exp_buffer.add_buffer("expert_action", torch.zeros_like(self._exp_buffer.get_data("action")))
        self._exp_buffer.add_buffer("norm_obs", torch.zeros_like(self._exp_buffer.get_data("obs")))
        self._exp_buffer.add_buffer("norm_expert_action", torch.zeros_like(self._exp_buffer.get_data("action")))
        return

    def _init_iter(self):
        return

    def _teacher_action(self, obs, info):
        with torch.no_grad():
            a, _ = self._teacher._decide_action(obs, info)
        return a

    def _student_action(self, obs):
        norm_obs = self._obs_norm.normalize(obs)
        a_dist = self._model.eval_actor(norm_obs)
        if self._mode == base_agent.AgentMode.TRAIN:
            norm_a = a_dist.sample()
        else:
            norm_a = a_dist.mode
        a = self._a_norm.unnormalize(norm_a.detach())
        return a

    def _decide_action(self, obs, info):
        expert_action = self._teacher_action(obs, info)

        # TODO: aggregrate the dataset 

        if self._iter < self._bc_pretrain_iters:
            action = expert_action
        else:
            if np.random.rand() < self._dagger_teacher_prob:
                action = expert_action
            else:
                action = self._student_action(obs=obs)

        action_info = {"expert_action": expert_action}
        return action, action_info

    def _record_data_pre_step(self, obs, info, action, action_info):
        super()._record_data_pre_step(obs, info, action, action_info)
        self._exp_buffer.record("expert_action", action_info["expert_action"])
        return

    def _build_train_data(self):
        self.eval()
        obs = self._exp_buffer.get_data("obs")
        expert_action = self._exp_buffer.get_data("expert_action")
        norm_obs = self._obs_norm.normalize(obs)
        norm_expert_action = self._a_norm.normalize(expert_action)
        self._exp_buffer.set_data("norm_obs", norm_obs)
        self._exp_buffer.set_data("norm_expert_action", norm_expert_action)

        # decay optional teacher forcing after each aggregation round
        self._dagger_teacher_prob *= self._dagger_teacher_prob_decay
        return {"dataset_size": self._exp_buffer.get_sample_count()}

    def _update_model(self):
        self.train()
        num_samples = self._exp_buffer.get_sample_count()
        batch_size = min(self._batch_size, num_samples)
        num_batches = int(np.ceil(float(num_samples) / batch_size))
        train_info = {}

        for _ in range(self._update_epochs):
            for _ in range(num_batches):
                batch = self._exp_buffer.sample(batch_size)
                batch_info = self._update_actor(batch)
                torch_util.add_torch_dict(batch_info, train_info)

        torch_util.scale_torch_dict(1.0 / (self._update_epochs * num_batches), train_info)
        return train_info

    def _update_actor(self, batch):
        loss_info = self._calc_bc_loss(batch["norm_obs"], batch["norm_expert_action"])
        loss = loss_info["loss"]

        if self._action_bound_weight != 0:
            a_dist = self._model.eval_actor(batch["norm_obs"])
            action_bound_loss = self._compute_action_bound_loss(a_dist)
            if action_bound_loss is not None:
                action_bound_loss = torch.mean(action_bound_loss)
                loss = loss + self._action_bound_weight * action_bound_loss
                loss_info["action_bound_loss"] = action_bound_loss.detach()

        self._optimizer.step(loss)
        return loss_info

    def _calc_bc_loss(self, norm_obs, norm_expert_action):
        # TODO: implement max likelihood loss loss 

        a_dist = self._model.eval_actor(norm_obs)
        logp = a_dist.log_prob(norm_expert_action)

        if logp.ndim > 1:
            logp = logp.sum(dim=-1)
        
        nll = -torch.mean(logp)

        entropy = torch.mean(a_dist.entropy())
        if entropy.ndim > 0:
            entropy = torch.mean(entropy)

        loss = nll - self._entropy_coef * entropy

        # MSE
        pred_action = a_dist.mode
        mse = torch.mean((pred_action - norm_expert_action) ** 2)
        
        return {
            "loss": loss,
            "bc_nll": nll.detach(),
            "bc_mse": mse.detach(),
            "entropy": entropy.detach(),
        }
