# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import division

import json
import os
import sys
import numpy as np
import random
import time
import math
import collections
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.distributions as D
from torch import optim
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR

from utils import padding_idx
from agent import BaseAgent
from oracle import make_oracle
from ask_agent import AskAgent
from verbal_ask_agent import VerbalAskAgent

ENCODE_MAX_LENGTH = 50      # Use to pad encoding-related states

# DQN HYPERPARAMETER

## Reward Shaping
SUCCESS_REWARD = 25
FAIL_REWARD = -10
STEP_REWARD = -1
ASK_REWARD = -2

## DQN Buffer
BUFFER_LIMIT = 10000
MIN_BUFFER_SIZE = 2000

## Training Constants
TRAIN_INTERVAL = 100            # Training Interval is defined as the minimum amount of experiences collected before next training
TARGET_UPDATE_INTERVAL = 3000       # The duration when we updated the target_update
PRINT_INTERVAL = 25            # Defined in unit of episodes
TRAIN_STEPS = 10
TRAIN_BATCH_SIZE = 32
GAMMA = 0.98            # Discount rate

MAX_EPSILON = 0.9
MIN_EPSILON = 0.01

# SWA Constants
SWA_START = 7500
SWA_FREQ = 100
SWA_LR = 5e-5

# Data structure to accumulate and preprocess training data before being inserted into our DQN Buffer for experience replay
class Transition:
    ASKING_ACTIONS = [1, 2, 3, 4]       # 0, 5, 6 are considered non-asking actions

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def _clone_tensor(self, t):
        return None if t is None else t.clone().detach()

    def _clone_states(self, states):
        a_t, q_t, f_t, decoder_h, ctx, seq_mask, nav_logit_mask, ask_logit_mask, b_t, cov = states

        a_t_copy = self._clone_tensor(a_t)
        q_t_copy = self._clone_tensor(q_t)
        f_t_copy = self._clone_tensor(f_t)

        # Decoder_h is a tuple of tensor, thus it needs special handling
        decoder_h_copy = None
        if decoder_h is not None:
            decoder_h_copy = (decoder_h[0].clone().detach(), decoder_h[1].clone().detach())

        ctx_copy = self._clone_tensor(ctx)

        seq_mask_copy = self._clone_tensor(seq_mask)
        nav_logit_mask_copy = self._clone_tensor(nav_logit_mask)
        ask_logit_mask_copy = self._clone_tensor(ask_logit_mask)
        b_t_copy = self._clone_tensor(b_t)
        cov_copy = self._clone_tensor(cov)

        return (a_t_copy, q_t_copy, f_t_copy, decoder_h_copy, ctx_copy, seq_mask_copy,
                nav_logit_mask_copy, ask_logit_mask_copy, b_t_copy, cov_copy)

    def add_states(self, states):
        self.states = self._clone_states(states)

    def add_next_states(self, next_states):
        self.next_states = self._clone_states(next_states)

    def add_filter(self, filter):
        self.filter = np.copy(filter)

    def add_is_done(self, is_done):
        self.is_done = np.copy(is_done)

    def add_is_success(self, is_success):
        self.is_success = np.copy(is_success)

    def add_actions(self, actions):
        self.actions = actions.clone().detach()

    def compute_reward_shaping(self):
        self.rewards = np.empty(self.batch_size)

        for i in range(self.batch_size):
            if self.filter[i]:
                continue

            self.rewards[i] = 0
            if (self.is_done[i]):
                self.rewards[i] += SUCCESS_REWARD if self.is_success[i] else FAIL_REWARD

            if self.actions[i] in Transition.ASKING_ACTIONS:
                self.rewards[i] += ASK_REWARD
            else:
                self.rewards[i] += STEP_REWARD

    def get_unfiltered_rewards(self):
        return [r for idx, r in enumerate(self.rewards) if not self.filter[idx]]

    # Return the key-th batch from states
    # States are a tuple of batched tensor, we want to return just the key-th batch
    def _get_state_by_key(self, states, key):
        temp = [None] * len(states)
        for i in range(len(states)):
            if states[i] is None:
                continue

            if type(states[i]) is tuple:         # This must be decoder_h which is as a pair of tensor
                # This tuple is the output of an LSTM model
                # Based on this LSTM model https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html,
                # the second dimension is the batch size
                assert(states[i][0].shape[1] == self.batch_size)
                assert(states[i][1].shape[1] == self.batch_size)

                # The idea is to slice out only the key-th batch
                temp[i] = (states[i][0][:, key : (key + 1)], states[i][1][:, key : (key + 1)])
                continue

            assert(states[i].shape[0] == self.batch_size)
            temp[i] = states[i][key]

        state = tuple(temp)
        return state

    def to_list(self):
        # Since all the states, rewards, actions, etc are grouped in batch, we need to divide them before inserting them to the queue
        # In addition we need to filter some of them which are invalid
        experiences = []
        for i in range(self.batch_size):
            if self.filter[i]:
                continue

            state = self._get_state_by_key(self.states, i)
            action = self.actions[i]
            reward = self.rewards[i]
            next_state = self._get_state_by_key(self.next_states, i)
            is_done = self.is_done[i]

            experiences.append((state, action, reward, next_state, is_done))
        return experiences

# Maintain past experiences for DQN experience replay
class ReplayBuffer():
    def __init__(self, device, buffer_limit=BUFFER_LIMIT):
        self.device = device
        self.buffer_limit = buffer_limit
        self.buffer = collections.deque()

    def push(self, experience):
        '''
        Input:
            * `experience`: A tuple of (state, action, reward, next_state, is_done).
                            if the buffer is full, the oldest experience will be discarded.
        Output:
            * None
        '''
        if self.__len__() == self.buffer_limit:
            self.buffer.popleft()           # Remove the oldest experience

        self.buffer.append(experience)

    def push_multiple(self, experiences):
        '''
        Input:
            * `experiences`: An array of tuple of (state, action, reward, next_state, is_done).
                            if the buffer is full, the oldest experience will be discarded.
        Output:
            * None
        '''

        for experience in experiences:
            self.push(experience)

    def _merge_states(self, array_states):
        tuple_len = len(array_states[0])
        temp = [None] * tuple_len
        for i in range(tuple_len):
            if i == 3:          # this is decoder_h that requires special handling
                # We concat based on second index since we split them based on their second index too
                first = torch.cat([e[i][0] for e in array_states], 1)
                second = torch.cat([e[i][1] for e in array_states], 1)
                temp[i] = (first, second)
                continue

            temp[i] = torch.stack([e[i] for e in array_states])

        states = tuple(temp)
        return states

    def sample(self, batch_size):
        '''
        Input:
            * `batch_size` (`int`): the size of the sample.

        Output:
            * A 5-tuple (`states`, `actions`, `rewards`, `next_states`, `dones`),
                * `states`      (`tuple` of size 10, see the details of each in the implementation)
                * `actions`     (`torch.tensor` [batch_size])
                * `rewards`     (`torch.tensor` [batch_size])
                * `next_states` (`tuple` of size 10)
                * `is_done`       (`torch.tensor` [batch_size])
              All `torch.tensor` (except `actions`) should have a datatype `torch.float` and resides in torch device `device`.
        '''
        experiences = random.sample(self.buffer, batch_size)

        states = self._merge_states([e[0] for e in experiences])
        actions = torch.tensor([e[1] for e in experiences], device=self.device)
        rewards = torch.tensor([e[2] for e in experiences], device=self.device)
        next_states = self._merge_states([e[3] for e in experiences])
        is_done = torch.tensor([1 if e[4] else 0 for e in experiences], device=self.device)

        return (states, actions, rewards, next_states, is_done)

    def __len__(self):
        '''
        Return the length of the replay buffer.
        '''
        return len(self.buffer)

# A module that keep strack of losses, reward, and success rate.
class Plotter:
    def __init__(self, hparams):
        self.exp_dir = hparams.exp_dir
        self.save_path = os.path.join(self.exp_dir, 'plot.jpg')

        # Initialize figures
        self.fig, axes = plt.subplots(3, 2, figsize=(24, 21))

        # Initialize column 1 as axes for training
        self.ax_success_rate = axes[0][0]
        self.ax_reward = axes[1][0]
        self.ax_loss = axes[2][0]

        # Initialize column 2 as axes for eval
        self.ax_eval_seen_longer_time_success_rate = axes[0][1]
        self.ax_eval_seen_success_rate = axes[1][1]
        self.ax_eval_unseen_success_rate = axes[2][1]

        self._decorate_figures()

        # Initialize train data points container
        self.episodes = []
        self.rewards = []
        self.losses = []
        self.success_rates = []

        # Initialize eval data points container
        self.eval_episodes = []
        self.eval_seen_longer_time_success_rates = []
        self.eval_seen_success_rates = []
        self.eval_unseen_success_rates = []

    def _decorate_figures(self):
        plt.rcParams['font.size'] = 18

        self.ax_reward.set_title('train reward', fontweight='bold', size=24)
        self.ax_reward.set_xlabel('episodes', fontsize=20)
        self.ax_reward.set_ylabel('average reward', fontsize=20)

        self.ax_loss.set_title('train loss', fontweight='bold', size=24)
        self.ax_loss.set_xlabel('episodes', fontsize=20)
        self.ax_loss.set_ylabel('average loss', fontsize=20)

        self.ax_success_rate.set_title('train success_rate', fontweight='bold', size=24)
        self.ax_success_rate.set_xlabel('episodes', fontsize=20)
        self.ax_success_rate.set_ylabel('success rate (%)', fontsize=20)

        self.ax_eval_seen_longer_time_success_rate.set_title('eval success_rate with max episode length (seen)', fontweight='bold', size=24)
        self.ax_eval_seen_longer_time_success_rate.set_xlabel('episodes', fontsize=20)
        self.ax_eval_seen_longer_time_success_rate.set_ylabel('success rate (%)', fontsize=20)

        self.ax_eval_seen_success_rate.set_title('eval success_rate (seen)', fontweight='bold', size=24)
        self.ax_eval_seen_success_rate.set_xlabel('episodes', fontsize=20)
        self.ax_eval_seen_success_rate.set_ylabel('success rate (%)', fontsize=20)

        self.ax_eval_unseen_success_rate.set_title('eval success_rate (unseen)', fontweight='bold', size=24)
        self.ax_eval_unseen_success_rate.set_xlabel('episodes', fontsize=20)
        self.ax_eval_unseen_success_rate.set_ylabel('success rate (%)', fontsize=20)

    def add_data_point(self, episode, reward, loss, success_rate):
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.losses.append(loss)
        self.success_rates.append(success_rate)

    # Only for seen environment first
    def add_eval_data_point(self, episode, eval_seen_longer_time_success_rate, eval_seen_success_rate, eval_unseen_success_rate):
        self.eval_episodes.append(episode)
        self.eval_seen_longer_time_success_rates.append(eval_seen_longer_time_success_rate)
        self.eval_seen_success_rates.append(eval_seen_success_rate)
        self.eval_unseen_success_rates.append(eval_unseen_success_rate)

    def save(self):
        self.ax_reward.plot(self.episodes, self.rewards)
        self.ax_loss.plot(self.episodes, self.losses)
        self.ax_success_rate.plot(self.episodes, self.success_rates)

        self.ax_eval_seen_longer_time_success_rate.plot(self.eval_episodes, self.eval_seen_longer_time_success_rates)
        self.ax_eval_seen_success_rate.plot(self.eval_episodes, self.eval_seen_success_rates)
        self.ax_eval_unseen_success_rate.plot(self.eval_episodes, self.eval_unseen_success_rates)

        self.fig.tight_layout()
        self.fig.savefig(self.save_path)

        print(f"Saved loss, reward, and success_rate graphs to {self.save_path}")

# This agent is a DQN Trainer that only trains ask_predictor
class M1Agent(VerbalAskAgent):

    def __init__(self, model, target, hparams, device, train_evaluator):
        super(M1Agent, self).__init__(model, hparams, device)

        self.total_episodes = hparams.n_iters

        # self.model is initialized in the super's constructor.
        # Over the course of training, self.model could be using swa_model or raw_model for its ask_predictor,
        # depending on which stage of the training we are
        self.swa_model = None
        self.raw_model = self.model.decoder.ask_predictor
        self.target_model = target
        self.target_model.load_state_dict(self.model.state_dict())

        # This evaluator will only be used if self.is_eval is False.
        # The evaluator is necessary for the RL to award the correct reward to the agent
        self.train_evaluator = train_evaluator
        self.buffer = ReplayBuffer(device, BUFFER_LIMIT)

        # Freeze everything except for ask_predictor
        # Implementation based on: https://discuss.pytorch.org/t/how-to-freeze-the-part-of-the-model/31409
        for name, p in self.model.named_parameters():
            p.requires_grad = "ask_predictor" in name

        self.train_interval = TRAIN_INTERVAL
        self.target_update_interval = TARGET_UPDATE_INTERVAL

        self.dqn_losses = []
        self.dqn_rewards = []
        self.dqn_successes = []

        # Initialize graph plotter to track training rewards, losses, and success rates
        self.plotter = Plotter(hparams)

        # Default settings, will be toggled in test method
        self.allow_max_episode_length = False

    def _advance_interval(self, delta_interval):
        if self.train_interval <= 0:
            self.train_interval = TRAIN_INTERVAL

        if self.target_update_interval <= 0:
            self.target_update_interval = TARGET_UPDATE_INTERVAL

        self.train_interval -= delta_interval
        self.target_update_interval -= delta_interval

    def compute_states(self, batch_size, obs, queries_unused, existing_states):
        # Unpack existing states
        a_t, q_t, decoder_h, ctx, seq_mask, cov = existing_states

        # Mask out invalid actions
        nav_logit_mask = torch.zeros(batch_size,
                                        AskAgent.n_output_nav_actions(), dtype=torch.bool, device=self.device)
        ask_logit_mask = torch.zeros(batch_size,
                                        AskAgent.n_output_ask_actions(), dtype=torch.bool, device=self.device)

        nav_mask_indices = []
        ask_mask_indices = []
        for i, ob in enumerate(obs):
            if len(ob['navigableLocations']) <= 1:
                nav_mask_indices.append((i, self.nav_actions.index('forward')))

            if queries_unused[i] <= 0:
                for question in self.question_pool:
                    ask_mask_indices.append((i, self.ask_actions.index(question)))

        nav_logit_mask[list(zip(*nav_mask_indices))] = True
        ask_logit_mask[list(zip(*ask_mask_indices))] = True

        # Image features
        f_t = self._feature_variable(obs)

        # Budget features
        b_t = torch.tensor(queries_unused, dtype=torch.long, device=self.device)

        return (a_t, q_t, f_t, decoder_h, ctx, seq_mask, nav_logit_mask, ask_logit_mask, b_t, cov)

    def _make_batch(self, obs):
        ''' Make a variable for a batch of input instructions. '''
        seq_tensor = np.array([self.env.encode(ob['instruction']) for ob in obs])

        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)

        max_length = max(seq_lengths)
        assert max_length <= ENCODE_MAX_LENGTH, f"Actual max_length: {max_length}"
        max_length = ENCODE_MAX_LENGTH      # The only modification from super._make_batch since dqn is simpler with static max length
                                            # This is easer than padding seq_mask, and cov states manually

        seq_tensor = torch.from_numpy(seq_tensor).long().to(self.device)[:, :max_length]
        seq_lengths = torch.from_numpy(seq_lengths).long().to(self.device)

        mask = (seq_tensor == padding_idx)

        return seq_tensor, mask, seq_lengths

    def _pad_ctx(self, ctx):
        # ctx.shape = (batch_size, encoding_length, 512)
        # Since encoding_length is variable, they are not suitable for DQN Training
        # Thus, we need to append them with 0 and make its demension to (batch_size, ENCODING_MAX_LENGTH, 512)
        return F.pad(input=ctx, pad=(0, 0, 0, ENCODE_MAX_LENGTH - ctx.shape[1], 0, 0), mode='constant', value=0.0)

    def _greedy_epsilon(self, distribution, epsilon):
        batch_size = distribution.size(0)
        action_size = distribution.size(1)

        actions = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for i in range(batch_size):
            if random.random() >= epsilon:
                # Exploitation: Choose the most optimal action
                actions[i] = distribution[i].argmax()
            else:
                # Exploration: Choose random action
                permutation = np.random.permutation(action_size)
                for action_choice in permutation:
                    if distribution[i, action_choice].item() == -float('inf'):      # Skip invalid mask
                        continue

                    actions[i] = int(action_choice)
                    break

        return actions

    def rollout(self, epsilon = 0.0):
        # Reset environment
        obs = self.env.reset(self.is_eval, self.allow_max_episode_length)
        batch_size = len(obs)

        # Index initial command
        seq, seq_mask, seq_lengths = self._make_batch(obs)

        # Roll-out bookkeeping
        traj = [{
            'scan': ob['scan'],
            'instr_id': ob['instr_id'],
            'goal_viewpoints': ob['goal_viewpoints'],
            'agent_path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
            'agent_ask': [],
            'teacher_ask': [],
            'teacher_ask_reason': [],
            'agent_nav': [],
            'subgoals': []
        } for ob in obs]

        # Encode initial command
        ctx, _ = self.model.encode(seq, seq_lengths)
        ctx = self._pad_ctx(ctx)

        # None is equivalent with zeros according to LSTM layer documentation. 512 is the lstm layer output size
        decoder_h = (torch.zeros(1, batch_size, 512, device=self.device),
                torch.zeros(1, batch_size, 512, device=self.device))

        if self.coverage_size is not None:
            cov = torch.zeros(seq_mask.size(0), seq_mask.size(1), self.coverage_size,
                              dtype=torch.float, device=self.device)
        else:
            cov = None

        # Initial actions
        a_t = torch.ones(batch_size, dtype=torch.long, device=self.device) * \
              self.nav_actions.index('<start>')
        q_t = torch.ones(batch_size, dtype=torch.long, device=self.device) * \
              self.ask_actions.index('<start>')

        # Whether agent decides to stop
        ended = np.array([False] * batch_size)
        # Whether the agent reaches its destination in the end
        is_success = np.array([False] * batch_size)

        self.nav_loss = 0
        self.ask_loss = 0

        # action_subgoals = [[] for _ in range(batch_size)]
        # n_subgoal_steps = [0] * batch_size

        env_action = [None] * batch_size
        queries_unused = [ob['max_queries'] for ob in obs]

        episode_len = max(ob['traj_len'] for ob in obs)

        for time_step in range(episode_len):
            transition = Transition(batch_size) if not self.is_eval else None   # Preparing training data for DQN

            if not self.is_eval:
                transition.add_filter(ended)            # Filter out episode that has already ended

            dqn_states = self.compute_states(batch_size, obs, queries_unused,
                    (a_t, q_t, decoder_h, ctx, seq_mask, cov))

            if not self.is_eval:
                transition.add_states(dqn_states)

            a_t, q_t, f_t, decoder_h, ctx, seq_mask, nav_logit_mask, ask_logit_mask, b_t, cov = dqn_states      # Unpack

            # Run first forward pass to compute ask logit
            _, _, nav_logit, nav_softmax, ask_logit, _ = self.model.decode(
                a_t, q_t, f_t, decoder_h, ctx, seq_mask, nav_logit_mask,
                ask_logit_mask, budget=b_t, cov=cov)

            self._populate_agent_state_to_obs(obs, nav_softmax, queries_unused,
                                              traj, ended, time_step)

            # Ask teacher for next ask action
            ask_target, ask_reason = self.teacher.next_ask(obs)
            ask_target = torch.tensor(ask_target, dtype=torch.long, device=self.device)
            if not self.is_eval and not (self.random_ask or self.ask_first or self.teacher_ask or self.no_ask):
                self.ask_loss += self.ask_criterion(ask_logit, ask_target)

            # Determine next ask action by sampling ask_logit with greedy_epsilon
            q_t = self._greedy_epsilon(ask_logit, epsilon)
            if self.random_ask:
                q_t = self._greedy_epsilon(ask_logit, 2.0)      # 200% randomness, which means entirely random

            # Find which agents have asked and prepend subgoals to their current instructions.
            ask_target_list = ask_target.data.tolist()
            q_t_list = q_t.data.tolist()
            has_asked = False
            verbal_subgoals = [None] * batch_size
            edit_types = [None] * batch_size
            for i in range(batch_size):
                if ask_target_list[i] != self.ask_actions.index('<ignore>'):
                    if self.ask_first:
                        q_t_list[i] = int(queries_unused[i] > 0)
                    elif self.teacher_ask:
                        q_t_list[i] = ask_target_list[i]
                    elif self.no_ask:
                        q_t_list[i] = 0

                if self.is_test:
                    self.add_to_plotter(ended[i], q_t_list[i], time_step)

                if self._should_ask(ended[i], q_t_list[i]):
                    # Query advisor for subgoal.
                    _, verbal_subgoals[i], edit_types[i] = self.advisor(obs[i], q_t_list[i])
                    # Prepend subgoal to the current instruction
                    self.env.modify_instruction(i, verbal_subgoals[i], edit_types[i])
                    # Reset subgoal step index
                    # n_subgoal_steps[i] = 0
                    # Decrement queries unused
                    queries_unused[i] -= 1
                    # Mark that some agent has asked
                    has_asked = True

            if has_asked:
                # Update observations
                obs = self.env.get_obs()
                # Make new batch with new instructions
                seq, seq_mask, seq_lengths = self._make_batch(obs)
                # Re-encode with new instructions
                ctx, _ = self.model.encode(seq, seq_lengths)
                ctx = self._pad_ctx(ctx)

                # Make new coverage vectors
                if self.coverage_size is not None:
                    cov = torch.zeros(seq_mask.size(0), seq_mask.size(1), self.coverage_size,
                                      dtype=torch.float, device=self.device)
                else:
                    cov = None

            # Run second forward pass to compute nav logit
            # NOTE: q_t and b_t changed since the first forward pass.
            q_t = torch.tensor(q_t_list, dtype=torch.long, device=self.device)
            if not self.is_eval:
                transition.add_actions(q_t)         # Keep track of the ask actions for DQN
            b_t = torch.tensor(queries_unused, dtype=torch.long, device=self.device)
            decoder_h, alpha, nav_logit, nav_softmax, cov = self.model.decode_nav(
                a_t, q_t, f_t, decoder_h, ctx, seq_mask, nav_logit_mask,
                budget=b_t, cov=cov)

            # Repopulate agent state
            # NOTE: queries_unused may have changed but it's fine since nav_teacher does not use it!
            self._populate_agent_state_to_obs(obs, nav_softmax, queries_unused,
                                              traj, ended, time_step)

            # Ask teacher for next nav action
            nav_target = self.teacher.next_nav(obs)
            nav_target = torch.tensor(nav_target, dtype=torch.long, device=self.device)
            # Nav loss
            if not self.is_eval:
                self.nav_loss += self.nav_criterion(nav_logit, nav_target)

            # Determine next nav action
            a_t = self._next_action('nav', nav_logit, nav_target, self.nav_feedback)

            # Translate agent action to environment action
            a_t_list = a_t.data.tolist()
            for i in range(batch_size):
                # Conditioned on teacher action during intervention
                # (training only or when teacher_interpret flag is on)
                # if (self.teacher_interpret or not self.is_eval) and \
                #         n_subgoal_steps[i] < len(action_subgoals[i]):
                #     a_t_list[i] = action_subgoals[i][n_subgoal_steps[i]]
                #     n_subgoal_steps[i] += 1

                env_action[i] = self.teacher.interpret_agent_action(a_t_list[i], obs[i])

            a_t = torch.tensor(a_t_list, dtype=torch.long, device=self.device)

            # Take nav action
            obs = self.env.step(env_action)

            # Save trajectory output
            ask_target_list = ask_target.data.tolist()
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['agent_path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                    traj[i]['agent_nav'].append(env_action[i])
                    traj[i]['teacher_ask'].append(ask_target_list[i])
                    traj[i]['agent_ask'].append(q_t_list[i])
                    traj[i]['teacher_ask_reason'].append(ask_reason[i])
                    traj[i]['subgoals'].append(verbal_subgoals[i])

                    if a_t_list[i] == self.nav_actions.index('<end>') or \
                            time_step >= ob['traj_len'] - 1:
                        ended[i] = True

                        if not self.is_eval:
                            # Evaluate whether we ended up in the correct spot
                            if not self.skip_dqn_train:         # Train evaluator only has train data. Therefore, need to skip if we are not actually training
                                instr_id = traj[i]['instr_id']
                                path = traj[i]['agent_path']
                                is_success[i] = self.train_evaluator.score_path(instr_id, path)
                                self.dqn_successes.append(1 if is_success[i] else 0)

                assert queries_unused[i] >= 0

            if not self.is_eval:
                transition.add_is_done(ended)               # Keep track of the episode that are ending for DQN
                transition.add_is_success(is_success)
                transition.compute_reward_shaping()
                self.dqn_rewards.extend(transition.get_unfiltered_rewards())        # Only considered rewards from episodes that have not ended
                dqn_next_states = self.compute_states(batch_size, obs, queries_unused,
                        (a_t, q_t, decoder_h, ctx, seq_mask, cov))
                transition.add_next_states(dqn_next_states)

                experiences = transition.to_list()
                self.buffer.push_multiple(experiences)
                # Uncomment this to observe the amount of experiences collected
                # print(f"Just collected {len(experiences)} at time_step {time_step}, buffer size: {len(self.buffer)}!")

                # Interval is defined in terms of unit of experiences collected
                self._advance_interval(len(experiences))

                if self.train_interval <= 0:
                    self.train_dqn()

                if self.target_update_interval <= 0:
                    self.target_model.load_state_dict(self.model.state_dict())

            # Early exit if all ended
            if ended.all():
                break

        if not self.is_eval:
            self._compute_loss()

        return traj

    # allow_max_episode_length is useful for evaluation that wants to try running for max_episode_length
    def test(self, env, feedback, use_dropout=False, allow_cheat=False, is_test=False, allow_max_episode_length=False):
        ''' Evaluate once on each instruction in the current environment '''
        self.skip_dqn_train = True          # Since torch.no_grad() would be activated
        self.allow_max_episode_length = allow_max_episode_length
        test_return =  VerbalAskAgent.test(self, env, feedback, use_dropout, allow_cheat, is_test)
        self.allow_max_episode_length = False       # Toggle this off after the testing

        return test_return

    def _compute_epsilon(self, episode):
        ''' Compute epsilon for epsilon-greedy exploration '''
        epsilon_decay = self.total_episodes / 4         # A constant for our decaying function
        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-1.0 * episode / epsilon_decay)
        return epsilon

    def train(self, env, optimizer, start_iter, end_iter, feedback):
        ''' Train for (end_iter - star_iter) number of iterations '''

        self.is_eval = False
        self.skip_dqn_train = False
        self._setup(env, feedback)
        self.model.train()
        self.optimizer = optimizer

        if self.swa_model is not None:
            self.model.decoder.ask_predictor = self.raw_model             # Ensure that we did not use swa_model for anything other than eval

        last_traj = []
        for episode in range(start_iter, end_iter):
            epsilon = self._compute_epsilon(episode)
            traj = self.rollout(epsilon)       # Train routine will be invoked by rollout method

            if end_iter - episode <= 10:
                last_traj.extend(traj)

            if (episode + 1) % PRINT_INTERVAL == 0:
                avg_reward = np.mean(self.dqn_rewards)
                avg_loss = np.mean(self.dqn_losses)
                success_rate = np.mean(self.dqn_successes)*100

                print("[Episode {}]\tavg reward : {:.3f},\tavg loss: {:.6f},\tsuccess rate: {:.2f}%,\tepsilon : {:.1f}%".format(
                        episode, avg_reward, avg_loss, success_rate, epsilon*100))

                self.plotter.add_data_point(episode, avg_reward, avg_loss, success_rate)

                # Reset losses and rewards
                self.dqn_losses = []
                self.dqn_rewards = []
                self.dqn_successes = []

            if (episode + 1) == SWA_START:
                self.swa_model = AveragedModel(self.raw_model)       # Only the ask predictor need swa_model
                self.swa_scheduler = SWALR(self.optimizer, swa_lr=SWA_LR)

            if (episode + 1) >= SWA_START and (episode + 1 - SWA_START) % SWA_FREQ == 0:
                self.swa_model.update_parameters(self.raw_model)
                self.swa_scheduler.step()

        if self.swa_model is not None:
            self.model.decoder.ask_predictor = self.swa_model             # Uses the swa_model for eval

        return last_traj

    # Return only the ask's distribution given the model and the input to the model, i.e. states
    def _get_ask_logit(self, model, states):
        a_t, q_t, f_t, decoder_h, ctx, seq_mask, nav_logit_mask, ask_logit_mask, b_t, cov = states # Unpack states
        _, _, _, _, ask_logit, _ = model.decode(
            a_t, q_t, f_t, decoder_h, ctx, seq_mask, nav_logit_mask, ask_logit_mask, budget=b_t, cov=cov)

        return ask_logit

    # Removes -inf from distribution.
    #
    # -inf is expected as part of the distribution output, but it messes with the DQN Training
    # Thus, we need to ignore all -inf values in the tensor
    #
    # distrubitions: 'tensor' of the shape of (batch_size, action_space)
    def _remove_negative_inf(self, distributions):
        distributions[distributions == -float('inf')] = 0

    def compute_loss(self, states, actions, rewards, next_states, is_done):
        estimations = self._get_ask_logit(self.model, states)
        target_estimations = torch.clone(estimations)
        target_next_estimations = self._get_ask_logit(self.target_model, next_states)

        batch_size = len(is_done)
        for i in range(batch_size):
            action = actions[i]
            reward = rewards[i]
            done = is_done[i]

            if done:
                target_estimations[i][action] = reward
            else:
                next_q = torch.max(target_next_estimations[i])
                target_estimations[i][action] = reward + GAMMA * next_q

        self._remove_negative_inf(estimations)
        self._remove_negative_inf(target_estimations)

        loss_fn = torch.nn.MSELoss()
        return loss_fn(estimations, target_estimations)

    def train_dqn(self):
        if self.skip_dqn_train:
            return

        if len(self.buffer) < MIN_BUFFER_SIZE:
            return

        losses = []
        for _ in range(TRAIN_STEPS):
            batch = self.buffer.sample(TRAIN_BATCH_SIZE)
            self.optimizer.zero_grad()
            loss = self.compute_loss(*batch)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        self.dqn_losses.append(np.mean(losses))
