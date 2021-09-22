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

import torch
import torch.nn as nn
import torch.distributions as D
from torch import optim
import torch.nn.functional as F

from utils import padding_idx
from agent import BaseAgent
from oracle import make_oracle
from ask_agent import AskAgent


class VerbalAskAgent(AskAgent):

    def __init__(self, model, hparams, device):
        super(VerbalAskAgent, self).__init__(model, hparams, device,
                                             should_make_advisor=False)

        assert 'verbal' in hparams.advisor
        if 'easy' in hparams.advisor:
            mode = 'easy'
        elif 'hard' in hparams.advisor:
            mode = 'hard'
        elif 'qa' in hparams.advisor:
            mode = 'qa'
        else:
            sys.exit('unknown advisor: %s' % hparams.advisor)

        self.advisor = make_oracle('verbal', hparams.n_subgoal_steps,
                                   self.nav_actions, self.ask_actions, mode=mode)
        self.hparams = hparams
        self.teacher_interpret = hasattr(hparams, 'teacher_interpret') and hparams.teacher_interpret

    def rollout(self):
        # Reset environment
        obs = self.env.reset(self.is_eval)
        batch_size = len(obs)

        # Sample random ask positions
        if self.random_ask:
            random_ask_positions = [None] * batch_size
            for i, ob in enumerate(obs):
                random_ask_positions[i] = self.random.sample(
                    range(ob['traj_len']), ob['max_queries'])

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
        decoder_h = None

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

        self.nav_loss = 0
        self.ask_loss = 0

        # action_subgoals = [[] for _ in range(batch_size)]
        # n_subgoal_steps = [0] * batch_size

        env_action = [None] * batch_size
        queries_unused = [ob['max_queries'] for ob in obs]

        episode_len = max(ob['traj_len'] for ob in obs)

        for time_step in range(episode_len):

            # Mask out invalid actions
            nav_logit_mask = torch.zeros(batch_size,
                                         AskAgent.n_output_nav_actions(), dtype=torch.uint8, device=self.device)
            ask_logit_mask = torch.zeros(batch_size,
                                         AskAgent.n_output_ask_actions(), dtype=torch.uint8, device=self.device)

            nav_mask_indices = []
            ask_mask_indices = []
            for i, ob in enumerate(obs):
                if len(ob['navigableLocations']) <= 1:
                    nav_mask_indices.append((i, self.nav_actions.index('forward')))

                if queries_unused[i] <= 0:
                    for question in self.question_pool:
                        ask_mask_indices.append((i, self.ask_actions.index(question)))

            nav_logit_mask[list(zip(*nav_mask_indices))] = 1
            ask_logit_mask[list(zip(*ask_mask_indices))] = 1

            # Image features
            f_t = self._feature_variable(obs)

            # Budget features
            b_t = torch.tensor(queries_unused, dtype=torch.long, device=self.device)

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

            # Determine next ask action
            q_t = self._next_action('ask', ask_logit, ask_target, self.ask_feedback)

            # Find which agents have asked and prepend subgoals to their current instructions.
            ask_target_list = ask_target.data.tolist()
            q_t_list = q_t.data.tolist()
            has_asked = False
            verbal_subgoals = [None] * batch_size
            edit_types = [None] * batch_size
            for i in range(batch_size):
                if ask_target_list[i] != self.ask_actions.index('<ignore>'):
                    if self.random_ask:
                        q_t_list[i] = time_step in random_ask_positions[i]
                    elif self.ask_first:
                        q_t_list[i] = int(queries_unused[i] > 0)
                    elif self.teacher_ask:
                        q_t_list[i] = ask_target_list[i]
                    elif self.no_ask:
                        q_t_list[i] = 0

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
                # Make new coverage vectors
                if self.coverage_size is not None:
                    cov = torch.zeros(seq_mask.size(0), seq_mask.size(1), self.coverage_size,
                                      dtype=torch.float, device=self.device)
                else:
                    cov = None

            # Run second forward pass to compute nav logit
            # NOTE: q_t and b_t changed since the first forward pass.
            q_t = torch.tensor(q_t_list, dtype=torch.long, device=self.device)
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

                assert queries_unused[i] >= 0

            # Early exit if all ended
            if ended.all():
                break

        if not self.is_eval:
            self._compute_loss()

        return traj
