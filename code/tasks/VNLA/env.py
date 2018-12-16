''' Batched Room-to-Room navigation environment '''

from __future__ import division
import os
import sys
import csv
import numpy as np
import math
import base64
import json
import random
import networkx as nx
from collections import defaultdict
import scipy.stats

sys.path.append('../../build')
import MatterSim

from oracle import make_oracle
from utils import load_datasets, load_nav_graphs
import utils

csv.field_size_limit(sys.maxsize)


class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, from_train_env=None, img_features=None, batch_size=100):
        if from_train_env is not None:
            self.features = from_train_env.features
            self.image_h  = from_train_env.image_h
            self.image_w  = from_train_env.image_w
            self.vfov     = from_train_env.vfov
        elif img_features is not None:
            self.image_h, self.image_w, self.vfov, self.features = \
                utils.load_img_features(img_features)
        else:
            print 'Image features not provided'
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.setNavGraphPath(
                os.path.join(os.getenv('PT_DATA_DIR', '../../../data'), 'connectivity'))
            sim.init()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0)

    def getStates(self):
        ''' Get list of states augmented with precomputed image features. rgb field will be empty. '''
        feature_states = []
        for sim in self.sims:
            state = sim.getState()
            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id][state.viewIndex,:]
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)

    def makeSimpleActions(self, simple_indices):
        ''' Take an action using a simple interface: 0-forward, 1-turn left, 2-turn right, 3-look up, 4-look down.
            All viewpoint changes are 30 degrees. Forward, look up and look down may not succeed - check state.
            WARNING - Very likely this simple interface restricts some edges in the graph. Parts of the
            environment may not longer be navigable. '''
        for i, index in enumerate(simple_indices):
            if index == 0:
                self.sims[i].makeAction(1, 0, 0)
            elif index == 1:
                self.sims[i].makeAction(0,-1, 0)
            elif index == 2:
                self.sims[i].makeAction(0, 1, 0)
            elif index == 3:
                self.sims[i].makeAction(0, 0, 1)
            elif index == 4:
                self.sims[i].makeAction(0, 0,-1)
            else:
                sys.exit("Invalid simple action");



class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, hparams, split=None, tokenizer=None, from_train_env=None,
                 traj_len_estimates=None):
        self.env = EnvBatch(
            from_train_env=from_train_env.env if from_train_env is not None else None,
            img_features=hparams.img_features, batch_size=hparams.batch_size)

        self.random = random
        self.random.seed(hparams.seed)

        self.tokenizer = tokenizer
        self.ask_for_help = hasattr(hparams, 'ask_for_help') and hparams.ask_for_help
        self.split = split
        self.batch_size = hparams.batch_size
        self.max_episode_length = hparams.max_episode_length
        self.n_subgoal_steps = hparams.n_subgoal_steps

        if self.ask_for_help:
            self.traj_len_estimates = defaultdict(list)
        else:
            self.traj_len_estimates = None

        if self.ask_for_help:
            self.query_ratio = hparams.query_ratio

        if hasattr(hparams, 'no_room') and hparams.no_room:
            self.no_room = hparams.no_room

        if self.split is not None:
            self.load_data(load_datasets([split], hparams.data_path))

        if self.ask_for_help:
            if traj_len_estimates is None:
                for k in self.traj_len_estimates:
                    self.traj_len_estimates[k] = min(self.max_episode_length,
                        float(np.average(self.traj_len_estimates[k]) +
                        1.95 * scipy.stats.sem(self.traj_len_estimates[k])))
                    assert not math.isnan(self.traj_len_estimates[k])
            else:
                """
                errors = []
                for item in self.data:
                    k = self.make_traj_estimate_key(item)
                    oracle_len = float(np.average([len(p) for p in item['trajectories']]))
                    if k in traj_len_estimates:
                        errors.append(abs(oracle_len - traj_len_estimates[k]))
                    else:
                        errors.append(abs(oracle_len - self.max_episode_length))

                print(np.average(errors), np.std(errors))
                """

                for k in self.traj_len_estimates:
                    if k in traj_len_estimates:
                        self.traj_len_estimates[k] = traj_len_estimates[k]
                    else:
                        # If (start_region, end_region) not in training set, use max_episode_length (worst case)
                        self.traj_len_estimates[k] = self.max_episode_length


    def make_traj_estimate_key(self, item):
        if hasattr(self, 'no_room') and self.no_room:
            key = (item['start_region_name'], item['object_name'])
        else:
            key = (item['start_region_name'], item['end_region_name'])
        return key

    def encode(self, instr):
        if self.tokenizer is None:
            sys.exit('No tokenizer!')
        return self.tokenizer.encode_sentence(instr)

    def load_data(self, data):
        self.data = []
        self.scans = set()
        for item in data:
            self.scans.add(item['scan'])
            if self.ask_for_help:
                key = self.make_traj_estimate_key(item)
                self.traj_len_estimates[key].extend(
                    len(t) for t in item['trajectories'])

            """
            if self.split == 'train':
                j = 0
                for instr in item['instructions']:
                    for dist, path, traj in zip(item['distances'], item['paths'], item['trajectories']):
                        new_item = dict(item)
                        del new_item['instructions']
                        new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                        new_item['instruction'] = instr
                        new_item['distances'] = [dist]
                        new_item['paths'] = [path]
                        new_item['trajectories'] = [traj]
                        self.data.append(new_item)
                        j += 1
            else:
            """
            for j,instr in enumerate(item['instructions']):
                new_item = dict(item)
                del new_item['instructions']
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instruction'] = instr
                self.data.append(new_item)

        self.reset_epoch()

        if self.split is not None:
            print 'R2RBatch loaded with %d instructions, using split: %s' % (
                len(self.data), self.split)

    def _next_minibatch(self):
        if self.ix == 0:
            self.random.shuffle(self.data)
        batch = self.data[self.ix:self.ix+self.batch_size]
        if len(batch) < self.batch_size:
            self.random.shuffle(self.data)
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.batch_size
        self.batch = batch

    def set_data_and_scans(self, data, scans):
        self.data = data
        self.scans = scans

    def reset_epoch(self):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        self.ix = 0

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'point': state.location.point,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'step' : state.step,
                'navigableLocations' : state.navigableLocations,
                'instruction' : self.instructions[i],
                'goal_viewpoints' : [path[-1] for path in item['paths']],
                'init_viewpoint' : item['paths'][0][0]
            })
            if self.ask_for_help:
                obs[-1]['max_queries'] = self.max_queries_constraints[i]
                obs[-1]['traj_len'] = self.traj_lens[i]
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
        return obs

    def _calculate_max_queries(self, traj_len):
        max_queries = self.query_ratio * traj_len / self.n_subgoal_steps
        int_max_queries = int(max_queries)
        frac_max_queries = max_queries - int_max_queries
        return int_max_queries + (self.random.random() < frac_max_queries)

    #@profile
    def reset(self, is_eval):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch()

        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['paths'][0][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.instructions = [item['instruction'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)

        if self.ask_for_help:
            self.max_queries_constraints = [None] * self.batch_size
            self.traj_lens = [None] * self.batch_size
            for i, item in enumerate(self.batch):
                if is_eval:
                    # If eval use expected length between start_region and end_region
                    key = self.make_traj_estimate_key(item)
                    traj_len_estimate = self.traj_len_estimates[key]
                else:
                    # If train use average oracle length
                    traj_len_estimate = sum(len(t)
                        for t in item['trajectories']) / len(item['trajectories'])
                self.traj_lens[i] = min(self.max_episode_length, int(round(traj_len_estimate)))
                self.max_queries_constraints[i] = self._calculate_max_queries(self.traj_lens[i])
                assert not math.isnan(self.max_queries_constraints[i])
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def prepend_instruction(self, idx, instr):
        self.instructions[idx] = instr + ' . ' + self.batch[idx]['instruction']

    def get_obs(self):
        return self._get_obs()



