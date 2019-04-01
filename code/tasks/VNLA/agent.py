import json
import os
import sys
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.distributions as D
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

from utils import padding_idx

class BaseAgent(object):

    def __init__(self):
        random.seed(1)
        self.results = {}
        self.losses = []
        self.results_path = None

    def write_results(self, traj):
        output = []
        for k, v in self.results.items():
            item = { 'instr_id' : k }
            item.update(v)
            output.append(item)

        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def rollout(self):
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, env):
        self.env.reset_epoch()

        self.results = {}
        looped = False
        traj = []
        with torch.no_grad():
            while True:
                for t in self.rollout():
                    if t['instr_id'] in self.results:
                        looped = True
                    else:
                        self.results[t['instr_id']] = {
                                'trajectory' : t['agent_path'],
                                'scan'       : t['scan'],
                                'agent_nav'  : t['agent_nav'],
                            }
                        for k in ['agent_ask', 'teacher_ask', 'teacher_ask_reason']:
                            if k in t:
                                self.results[t['instr_id']][k] = t[k]
                        if 'subgoals' in t:
                            self.results[t['instr_id']]['subgoals'] = t['subgoals']
                        traj.append(t)
                if looped:
                    break
        return traj

    def add_is_success(self, is_success):
        for k, v in is_success:
            self.results[k]['is_success'] = v


