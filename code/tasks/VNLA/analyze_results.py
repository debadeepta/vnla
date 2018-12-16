from __future__ import print_function
from __future__ import division

import os
import sys
import json
import matplotlib.pyplot as plt
import math
import numpy as np

from scipy.stats import sem
from collections import defaultdict
from oracle import make_oracle
from ask_agent import AskSeq2SeqAgent

in_file = sys.argv[1]

with open(in_file) as f:
    data = json.load(f)

with open('scripts/house_ids.txt') as f:
    house_ids = [l.rstrip() for l in f.readlines()]

oracle = make_oracle('shortest', AskSeq2SeqAgent.nav_actions)
oracle.add_scans(set(house_ids))


set_name = None
if 'test_unseen' in in_file:
    set_name = 'test_unseen'

if 'test_seen' in in_file:
    set_name = 'test_seen'

oracle_distances = {}
oracle_nodes = {}
oracle_steps = {}
with open('../../../data/intent_based_commands_v3/R2R_' + set_name + '.json') as f:
    tmp = json.load(f)
    for item in tmp:
        l = []
        for path in item['paths']:
            l.append(len(path))
        oracle_nodes[item['path_id']] = sum(l) / len(l)
        oracle_distances[item['path_id']] = sum(item['distances']) / len(item['distances'])
        l = []
        for t in item['trajectories']:
            l.append(len(t))
        oracle_steps[item['path_id']] = sum(l) / len(l)

def bin_and_average(plot_data, num_bins=10):
    print()
    plot_data = sorted(plot_data, key=lambda x: x[1])

    bin_size = len(plot_data) // num_bins

    print(bin_size)

    bins = [{ 'x' : [], 'y' : [] } for _ in range(1000)]

    idx = 0
    while idx < len(plot_data):
        bin_id = idx // bin_size
        tmp = plot_data[idx : idx + bin_size]
        y, x = zip(*tmp)
        bins[bin_id]['x'] = x
        bins[bin_id]['y'] = y

        idx += bin_size

    for i, b in enumerate(bins):
        if b['x']:
            print('%.2f' % np.average(b['x']), '%.2f' % np.average(b['y']), sep='\t')

def unique_views(traj):
    res = [traj[0]]
    for i in range(1, len(traj)):
        if traj[i] != traj[i - 1]:
            res.append(traj[i])
    return res


def make_bins(plot_data, levels):
    plot_data = sorted(plot_data, key=lambda x: x[1])

    bins = defaultdict(list)
    for item in plot_data:
        for l in levels:
            found = False
            if item[1] < l:
                found = True
                bins[l].append(item[0])
                break
        if not found:
            bins[levels[-1]].append(item[0])

    bins = sorted(list(bins.iteritems()))
    for x, y in bins:
        print('%d %d %.2f %.2f' % (x, len(y), np.average(y) * 100 / 100., 1.96 * sem(y)))

### SUCCESS RATE BY ORACLE LENGTH ###

plot_data = []

tmp = []
for item in data:
    #length = oracle.distances[item['scan']][item['oracle_path'][0]][item['oracle_path'][-1]]
    idx = int(item['instr_id'].split('_')[0])
    length = oracle_steps[idx]
    plot_data.append((int(item['is_success']) * 100, length))
    tmp.append(length)

print(max(tmp), min(tmp))

make_bins(plot_data, (0, 5, 10, 15, 20, 25))

### SUCCESS RATE BY AGENT STEPS ###

plot_data = []

tmp = []
for item in data:
    length = len(item['agent_nav'])
    plot_data.append((int(item['is_success']) * 100, length))
    tmp.append(length)

print(max(tmp), min(tmp))

make_bins(plot_data, (0, 5, 10, 15, 20, 25))

