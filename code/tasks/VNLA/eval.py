import json
import os
import sys
from collections import defaultdict
import networkx as nx
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

from utils import load_datasets, load_nav_graphs, load_region_label_to_name, load_panos_to_region


class Evaluation(object):

    def __init__(self, hparams, splits, data_path):
        self.success_radius = hparams.success_radius
        self.splits = splits

        self.scans = set()
        self.graphs = {}
        self.distances = {}

        self.no_room = hasattr(hparams, 'no_room') and hparams.no_room
        if splits:
            self.load_data(load_datasets(splits, data_path,
                prefix='noroom' if self.no_room else 'asknav'))

        self.region_label_to_name = load_region_label_to_name()
        self.panos_to_region = {}
        for scan in self.scans:
            self.panos_to_region[scan] = load_panos_to_region(scan, self.region_label_to_name)


    def load_data(self, data):
        self.gt = {}
        self.instr_ids = []
        scans = []
        for item in data:
            self.gt[str(item['path_id'])] = item
            if isinstance(item['path_id'], int):
                self.instr_ids.extend(['%d_%d' % (item['path_id'],i)
                    for i in range(len(item['instructions']))])
            else:
                self.instr_ids.extend(['%s_%d' % (item['path_id'],i)
                    for i in range(len(item['instructions']))])
            scans.append(item['scan'])
        self.instr_ids = set(self.instr_ids)
        scans = set(scans)

        new_scans = set.difference(scans, self.scans)
        if new_scans:
            for scan in new_scans:
                self.graphs[scan] = load_nav_graphs(scan)
                self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(self.graphs[scan]))
        self.scans.update(new_scans)

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, instr_id, path):
        gt = self.gt[instr_id[:instr_id.rfind('_')]]
        scan = gt['scan']

        self.scores['instr_id'].append(instr_id)
        self.scores['trajectory_steps'].append(len(path) - 1)

        nav_errors = oracle_errors = 1e9
        for shortest_path in gt['paths']:
            start = shortest_path[0]
            assert start == path[0][0], 'Result trajectories should include the start position'
            goal = shortest_path[-1]
            final_pos = path[-1][0]
            nearest_pos = self._get_nearest(scan, goal, path)
            nav_errors = min(nav_errors, self.distances[scan][final_pos][goal])
            oracle_errors = min(oracle_errors, self.distances[scan][nearest_pos][goal])

        self.scores['nav_errors'].append(nav_errors)
        self.scores['oracle_errors'].append(oracle_errors)
        distance = 0
        prev = path[0]
        for curr in path[1:]:
            distance += self.distances[scan][prev[0]][curr[0]]
            prev = curr
        self.scores['trajectory_lengths'].append(distance)

        if not self.no_room:
            goal_room = None
            for shortest_path in gt['paths']:
                assert goal_room is None or goal_room == \
                    self.panos_to_region[scan][shortest_path[-1]]
                goal_room = self.panos_to_region[scan][shortest_path[-1]]

            assert goal_room is not None
            final_room = self.panos_to_region[scan][path[-1][0]]
            self.scores['room_successes'].append(final_room == goal_room)

    def check_success(self, d):
        return d <= self.success_radius

    def score_path(self, instr_id, path):
        '''
            Evaluate a single agent trajectory based on how close it gets to the goal location
            Returns true if the agent stopped within a certain radius from one of the goal positions
            Returns false otherwise
        '''

        # Note: This function is just a simpler implementation of _score_item
        gt = self.gt[instr_id[:instr_id.rfind('_')]]
        scan = gt['scan']

        final_pos = path[-1][0]
        dist_to_goal = 1e9
        for shortest_path in gt['paths']:       # Find the minimum distance to all of the available goals
            goal = shortest_path[-1]
            dist_to_goal = min(dist_to_goal, self.distances[scan][final_pos][goal])

        return self.check_success(dist_to_goal)

    def score(self, output_file):
        ''' Evaluate each agent trajectory in the output_file based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids)
        with open(output_file) as f:
            for item in json.load(f):
                # Check against expected ids
                if item['instr_id'] in instr_ids:
                    instr_ids.remove(item['instr_id'])
                    self._score_item(item['instr_id'], item['trajectory'])
        assert len(instr_ids) == 0, 'Missing %d of %d instruction ids from %s - not in %s'\
                       % (len(instr_ids), len(self.instr_ids), ",".join(self.splits), output_file)
        assert len(self.scores['nav_errors']) == len(self.instr_ids)
        score_summary = {
            'nav_error': np.average(self.scores['nav_errors']),
            'oracle_error': np.average(self.scores['oracle_errors']),
            'steps': np.average(self.scores['trajectory_steps']),
            'length': np.average(self.scores['trajectory_lengths'])
        }
        is_success = [(instr_id, self.check_success(d)) for d, instr_id
            in zip(self.scores['nav_errors'], self.scores['instr_id'])]
        num_successes = len([d for d in self.scores['nav_errors'] if self.check_success(d)])
        score_summary['success_rate'] = float(num_successes)/float(len(self.scores['nav_errors']))
        oracle_successes = len([d for d in self.scores['oracle_errors'] if self.check_success(d)])
        score_summary['oracle_rate'] = float(oracle_successes)/float(len(self.scores['oracle_errors']))
        if not self.no_room:
            score_summary['room_success_rate'] = float(sum(self.scores['room_successes'])) / \
                len(self.scores['room_successes'])
        return score_summary, self.scores, is_success





