# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import math
import networkx as nx
import functools
import scipy.stats
import random
import sys
import copy
import numpy as np

import torch

import utils
sys.path.append('../../build')
import MatterSim

class ShortestPathOracle(object):
    ''' Shortest navigation teacher '''

    def __init__(self, agent_nav_actions, env_nav_actions=None):
        self.scans = set()
        self.graph = {}
        self.paths = {}
        self.distances = {}
        self.agent_nav_actions = agent_nav_actions

        if env_nav_actions is not None:
            self.env_nav_actions = env_nav_actions

    def add_scans(self, scans, path=None):
        new_scans = set.difference(scans, self.scans)
        if new_scans:
            print('Loading navigation graphs for %d scans' % len(new_scans))
            for scan in new_scans:
                graph, paths, distances = self._compute_shortest_paths(scan, path=path)
                self.graph[scan] = graph
                self.paths[scan] = paths
                self.distances[scan] = distances
            self.scans.update(new_scans)

    def _compute_shortest_paths(self, scan, path=None):
        ''' Load connectivity graph for each scan, useful for reasoning about shortest paths '''
        graph = utils.load_nav_graphs(scan, path=path)
        paths = dict(nx.all_pairs_dijkstra_path(graph))
        distances = dict(nx.all_pairs_dijkstra_path_length(graph))
        return graph, paths, distances

    def _find_nearest_point(self, scan, start_point, end_points):
        best_d = 1e9
        best_point = None

        for end_point in end_points:
            d = self.distances[scan][start_point][end_point]
            if d < best_d:
                best_d = d
                best_point = end_point
        return best_d, best_point

    def _find_nearest_point_on_a_path(self, scan, current_point, start_point, goal_point):
        path = self.paths[scan][start_point][goal_point]
        return self._find_nearest_point(scan, current_point, path)

    def _shortest_path_action(self, ob):
        ''' Determine next action on the shortest path to goals. '''

        scan = ob['scan']
        start_point = ob['viewpoint']

        # Find nearest goal
        _, goal_point = self._find_nearest_point(scan, start_point, ob['goal_viewpoints'])

        # Stop if a goal is reached
        if start_point == goal_point:
            return (0, 0, 0)

        path = self.paths[scan][start_point][goal_point]
        next_point = path[1]

        # Can we see the next viewpoint?
        for i, loc in enumerate(ob['navigableLocations']):
            if loc.viewpointId == next_point:
                # Look directly at the viewpoint before moving
                if loc.rel_heading > math.pi/6.0:
                      return (0, 1, 0) # Turn right
                elif loc.rel_heading < -math.pi/6.0:
                      return (0,-1, 0) # Turn left
                elif loc.rel_elevation > math.pi/6.0 and ob['viewIndex'] // 12 < 2:
                      return (0, 0, 1) # Look up
                elif loc.rel_elevation < -math.pi/6.0 and ob['viewIndex'] // 12 > 0:
                      return (0, 0,-1) # Look down
                else:
                      return (i, 0, 0) # Move

        # Can't see it - first neutralize camera elevation
        if ob['viewIndex'] // 12 == 0:
            return (0, 0, 1) # Look up
        elif ob['viewIndex'] // 12 == 2:
            return (0, 0,-1) # Look down

        # Otherwise decide which way to turn
        target_rel = self.graph[ob['scan']].nodes[next_point]['position'] - ob['point']
        target_heading = math.pi / 2.0 - math.atan2(target_rel[1], target_rel[0])
        if target_heading < 0:
            target_heading += 2.0 * math.pi
        if ob['heading'] > target_heading and ob['heading'] - target_heading < math.pi:
            return (0, -1, 0) # Turn left
        if target_heading > ob['heading'] and target_heading - ob['heading'] > math.pi:
            return (0, -1, 0) # Turn left

        return (0, 1, 0) # Turn right

    def _map_env_action_to_agent_action(self, action, ob):
        ix, heading_chg, elevation_chg = action
        if heading_chg > 0:
            return self.agent_nav_actions.index('right')
        if heading_chg < 0:
            return self.agent_nav_actions.index('left')
        if elevation_chg > 0:
            return self.agent_nav_actions.index('up')
        if elevation_chg < 0:
            return self.agent_nav_actions.index('down')
        if ix > 0:
            return self.agent_nav_actions.index('forward')
        if ob['ended']:
            return self.agent_nav_actions.index('<ignore>')
        return self.agent_nav_actions.index('<end>')

    def interpret_agent_action(self, action_idx, ob):

        # If the action is not `forward`, simply map it to the simulator's
        # action space
        if action_idx != self.agent_nav_actions.index('forward'):
            return self.env_nav_actions[action_idx]

        scan = ob['scan']
        start_point = ob['viewpoint']

        # Find nearest goal view point
        _, goal_point = self._find_nearest_point(scan, start_point, ob['goal_viewpoints'])

        optimal_path = self.paths[scan][start_point][goal_point]

        # If it is at the goal, take action 1.
        # The dataset guarantees that the goal is always reachable.
        if len(optimal_path) < 2:
            return (1, 0, 0)

        next_optimal_point = optimal_path[1]

        # If the next optimal viewpoint is within 30 degrees of the center of
        # the view, go to it.
        for i, loc in enumerate(ob['navigableLocations']):
            if loc.viewpointId == next_optimal_point:
                if loc.rel_heading > math.pi/6.0 or loc.rel_heading < -math.pi/6.0 or \
                   (loc.rel_elevation > math.pi/6.0  and ob['viewIndex'] // 12 < 2) or \
                   (loc.rel_elevation < -math.pi/6.0 and ob['viewIndex'] // 12 > 0):
                    continue
                else:
                    return (i, 0, 0)

        # Otherwise, take action 1.
        return (1, 0, 0)

    def __call__(self, obs):
        self.actions = list(map(self._shortest_path_action, obs))
        return list(map(self._map_env_action_to_agent_action, self.actions, obs))



class AskOracle(object):  #### Change the asking teacher here ####

    def __init__(self, hparams, agent_ask_actions):
        self.deviate_threshold = hparams.deviate_threshold
        self.uncertain_threshold = hparams.uncertain_threshold
        self.unmoved_threshold = hparams.unmoved_threshold
        self.same_room_threshold = hparams.same_room_threshold
        self.agent_ask_actions = agent_ask_actions

        self.rule_a_e = hasattr(hparams, 'rule_a_e') and hparams.rule_a_e
        self.rule_b_d = hasattr(hparams, 'rule_b_d') and hparams.rule_b_d

    def _should_ask_rule_a_e(self, ob, nav_oracle=None):

        if ob['queries_unused'] <= 0:
            return self.agent_ask_actions.index('dont_ask'), 'exceed'

        scan = ob['scan']
        current_point = ob['viewpoint']
        _, goal_point = nav_oracle._find_nearest_point(scan, current_point, ob['goal_viewpoints'])

        agent_decision = int(np.argmax(ob['nav_dist']))
        if current_point == goal_point and \
           agent_decision == nav_oracle.agent_nav_actions.index('forward'):
            return self.agent_ask_actions.index('arrive'), 'arrive'

        start_point = ob['init_viewpoint']
        d, _ = nav_oracle._find_nearest_point_on_a_path(scan, current_point, start_point, goal_point)
        if d > self.deviate_threshold:
            return self.agent_ask_actions.index('direction'), 'deviate'

        return self.agent_ask_actions.index('dont_ask'), 'pass'

    def _should_ask_rule_b_d(self, ob, nav_oracle=None):

        if ob['queries_unused'] <= 0:
            return self.agent_ask_actions.index('dont_ask'), 'exceed'

        agent_dist = ob['nav_dist']
        uniform = [1. / len(agent_dist)] * len(agent_dist)
        entropy_gap = scipy.stats.entropy(uniform) - scipy.stats.entropy(agent_dist)
        if entropy_gap < self.uncertain_threshold - 1e-9:
            return self.agent_ask_actions.index('instruction'), 'uncertain'

        if len(ob['agent_path']) >= self.unmoved_threshold:
            last_nodes = [t[0] for t in ob['agent_path']][-self.unmoved_threshold:]
            if all(node == last_nodes[0] for node in last_nodes):
                return self.agent_ask_actions.index('distance'), 'unmoved'

        if ob['queries_unused'] >= ob['traj_len'] - ob['time_step']:
            return self.agent_ask_actions.index('instruction'), 'why_not'

        return self.agent_ask_actions.index('dont_ask'), 'pass'

    def _should_ask(self, ob, nav_oracle=None):

        if self.rule_a_e:
            return self._should_ask_rule_a_e(ob, nav_oracle=nav_oracle)

        if self.rule_b_d:
            return self._should_ask_rule_b_d(ob, nav_oracle=nav_oracle)

        if ob['queries_unused'] <= 0:
            return self.agent_ask_actions.index('dont_ask'), 'exceed'

        # Find nearest point on the current shortest path
        scan = ob['scan']
        current_point = ob['viewpoint']
        # Find nearest goal to current point
        d, goal_point = nav_oracle._find_nearest_point(scan, current_point, ob['goal_viewpoints'])

        panos_to_region = utils.load_panos_to_region(scan, None, include_region_id=False)

        # Rule (e): ask if the goal has been reached
        agent_decision = int(np.argmax(ob['nav_dist']))
        if current_point == goal_point or agent_decision == nav_oracle.agent_nav_actions.index('<end>'):
            return self.agent_ask_actions.index('arrive'), 'arrive'

        # Rule (a): ask if the agent deviates too far from the optimal path
        if d > self.deviate_threshold:
            return self.agent_ask_actions.index('direction'), 'deviate'

        # Rule (c): ask if not moving for too long
        if len(ob['agent_path']) >= self.unmoved_threshold:
            last_nodes = [t[0] for t in ob['agent_path']][-self.unmoved_threshold:]
            if all(node == last_nodes[0] for node in last_nodes):
                return self.agent_ask_actions.index('distance'), 'unmoved'

        # Rule (f): ask if staying in the same room for too long
        if len(ob['agent_path']) >= self.same_room_threshold:
            last_ask = [a for a in ob['agent_ask']][-self.same_room_threshold:]
            last_nodes = [t[0] for t in ob['agent_path']][-self.same_room_threshold:]
            if all(panos_to_region[node] == panos_to_region[last_nodes[0]] for node in last_nodes) and \
               self.agent_ask_actions.index('room') not in last_ask:
                return self.agent_ask_actions.index('room'), 'same_room'

        # Rule (b): ask if uncertain
        agent_dist = ob['nav_dist']
        uniform = [1. / len(agent_dist)] * len(agent_dist)
        entropy_gap = scipy.stats.entropy(uniform) - scipy.stats.entropy(agent_dist)
        if entropy_gap < self.uncertain_threshold - 1e-9:
            return self.agent_ask_actions.index('direction'), 'uncertain'

        return self.agent_ask_actions.index('dont_ask'), 'pass'

    def _add_ignore(self, action, ob):
        if ob['ended']:
            return self.agent_ask_actions.index('<ignore>')
        else:
            return action

    def __call__(self, obs, nav_oracle):
        should_ask_fn = functools.partial(self._should_ask, nav_oracle=nav_oracle)
        actions, reasons = zip(*list(map(should_ask_fn, obs)))
        actions = list(map(self._add_ignore, actions, obs))
        return actions, reasons


class MultistepShortestPathOracle(ShortestPathOracle):

    def __init__(self, n_steps, agent_nav_actions):
        super(MultistepShortestPathOracle, self).__init__(agent_nav_actions)
        self.sim = MatterSim.Simulator()
        self.sim.setRenderingEnabled(False)
        self.sim.setDiscretizedViewingAngles(True)
        self.sim.setCameraResolution(640, 480)
        self.sim.setCameraVFOV(math.radians(60))
        self.sim.setNavGraphPath(
            os.path.join(os.getenv('PT_DATA_DIR', '../../../data'), 'connectivity'))
        self.sim.init()
        self.n_steps = n_steps

    def _shortest_path_actions(self, ob):
        actions = []
        self.sim.newEpisode(ob['scan'], ob['viewpoint'], ob['heading'], ob['elevation'])

        assert not ob['ended']

        for _ in range(self.n_steps):
            # Query oracle for next action
            action = self._shortest_path_action(ob)
            # Convert to agent action
            agent_action = self._map_env_action_to_agent_action(action, ob)
            actions.append(agent_action)
            # Take action
            self.sim.makeAction(*action)

            if action == (0, 0, 0):
                break

            state = self.sim.getState()
            ob = {
                    'viewpoint': state.location.viewpointId,
                    'viewIndex': state.viewIndex,
                    'heading'  : state.heading,
                    'elevation': state.elevation,
                    'navigableLocations': state.navigableLocations,
                    'point'    : state.location.point,
                    'ended'    : ob['ended'] or action == (0, 0, 0),
                    'goal_viewpoints': ob['goal_viewpoints'],
                    'scan'     : ob['scan']
                }

        return actions

    def __call__(self, ob):
        return self._shortest_path_actions(ob)


class NextOptimalOracle(object):

    def __init__(self, hparams, agent_nav_actions, env_nav_actions,
                 agent_ask_actions):
        self.type = 'next_optimal'
        self.ask_oracle = make_oracle('ask', hparams, agent_ask_actions)
        self.nav_oracle = make_oracle('shortest', agent_nav_actions, env_nav_actions)

    def __call__(self, obs):
        ask_actions, ask_reasons = self.ask_oracle(obs, self.nav_oracle)

        self.nav_oracle.add_scans(set(ob['scan'] for ob in obs))
        nav_actions = self.nav_oracle(obs)

        return nav_actions, ask_actions, ask_reasons

    def add_scans(self, scans):
        self.nav_oracle.add_scans(scans)

    def next_ask(self, obs):
        return self.ask_oracle(obs, self.nav_oracle)

    def next_nav(self, obs):
        return self.nav_oracle(obs)

    def interpret_agent_action(self, *args, **kwargs):
        return self.nav_oracle.interpret_agent_action(*args, **kwargs)


class StepByStepSubgoalOracle(object):

    def __init__(self, n_steps, agent_nav_actions, agent_ask_actions, mode=None):
        self.type = 'step_by_step'
        self.nav_oracle = make_oracle('direct', n_steps, agent_nav_actions)
        self.agent_nav_actions = agent_nav_actions
        self.agent_ask_actions = agent_ask_actions
        self.mode = mode
        if mode not in ['easy', 'hard', 'qa']:
            sys.exit('unknown step by step mode!')

    def add_scans(self, scans):
        self.nav_oracle.add_scans(scans)

    def _make_action_name(self, a):
        action_name = self.agent_nav_actions[a]
        if action_name in ['up', 'down']:
            return 'look ' + action_name
        elif action_name in ['left', 'right']:
            return 'turn ' + action_name
        elif action_name == 'forward':
            return 'go ' + action_name
        elif action_name == '<end>':
            return 'stop'
        elif action_name == '<ignore>':
            return ''
        return None

    def _answer_question(self, actions, ob, q):  #### Change the agent's interpretation of the answers here ####
        scan = ob['scan']
        instr = ob['instruction']
        current_viewpoint = ob['viewpoint']
        start_viewpoint = ob['init_viewpoint']
        goal_viewpoints = ob['goal_viewpoints']

        panos_to_region = utils.load_panos_to_region(scan, None, include_region_id=True)
        current_region_id, current_region = panos_to_region[current_viewpoint]
        goal_region_ids = []
        for viewpoint in goal_viewpoints:
            id, region = panos_to_region[viewpoint]
            goal_region_ids.append(id)
            goal_region = region

        d, goal_point = self.nav_oracle._find_nearest_point(scan, current_viewpoint, ob['goal_viewpoints'])

        actions_names = [self._make_action_name(action) for action in actions]

        # answer for 'do I arrive?'
        if self.agent_ask_actions[q] == 'arrive':
            if current_viewpoint in goal_viewpoints:
                return 'stop .', 'replace'
            else:
                return 'go, ', 'prepend'

        # answer for 'am I in the right room?'
        if self.agent_ask_actions[q] == 'room':
            if current_region == goal_region and current_region_id in goal_region_ids:
                if ('find' in instr) and (' in ' in instr):
                    return instr[instr.index('find'):instr.index(' in ')], 'replace'
                else:
                    return instr, 'replace'
            else:
                return 'exit room, ', 'prepend'

        # answer for 'am I on the right direction?'
        elif self.agent_ask_actions[q] == 'direction':
            if 'turn' in actions_names[0]:
                return 'turn around, ', 'prepend'
            else:
                return 'go straight, ', 'prepend'

        # answer for 'is the goal far from me?'
        elif self.agent_ask_actions[q] == 'distance':
            if d >= 10:
                return 'far, ', 'prepend'
            elif d >= 5:
                return 'middle, ', 'prepend'
            else:
                return 'close, ', 'prepend'
    # TBD

    def _map_actions_to_instruction_hard(self, actions):
        agg_actions = []
        cnt = 1
        for i in range(1, len(actions)):
            if actions[i] != actions[i - 1]:
                agg_actions.append((actions[i - 1], cnt))
                cnt = 1
            else:
                cnt += 1
        agg_actions.append((actions[-1], cnt))
        instruction = []
        for a, c in agg_actions:
            action_name = self._make_action_name(a)
            if c > 1:
                if 'turn' in action_name:
                    degree = 30 * c
                    if 'left' in action_name:
                        instruction.append('turn %d degrees left' % degree)
                    elif 'right' in action_name:
                        instruction.append('turn %d degrees right' % degree)
                    else:
                        raise(ValueError, action_name)
                elif 'go' in action_name:
                    instruction.append('%s %d steps' % (action_name, c))
            elif action_name != '':
                instruction.append(action_name)
        return ' , '.join(instruction), 'prepend'

    def _map_actions_to_instruction_easy(self, actions):
        instruction = []
        for a in actions:
            instruction.append(self._make_action_name(a))
        return ' , '.join(instruction), 'prepend'

    def __call__(self, ob, q=None):
        action_seq = self.nav_oracle(ob)
        if self.mode == 'qa':
            assert(q is not None)
            verbal_instruction, edit_type = self._answer_question(action_seq, ob, q)
        elif self.mode == 'easy':
            verbal_instruction, edit_type = self._map_actions_to_instruction_easy(action_seq)
        elif self.mode == 'hard':
            verbal_instruction, edit_type = self._map_actions_to_instruction_hard(action_seq)
        return action_seq, verbal_instruction, edit_type


def make_oracle(oracle_type, *args, **kwargs):
    if oracle_type == 'shortest':
        # returns the next optimal agent action, in batch
        return ShortestPathOracle(*args, **kwargs)
    if oracle_type == 'next_optimal':
        # returns (the next optimal agent action, the next optimal ask action, ask reason), in batch
        return NextOptimalOracle(*args, **kwargs)
    if oracle_type == 'ask':
        # returns (the next optimal ask action, ask reason), in batch
        return AskOracle(*args, **kwargs)
    if oracle_type == 'direct':
        # returns the next n_step of optimal agent actions, NOT in batch
        return MultistepShortestPathOracle(*args, **kwargs)
    if oracle_type == 'verbal':
        # returns (the next n_step of optimal agent actions, verbal instruction in string), NOT in batch
        return StepByStepSubgoalOracle(*args, **kwargs)

    return None



