import os
import sys
import json
import random
import math
from termcolor import colored
from argparse import Namespace
import scipy.stats

random.seed(213)

import torch

from oracle import *
from ask_agent import AskAgent
sys.path.append('../../build')
import MatterSim

sim = MatterSim.Simulator()
sim.setRenderingEnabled(False)
sim.setDiscretizedViewingAngles(True)
sim.setCameraResolution(640, 480)
sim.setCameraVFOV(math.radians(60))
sim.setNavGraphPath(
    os.path.join(os.getenv('PT_DATA_DIR', '../../../data'), 'connectivity'))
sim.init()


scans = []
with open('scripts/house_ids.txt') as f:
    for line in f:
        scans.append(line.rstrip())

with open('../../../data/intent_based_commands/R2R_val_unseen.json') as f:
    data = json.load(f)

sample = random.sample(data, 100)

print(scans)

# Shortest path oracle

def test_shortest_oracle():

    oracle = make_oracle('shortest', AskAgent.nav_actions)
    oracle.add_scans(set(scans))

    for s in sample:
        sim.newEpisode(s['scan'], s['paths'][0][0], s['heading'], 0)
        actions = s['trajectories'][0]
        ob = { 'ended' : False }
        for i, a in enumerate(actions):
            state = sim.getState()
            ob.update({
                'viewpoint': state.location.viewpointId,
                'viewIndex': state.viewIndex,
                'heading'  : state.heading,
                'elevation': state.elevation,
                'navigableLocations': state.navigableLocations,
                'point'    : state.location.point,
                'goal_viewpoints': [s['paths'][0][-1]],
                'scan'     : s['scan']
            })

            oracle_a = oracle([ob])[0]
            ground_a = oracle._map_env_action_to_agent_action(a, ob)

            assert oracle_a == ground_a, \
                colored('FAIL SHORTEST ORACLE TEST', 'red') + ' ' + str(oracle_a) + ' ' + str(ground_a)

            if a == AskAgent.nav_actions.index('<end>'):
                ob['ended'] = True

            sim.makeAction(*tuple(a))

    print(colored('PASS SHORTEST ORACLE TEST', 'green'))

# Multi-step shortest path oracle

def test_multistep_shortest_oracle(n_steps):

    oracle = MultistepShortestPathOracle(n_steps, AskAgent.nav_actions, AskAgent.env_actions)
    oracle.add_scans(set(scans))

    for s in sample:
        sim.newEpisode(s['scan'], s['paths'][0][0], s['heading'], 0)
        actions = s['trajectories'][0]
        ob = { 'ended' : False }

        idx = 0
        for i, a in enumerate(actions):
            state = sim.getState()
            ob.update({
                'viewpoint': state.location.viewpointId,
                'viewIndex': state.viewIndex,
                'heading'  : state.heading,
                'elevation': state.elevation,
                'navigableLocations': state.navigableLocations,
                'point'    : state.location.point,
                'goal_viewpoints': [s['paths'][0][-1]],
                'scan'     : s['scan']
            })

            if i == 0 or idx >= len(oracle_as):
                oracle_as = oracle(ob)
                idx = 0

            oracle_a = oracle_as[idx]
            idx += 1

            ground_a = oracle._map_env_action_to_agent_action(a, ob)

            assert oracle_a == ground_a, \
                colored('FAIL MULISTEP SHORTEST ORACLE TEST', 'red') + ' ' + str(oracle_a) + ' ' + str(ground_a)

            if a == AskAgent.nav_actions.index('<end>'):
                ob['ended'] = True

            sim.makeAction(*tuple(a))


    print(colored('PASS MULTISTEP SHORTEST ORACLE TEST %d steps' % n_steps, 'green'))


# TEST ASK ORACLE

def test_ask_oracle():
    print 'asdfafasda'
    nav_oracle = ShortestPathOracle(AskAgent.nav_actions)
    nav_oracle.add_scans(set(scans))

    hparams = Namespace()
    hparams.deviate_threshold = 1.0
    hparams.uncertain_threshold = 1.0
    hparams.unmoved_threshold = 3
    hparams.max_queries = 3

    ask_oracle = AskOracle(hparams, AskAgent.ask_actions)

    ob = { 'query_count' : 3 }
    a, r = ask_oracle._should_ask(ob, nav_oracle)
    assert a == AskOracle.DONT_ASK and r == 'exceed', str(a) + ' ' + r

    for s in sample:
        ob = {
                    'query_count' : 2,
                    'scan' : s['scan'],
                    'init_viewpoint' : s['paths'][0][0],
                    'goal_viewpoints' : [s['paths'][0][-1]]
            }
        for viewpoint in s['paths'][0]:
            ob['viewpoint'] = viewpoint
            d, _ = nav_oracle._find_nearest_point_on_a_path(ob['scan'], ob['viewpoint'], ob['goal_viewpoints'][0])
            assert d < 1e-9

        if len(s['paths'][0]) < 4:
            continue

        ob['goal_viewpoints'] = [s['paths'][0][len(s['paths'][0]) // 2]]
        for i in range(len(s['paths'][0]) // 2, len(s['paths'][0])):
            ob['viewpoint'] = s['paths'][0][i]
            d, _ = nav_oracle._find_nearest_point_on_a_path(ob['scan'], ob['viewpoint'], ob['goal_viewpoints'][0])
            if d > hparams.deviate_threshold:
                a, r = ask_oracle._should_ask(ob, nav_oracle)
                assert a == AskOracle.ASK and r == 'deviate', str(a) + ' ' + r

        ob['viewpoint'] = s['paths'][0][0]
        ob['nav_dist'] = [0.2, 0.4, 0.2, 0.1, 0.1]
        a, r = ask_oracle._should_ask(ob, nav_oracle)
        assert a == AskOracle.ASK and r == 'uncertain', str(a) + ' ' + r
        ob['nav_dist'] = [0.8, 0.1, 0.0, 0.0, 0.0]
        try:
            a, r = ask_oracle._should_ask(ob, nav_oracle)
        except KeyError:
            pass

        ob['agent_path'] = ['a', 'a', 'a']
        a, r = ask_oracle._should_ask(ob, nav_oracle)
        assert a == AskOracle.ASK and r == 'unmoved', str(a) + ' ' + r
        ob['agent_path'] = ['a', 'a']
        a, r = ask_oracle._should_ask(ob, nav_oracle)
        assert a == AskOracle.DONT_ASK and r == 'pass', str(a) + ' ' + r
        ob['agent_path'] = ['a', 'a', 'b', 'b']
        a, r = ask_oracle._should_ask(ob, nav_oracle)
        assert a == AskOracle.DONT_ASK and r == 'pass', str(a) + ' ' + r

    print(colored('PASS ASK ORACLE TEST', 'green'))


# Test step by step oracle

def test_step_oracle():

    hparams = Namespace()
    hparams.n_subgoal_steps = 2
    oracle = StepByStepSubgoalOracle(
        hparams, AskAgent.nav_actions, AskAgent.env_actions, mode='hard')

    nav_actions = AskAgent.nav_actions

    actions = [nav_actions.index('up'), nav_actions.index('right'), nav_actions.index('right')]
    instr = oracle._map_actions_to_instruction(actions)
    assert instr == 'look up , turn 60 degrees right', instr

    actions = [nav_actions.index('right'), nav_actions.index('forward'), nav_actions.index('<end>')]
    instr = oracle._map_actions_to_instruction(actions)
    assert instr == 'turn right , go forward , stop', instr

    actions = [nav_actions.index('forward'), nav_actions.index('forward'), nav_actions.index('forward')]
    instr = oracle._map_actions_to_instruction(actions)
    assert instr == 'go forward 3 steps', instr


    actions = [nav_actions.index('right'), nav_actions.index('forward'), nav_actions.index('<ignore>')]
    instr = oracle._map_actions_to_instruction(actions)
    assert instr == 'turn right , go forward', instr

    print(colored('PASS STEP BY STEP ORACLE TEST', 'green'))


def test_region_oracle():

    hparams = Namespace()
    oracle = RegionOracle(hparams, AskAgent.reg_actions)
    oracle.add_scans(set(scans))

    for scan in scans:
        with open('../../../data/view_to_region/' + scan + '.panorama_to_region.txt') as f:
            for line in f:
                fields = line.rstrip().split()
                view = fields[1]
                r1 = fields[-1]
                r2 = oracle._determine_region({ 'scan' : scan, 'viewpoint' : view })
                assert r1 == r2
                assert oracle._map_env_action_to_agent_action(r2,
                        { 'scan' : scan, 'viewpoint' : view, 'ended' : False }
                    ) == AskAgent.reg_actions.index(r1)

    print(colored('PASS REGION ORACLE TEST', 'green'))


test_shortest_oracle()
test_multistep_shortest_oracle(1)
test_multistep_shortest_oracle(2)
test_multistep_shortest_oracle(3)
#test_ask_oracle()
test_step_oracle()
