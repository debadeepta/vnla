from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import os
import time
import numpy as np
import pandas as pd
import argparse
import json
from collections import defaultdict, Counter
from argparse import Namespace
from pprint import pprint
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy import stats

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, load_panos_to_region
from m1_env import VNLABatch
from model import AttentionSeq2SeqModel
from ask_agent import AskAgent
from verbal_ask_agent import VerbalAskAgent
from m1_agent import M1Agent

from eval import Evaluation
from oracle import *
from flags import make_parser

def set_path():
    OUTPUT_DIR = os.getenv('PT_OUTPUT_DIR', 'output')

    hparams.model_prefix = '%s_nav_%s_ask_%s' % (hparams.exp_name,
        hparams.nav_feedback, hparams.ask_feedback)

    hparams.exp_dir = os.path.join(OUTPUT_DIR, hparams.model_prefix)
    if not os.path.exists(hparams.exp_dir):
        os.makedirs(hparams.exp_dir)

    hparams.load_path = hparams.load_path if hasattr(hparams, 'load_path') and \
        hparams.load_path is not None else \
        os.path.join(hparams.exp_dir, '%s_last.ckpt' % hparams.model_prefix)

    DATA_DIR = os.getenv('PT_DATA_DIR', '../../../data')
    hparams.data_path = os.path.join(DATA_DIR, hparams.data_dir)
    hparams.img_features = os.path.join(DATA_DIR, 'img_features/ResNet-152-imagenet.tsv')

    # Imitation learning path
    hparams.start_path = "output/main_learned_nav_sample_ask_sample/main_learned_nav_sample_ask_sample_val_seen.ckpt"
    # TODO: add another path for val_unseen.ckpt

def save(path, model, optimizer, iter, best_metrics, train_env):
    ckpt = {
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'hparams'         : hparams,
            'iter'            : iter,
            'best_metrics'    : best_metrics,
            'data_idx'        : train_env.ix,
            'vocab'           : train_env.tokenizer.vocab
        }
    torch.save(ckpt, path)

def load(path, device):
    ckpt = torch.load(path, map_location=device)
    set_path()
    return ckpt

def compute_ask_stats(traj,agent):
    total_steps = 0
    total_agent_ask = 0
    total_teacher_ask = 0
    queries_per_ep = []
    ask_pred = []
    ask_true = []
    bad_questions = []

    all_reasons = []
    loss_str = ''

    nav_oracle = agent.advisor.nav_oracle

    for i, t in enumerate(traj):
        assert len(t['agent_ask']) == len(t['teacher_ask'])

        end_step = len(t['agent_path'])

        pred = t['agent_ask'][:end_step]
        true = t['teacher_ask'][:end_step]
        
        ### BAD QUESTION

        path = t['agent_path']

        bad_question_marks = [0] * len(path)

        # bad question rule 1
        for index in range(len(pred) - 1):
            if pred[index] == pred[index + 1] == AskAgent.ask_actions.index('direction') and \
                    path[index] == path[index + 1]:
                bad_question_marks[index + 1] = 1
        
        # bad question rule 2
        scan = t['scan']
        goal_viewpoints = t['goal_viewpoints']

        distance_indices = [index for index, question in enumerate(pred) if question == AskAgent.ask_actions.index('distance')]
        for index in range(len(distance_indices)-1):
            _, goal_point = nav_oracle._find_nearest_point(scan, path[distance_indices[index]][0], goal_viewpoints)
            d1, _ = nav_oracle._find_nearest_point_on_a_path(scan, path[distance_indices[index]][0], path[0][0],
                                                             goal_point)
            d2, _ = nav_oracle._find_nearest_point_on_a_path(scan, path[distance_indices[index+1]][0], path[0][0],
                                                             goal_point)
            if abs(d1-d2) <= 3:
                bad_question_marks[distance_indices[index+1]] = 1

        # bad question rule 3
        panos_to_region = load_panos_to_region(scan, None, include_region_id=True)
        room_indices = [index for index, question in enumerate(pred) if
                        question == AskAgent.ask_actions.index('room')]
        for index in range(len(room_indices) - 1):
            region_id_1, region_1 = panos_to_region[path[room_indices[index]][0]]
            region_id_2, region_2 = panos_to_region[path[room_indices[index + 1]][0]]
            if region_id_1 == region_id_2 and region_1 == region_2:
                bad_question_marks[room_indices[index + 1]] = 1

        # bad question rule 4
        goal_viewpoints = t['goal_viewpoints']
        for index in range(len(pred) - 1):
            if pred[index] == AskAgent.ask_actions.index('arrive'):
                d, goal_point = nav_oracle._find_nearest_point(scan, path[index][0], goal_viewpoints)
                if d >= 4:
                    bad_question_marks[index] = 1

        bad_questions.append(sum(bad_question_marks))

        ### BAD QUESTION


        total_steps += len(true)
        total_agent_ask += sum(any(x == AskAgent.ask_actions.index(question) for question in AskAgent.question_pool)
                               for x in pred)   # TBD
        total_teacher_ask += sum(any(x == AskAgent.ask_actions.index(question) for question in AskAgent.question_pool)
                                 for x in true)
        ask_pred.extend(pred)
        ask_true.extend(true)

        queries_per_ep.append(sum(any(x == AskAgent.ask_actions.index(question) for question in AskAgent.question_pool)
                                  for x in pred))
        teacher_reason = t['teacher_ask_reason'][:end_step]
        all_reasons.extend(teacher_reason)

    loss_str += '\n *** ASK:'
    loss_str += ' queries_per_ep %.1f' % (sum(queries_per_ep) / len(queries_per_ep))
    loss_str += ', agent_ratio %.3f' %  (total_agent_ask  / total_steps)
    loss_str += ', teacher_ratio %.3f' % (total_teacher_ask / total_steps)
    loss_str += ', A/P/R/F %.3f / %.3f / %.3f / %.3f' % (
                                            accuracy_score(ask_true, ask_pred),
                                            precision_score(ask_true, ask_pred, average='macro'),
                                            recall_score(ask_true, ask_pred, average='macro'),
                                            f1_score(ask_true, ask_pred, average='macro'))
    loss_str += ', bad_questions_per_ep %.1f' % (sum(bad_questions) / len(bad_questions))

    loss_str += '\n *** TEACHER ASK:'
    reason_counter = Counter(all_reasons)
    total_asks = sum(x != 'pass' and x != 'exceed' for x in all_reasons)
    loss_str += ' ask %.3f, dont_ask %.3f, ' % (
            total_asks / len(all_reasons),
            (len(all_reasons) - total_asks) / len(all_reasons)
        )
    loss_str += ', '.join(
        ['%s %.3f' % (k, reason_counter[k] / total_asks)
            for k in reason_counter.keys() if k not in ['pass', 'exceed']])

    return loss_str

def train(train_env, val_envs, agent, model, optimizer, start_iter, end_iter,
    best_metrics, eval_mode):

    if not eval_mode:
        print('Training with with lr = %f' % optimizer.param_groups[0]['lr'])

    # Ask feedback does not matter as we are using epsilon greedy
    train_feedback = { 'nav' : 'argmax', 'ask' : 'argmax' }
    test_feedback  = { 'nav' : 'argmax', 'ask' : 'argmax' }

    start = time.time()
    sr = 'success_rate'
    
    print(start_iter)
    print(end_iter)
    for idx in range(start_iter, end_iter, hparams.log_every):
        loss_str = ''

        interval = min(hparams.log_every, end_iter - idx)
        start_episode = idx
        end_episode = idx + interval
        iter = end_episode          # Legacy code still use this variable

        # Train for log_every iterations
        if eval_mode:
            loss_str = '\n * eval mode'
        else:
            agent.train(train_env, optimizer, start_episode, end_episode, train_feedback)

        metrics = defaultdict(dict)
        should_save_ckpt = []

        # Run validation
        eval_success_rates = [None] * 3
        for env_name, (env, evaluator) in val_envs.items():
            # Get validation distance from goal under test evaluation conditions
            longer_time = env_name == 'val_seen_longer_time'        # This validation environment lets the agent run with maximum time
            traj = agent.test(env, test_feedback, use_dropout=False, allow_cheat=False, allow_max_episode_length=longer_time)

            agent.results_path = os.path.join(hparams.exp_dir,
                '%s_%s_for_eval.json' % (hparams.model_prefix, env_name))
            agent.write_results(traj)
            score_summary, _, is_success = evaluator.score(agent.results_path)

            if eval_mode:
                agent.results_path = hparams.load_path.replace('ckpt', '') + env_name + '.json'
                agent.add_is_success(is_success)
                print('Save result to', agent.results_path)
                agent.write_results(traj)

            loss_str += '\n *** MAIN METRICS (%s)' % (env_name)
            for metric, val in score_summary.items():
                if metric in ['success_rate', 'oracle_rate', 'room_success_rate',
                    'nav_error', 'length', 'steps']:
                    metrics[metric][env_name] = (val, len(traj))
                if metric in ['success_rate', 'oracle_rate', 'room_success_rate']:
                    loss_str += ', %s: %.3f' % (metric, val)

            loss_str += '\n *** OTHER METRICS: '
            loss_str += '%s: %.2f' % ('nav_error', score_summary['nav_error'])
            loss_str += ', %s: %.2f' % ('oracle_error', score_summary['oracle_error'])
            loss_str += ', %s: %.2f' % ('length', score_summary['length'])
            loss_str += ', %s: %.2f' % ('steps', score_summary['steps'])
            loss_str += compute_ask_stats(traj, agent)

            if not eval_mode:
                success_rate = metrics[sr][env_name][0] * 100.0
                idx = 0
                if env_name == 'val_seen':
                    idx = 1
                elif env_name == 'val_unseen':
                    idx = 2
                eval_success_rates[idx] = success_rate

                if env_name in best_metrics and metrics[sr][env_name][0] > best_metrics[env_name]:
                    should_save_ckpt.append(env_name)
                    best_metrics[env_name] = metrics[sr][env_name][0]
                    print('best %s success rate %.3f' % (env_name, best_metrics[env_name]))

        if not eval_mode:
            combined_metric = (
                metrics[sr]['val_seen'][0]   * metrics[sr]['val_seen'][1] + \
                metrics[sr]['val_unseen'][0] * metrics[sr]['val_unseen'][1]) / \
                (metrics[sr]['val_seen'][1]  + metrics[sr]['val_unseen'][1])
            if combined_metric > best_metrics['combined']:
                should_save_ckpt.append('combined')
                best_metrics['combined'] = combined_metric
                print('best combined success rate %.3f' % combined_metric)

        if not eval_mode:
            # Add a single datapoint to our eval
            agent.plotter.add_eval_data_point(end_episode - 1, *eval_success_rates)

            # Save graph plotter for easier tracking of DQN's performance
            agent.plotter.save()

        print('%s (%d %d%%) %s' % (timeSince(start, float(iter)/end_iter),
            iter, float(iter)/end_iter*100, loss_str))

        if eval_mode:
            res = defaultdict(dict)
            for metric in metrics:
                for k, v in metrics[metric].items():
                    res[metric][k] = v[0]
            return res

        if not eval_mode:
            # Learning rate decay
            if hparams.lr_decay_rate and combined_metric < best_metrics['combined'] \
                and iter >= hparams.start_lr_decay and iter % hparams.decay_lr_every == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= hparams.lr_decay_rate
                    print('New learning rate %f' % param_group['lr'])

            # Save lastest model?
            if iter == end_iter or iter % hparams.save_every == 0:
                should_save_ckpt.append('last')

            for env_name in should_save_ckpt:
                save_path = os.path.join(hparams.exp_dir,
                    '%s_%s.ckpt' % (hparams.model_prefix, env_name))
                save(save_path, model, optimizer, iter, best_metrics, train_env)
                print("Saved %s model to %s" % (env_name, save_path))

    return None

def setup(seed=None):

    if seed is not None:
        hparams.seed = seed
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    # Check for vocabs
    train_vocab_path = os.path.join(hparams.data_path, 'train_vocab.txt')
    if not os.path.exists(train_vocab_path):
        write_vocab(build_vocab(
                    hparams.data_path,
                    splits=['train'],
                    min_count=hparams.min_word_count,
                    max_length=hparams.max_input_length,
                    split_by_spaces=hparams.split_by_spaces,
                    prefix='noroom' if hasattr(hparams, 'no_room') and
                           hparams.no_room else 'asknav'),
            train_vocab_path)

def train_val(seed=None):
    ''' Train on the training set, and validate on seen and unseen splits. '''

    global hparams

    # which GPU to use
    device = torch.device('cuda', hparams.device_id)

    new_training = True

    # Resume from latest checkpoint (if any)
    if os.path.exists(hparams.load_path):
        print('Load model from %s' % hparams.load_path)
        ckpt = load(hparams.load_path, device)

        # Reload hparams using checkpoint's hparams
        hparams = ckpt['hparams']
        # Overwrite hparams by args
        for flag in vars(args):
            value = getattr(args, flag)
            if value is not None:
                setattr(hparams, flag, value)

        start_iter = ckpt['iter']
        new_training = False
    elif os.path.exists(hparams.start_path):         # Continue training from imitation learning
        print('Continue training from imitation learning phase!\nLoad model from %s' % hparams.start_path)
        ckpt = load(hparams.start_path, device)
        start_iter = 0              # Start iteration from 0
    else:
        sys.exit("Checkpoint from imitation learning was not provided!")
    end_iter = hparams.n_iters

    # Setup seed and read vocab
    setup(seed=seed)

    train_vocab_path = os.path.join(hparams.data_path, 'train_vocab.txt')
    if hasattr(hparams, 'external_main_vocab') and hparams.external_main_vocab:
        train_vocab_path = hparams.external_main_vocab

    if 'verbal' in hparams.advisor:
        subgoal_vocab_path = os.path.join(hparams.data_path, hparams.subgoal_vocab)
        vocab = read_vocab([train_vocab_path, subgoal_vocab_path])
    else:
        vocab = read_vocab([train_vocab_path])
    tok = Tokenizer(vocab=vocab, encoding_length=hparams.max_input_length)

    # Create a training environment
    train_env = VNLABatch(hparams, split='train', tokenizer=tok)

    # Create validation environments
    val_splits = ['val_seen', 'val_unseen']
    eval_mode = hasattr(hparams, 'eval_only') and hparams.eval_only
    if eval_mode:
        if '_unseen' in hparams.load_path:
            val_splits = ['test_unseen']
        if '_seen' in hparams.load_path:
            val_splits = ['test_seen']
        end_iter = start_iter + hparams.log_every

    val_envs = { split: (VNLABatch(hparams, split=split, tokenizer=tok,
        from_train_env=train_env, traj_len_estimates=train_env.traj_len_estimates),
        Evaluation(hparams, [split], hparams.data_path)) for split in val_splits}

    if not eval_mode:
        # The longer_time settings will be performed when we are ran the validation
        val_envs['val_seen_longer_time'] = val_envs['val_seen']

    # Build models
    model = AttentionSeq2SeqModel(len(vocab), hparams, device).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    target = AttentionSeq2SeqModel(len(vocab), hparams, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=hparams.lr,
        weight_decay=hparams.weight_decay)

    best_metrics = { 'val_seen'  : -1,
                     'val_unseen': -1,
                     'combined'  : -1 }

    # Load model parameters from a checkpoint (if any)
    if not new_training:
        optimizer.load_state_dict(ckpt['optim_state_dict'])
        train_env.ix = ckpt['data_idx']
        best_metrics = ckpt['best_metrics']

    print('')
    pprint(vars(hparams), width=1)
    print('')
    print(model)

    # Initialize agent
    train_evaluator = Evaluation(hparams, ['train'], hparams.data_path)
    agent = M1Agent(model, target, hparams, device, train_evaluator)

    # Train
    return train(train_env, val_envs, agent, model, optimizer, start_iter, end_iter,
          best_metrics, eval_mode)



if __name__ == "__main__":

    parser = make_parser()
    args = parser.parse_args()

    # Read configuration from a json file
    with open(args.config_file) as f:
        hparams = Namespace(**json.load(f))

    # Overwrite hparams by args
    for flag in vars(args):
        value = getattr(args, flag)
        if value is not None:
            setattr(hparams, flag, value)

    set_path()

    with torch.cuda.device(hparams.device_id):
        # Multi-seed evaluation
        if hasattr(hparams, 'multi_seed_eval') and hparams.multi_seed_eval:
            args.eval_only = 1
            seeds = [123, 435, 6757, 867, 983]
            metrics = defaultdict(lambda: defaultdict(list))
            for seed in seeds:
                this_metrics = train_val(seed=seed)
                for metric in this_metrics:
                    for k, v in this_metrics[metric].items():
                        if 'rate' in metric:
                            v *= 100
                        metrics[metric][k].append(v)
            for metric in metrics:
                for k, v in metrics[metric].items():
                   print('%s %s: %.2f %.2f' % (metric, k, np.average(v), stats.sem(v) * 1.95))
        else:
            # Train
            train_val()

