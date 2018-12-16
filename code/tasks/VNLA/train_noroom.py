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

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince
from env import R2RBatch
from model import AttentionSeq2SeqModel
from agent import Seq2SeqAgent
from ask_agent import AskSeq2SeqAgent
from verbal_ask_agent import StepByStepAskSeq2SeqAgent

from eval import Evaluation
from oracle import *
from flags import make_parser

def set_path():
    OUTPUT_DIR = os.getenv('PT_OUTPUT_DIR', 'output')

    if hasattr(hparams, 'ask_for_help') and hparams.ask_for_help:
        hparams.model_prefix = '%s_nav_%s_ask_%s' % (hparams.exp_name,
            hparams.nav_feedback, hparams.ask_feedback)
    else:
        hparams.model_prefix = '%s_nav_%s' % (hparams.exp_name, hparams.nav_feedback)

    hparams.exp_dir = os.path.join(OUTPUT_DIR, hparams.model_prefix)
    if not os.path.exists(hparams.exp_dir):
        os.makedirs(hparams.exp_dir)

    hparams.result_dir = os.path.join(hparams.exp_dir, 'results')
    if not os.path.exists(hparams.result_dir):
        os.makedirs(hparams.result_dir)

    hparams.snapshot_dir = os.path.join(hparams.exp_dir, 'snapshots')
    if not os.path.exists(hparams.snapshot_dir):
        os.makedirs(hparams.snapshot_dir)

    hparams.plot_dir = os.path.join(hparams.exp_dir, 'plots')
    if not os.path.exists(hparams.plot_dir):
        os.makedirs(hparams.plot_dir)

    DATA_DIR = os.getenv('PT_DATA_DIR', '../../../data')
    hparams.data_path = os.path.join(DATA_DIR, hparams.data_dir)
    hparams.img_features = os.path.join(DATA_DIR, 'img_features/ResNet-152-imagenet.tsv')

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

def load(path):
    global hparams
    ckpt = torch.load(path)

    data_dir = hparams.data_dir
    hparams = ckpt['hparams']
    hparams.data_dir = data_dir
    hparams.find_object = True

    # Overwrite hparams by args
    for flag in vars(args):
        value = getattr(args, flag)
        if value is not None:
            setattr(hparams, flag, value)

    set_path()
    return ckpt

def compute_ask_stats(traj):
    total_steps = 0
    total_agent_ask = 0
    total_oracle_ask = 0
    queries_per_ep = []
    ask_pred = []
    ask_true = []
    reg_pred = []
    reg_true = []

    all_reasons = []
    loss_str = ''

    for i, t in enumerate(traj):
        assert len(t['agent_ask']) == len(t['oracle_ask'])

        end_step = len(t['agent_path'])

        pred = t['agent_ask'][:end_step]
        true = t['oracle_ask'][:end_step]

        total_steps += len(true)
        total_agent_ask  += len(
            filter(lambda x: x == AskSeq2SeqAgent.ask_actions.index('ask'), pred))
        total_oracle_ask += len(
            filter(lambda x: x == AskSeq2SeqAgent.ask_actions.index('ask'), true))
        ask_pred.extend(pred)
        ask_true.extend(true)

        queries_per_ep.append(len(filter(lambda x: x == 1, pred)))
        oracle_reason = t['oracle_ask_reason'][:end_step]
        all_reasons.extend(oracle_reason)

        if hparams.region_predict or hparams.oracle_region:
            reg_pred.extend(t['agent_reg'][:end_step])
            reg_true.extend(t['oracle_reg'][:end_step])

    loss_str += '\n *** ASK:'
    loss_str += ' queries_per_ep %.1f' % (sum(queries_per_ep) / len(queries_per_ep))
    loss_str += ', agent_ratio %.3f' %  (total_agent_ask  / total_steps)
    loss_str += ', oracle_ratio %.3f' % (total_oracle_ask / total_steps)
    loss_str += ', A/P/R/F %.3f / %.3f / %.3f / %.3f' % (
                                            accuracy_score(ask_true, ask_pred),
                                            precision_score(ask_true, ask_pred),
                                            recall_score(ask_true, ask_pred),
                                            f1_score(ask_true, ask_pred))

    loss_str += '\n *** ORACLE ASK:'
    reason_counter = Counter(all_reasons)
    total_asks = len(filter(lambda x: x != 'pass' and x != 'exceed', all_reasons))
    loss_str += ' ask %.3f, dont_ask %.3f, ' % (
            total_asks / len(all_reasons),
            (len(all_reasons) - total_asks) / len(all_reasons)
        )
    loss_str += ', '.join(
        ['%s %.3f' % (k, reason_counter[k] / total_asks)
            for k in reason_counter.keys() if k not in ['pass', 'exceed']])

    if hparams.region_predict or hparams.oracle_region:
        loss_str += '\n *** REGION: acc %.3f' % accuracy_score(reg_true, reg_pred)

    return loss_str

def train(train_env, agent, model, optimizer, start_iter, end_iter, best_metrics, val_envs={}):
    ''' Train on training set, validating on both seen and unseen. '''

    #save('tmp.ckpt', model, optimizer, start_iter, best_metrics, train_env)

    print 'Training with %s feedback with lr = %f' % (
        hparams.nav_feedback, optimizer.param_groups[0]['lr'])

    train_feedback = { 'nav' : hparams.nav_feedback }
    test_feedback  = { 'nav' : 'argmax' }

    if hasattr(hparams, 'ask_for_help') and hparams.ask_for_help:
        train_feedback['ask'] = hparams.ask_feedback
        test_feedback['ask'] = 'argmax'

        if hparams.region_predict:
            train_feedback['reg'] = hparams.reg_feedback
            test_feedback['reg']  = 'argmax'

        if hparams.region_feature:
            assert hparams.oracle_region or hparams.region_predict, \
                'region_feature requires oracle_region or region_predict to be True'

        if hparams.oracle_region:
            assert hparams.region_feature, 'oracle_region requires region_feature to be True'
            assert not hparams.region_predict, 'oracle_region requires region_predict to be False'
            train_feedback['reg'] = 'teacher'
            test_feedback['reg']  = 'teacher'

    start = time.time()

    for idx in range(start_iter, end_iter, hparams.log_every):
        interval = min(hparams.log_every, end_iter - idx)
        iter = idx + interval

        # Train for log_every interval

        if not hasattr(hparams, 'eval_only') or not hparams.eval_only:
            traj = agent.train(train_env, optimizer, interval, train_feedback)
            train_losses = np.array(agent.losses)
            assert len(train_losses) == interval
            train_loss_avg = np.average(train_losses)
            loss_str = '\n * train loss: %.4f' % train_loss_avg

            if hasattr(hparams, 'ask_for_help') and hparams.ask_for_help:
                train_nav_loss_avg = np.average(np.array(agent.nav_losses))
                train_ask_loss_avg = np.average(np.array(agent.ask_losses))
                train_reg_loss_avg = np.average(np.array(agent.reg_losses))
                loss_str += ', nav loss: %.4f' % train_nav_loss_avg
                loss_str += ', ask loss: %.4f' % train_ask_loss_avg
                loss_str += ', reg loss: %.4f' % train_reg_loss_avg
                loss_str += compute_ask_stats(traj)
        else:
            loss_str = '\n * eval mode'

        # Run validation
        metrics = {}
        should_save_ckpt = []

        for env_name, (env, evaluator) in val_envs.iteritems():
            # Get validation loss under the same conditions as training
            agent.test(env, train_feedback, use_dropout=True, allow_cheat=True)
            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)

            # Get validation distance from goal under test evaluation conditions
            traj = agent.test(env, test_feedback, use_dropout=False, allow_cheat=False)

            agent.results_path = os.path.join(hparams.result_dir,
                '%s_%s_for_eval.json' % (hparams.model_prefix, env_name))
            agent.write_results(traj)
            score_summary, _, is_success = evaluator.score(agent.results_path)

            if hasattr(hparams, 'eval_only') and hparams.eval_only:
                agent.results_path = hparams.load_path.replace('ckpt', '') + env_name + '.json'
                agent.add_is_success(is_success)
                print 'Save result to', agent.results_path
                agent.write_results(traj)

            loss_str += '\n * %s loss: %.4f' % (env_name, val_loss_avg)
            if hasattr(hparams, 'ask_for_help') and hparams.ask_for_help:
                val_nav_loss_avg = np.average(np.array(agent.nav_losses))
                val_ask_loss_avg = np.average(np.array(agent.ask_losses))
                val_reg_loss_avg = np.average(np.array(agent.reg_losses))
                loss_str += ', nav loss: %.4f' % val_nav_loss_avg
                loss_str += ', ask loss: %.4f' % val_ask_loss_avg
                loss_str += ', reg loss: %.4f' % val_reg_loss_avg

            for metric,val in score_summary.iteritems():
                if metric in ['success_rate', 'oracle_rate']:
                    loss_str += ', %s: %.3f' % (metric, val)
                    if metric == 'success_rate':
                        metrics[env_name] = (val, len(traj))

            loss_str += '\n *** OTHER METRICS: '
            loss_str += '%s: %.2f' % ('nav_error', score_summary['nav_error'])
            loss_str += ', %s: %.2f' % ('oracle_error', score_summary['oracle_error'])
            loss_str += ', %s: %.2f' % ('length', score_summary['length'])
            loss_str += ', %s: %.2f' % ('steps', score_summary['steps'])

            if hasattr(hparams, 'ask_for_help') and hparams.ask_for_help:
                loss_str += compute_ask_stats(traj)

            # Update best success rate for unseen/seen val
            if 'test' not in env_name and metrics[env_name][0] > best_metrics[env_name]:
                should_save_ckpt.append(env_name)
                best_metrics[env_name] = metrics[env_name][0]
                print('best %s success rate %.3f' % (env_name, metrics[env_name][0]))

        """
        if not hasattr(hparams, 'eval_only') or not hparams.eval_only:
            combined_metric = (
                metrics['val_seen'][0]   * metrics['val_seen'][1] + \
                metrics['val_unseen'][0] * metrics['val_unseen'][1]) / \
                (metrics['val_seen'][1] + metrics['val_unseen'][1])
            if combined_metric > best_metrics['combined']:
                should_save_ckpt.append('combined')
                best_metrics['combined'] = combined_metric
                print('best combined success rate %.3f' % combined_metric)
        """

        print('%s (%d %d%%) %s' % (timeSince(start, float(iter)/end_iter),
                                             iter, float(iter)/end_iter*100, loss_str))

        if hasattr(hparams, 'eval_only') and hparams.eval_only:
            break

        # Learning rate decay
        if hparams.lr_decay_rate and metrics['val_unseen'][0] < best_metrics['val_unseen'] and \
            iter >= hparams.start_lr_decay and iter % hparams.decay_lr_every == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= hparams.lr_decay_rate
                print('New learning rate %f' % param_group['lr'])

        # Save lastest model
        if iter == end_iter or iter % hparams.save_every == 0:
            should_save_ckpt.append('last')

        for env_name in should_save_ckpt:
            save_path = os.path.join(hparams.snapshot_dir,
                '%s_%s.ckpt' % (hparams.model_prefix, env_name))
            save(save_path, model, optimizer, iter, best_metrics, train_env)
            print("Saved %s model to %s" % (env_name, save_path))



def setup():
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
                    split_by_spaces=hparams.split_by_spaces),
            train_vocab_path)

def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''

    # Resume from lastest checkpoint (if any)
    hparams.load_path = hparams.load_path if hasattr(hparams, 'load_path') and \
        hparams.load_path is not None else \
        os.path.join(hparams.snapshot_dir, '%s_last.ckpt' % hparams.model_prefix)
    if os.path.exists(hparams.load_path):
        print('Load model from %s' % hparams.load_path)
        ckpt = load(hparams.load_path)
        start_iter = ckpt['iter']
    else:
        ckpt = None
        start_iter = 0

    end_iter = hparams.n_iters

    setup()
    # Create a batch training environment that will also preprocess text
    train_vocab_path = os.path.join(hparams.data_path, 'train_vocab.txt')
    if hparams.oracle == 'verbal':
        subgoal_vocab_path = os.path.join(hparams.data_path, hparams.subgoal_vocab)
        vocab = read_vocab([train_vocab_path, subgoal_vocab_path])
    else:
        vocab = read_vocab([train_vocab_path])

    if ckpt is not None:
        vocab = ckpt['vocab']
        print 'Load vocab of size', len(vocab)

    tok = Tokenizer(vocab=vocab, encoding_length=hparams.max_input_length)
    train_env = R2RBatch(hparams, splits=['train'], tokenizer=tok)

    # Create validation environments

    val_splits = ['val_unseen']
    if hasattr(hparams, 'eval_only') and hparams.eval_only:
        if hasattr(hparams, 'ask_for_help') and hparams.ask_for_help:
            val_splits.extend(['test_unseen'])
        else:
            val_splits.append('test')

    val_envs = { split: (R2RBatch(hparams, splits=[split], tokenizer=tok,
        from_train_env=train_env, traj_len_estimates=train_env.traj_len_estimates),
        Evaluation(hparams, [split], hparams.data_path)) for split in val_splits}

    # Build models and train
    model = AttentionSeq2SeqModel(len(vocab), hparams)

    optimizer = optim.Adam(model.parameters(), lr=hparams.lr,
        weight_decay=hparams.weight_decay)

    best_metrics = { 'val_unseen': -1 }

    if ckpt is not None:
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optim_state_dict'])
        best_metrics = ckpt['best_metrics']
        train_env.ix = ckpt['data_idx']

    print('')
    pprint(vars(hparams), width=1)
    print('')
    print(model)

    if hasattr(hparams, 'ask_for_help') and hparams.ask_for_help:
        if hparams.oracle == 'verbal':
            if hparams.subgoal_oracle == 'step_by_step':
                agent = StepByStepAskSeq2SeqAgent(model, hparams)
            if hparams.subgoal_oracle == 'natural':
                agent = NaturalAskSeq2SeqAgent(model, hparams)
        elif hparams.oracle == 'next_optimal':
            agent = AskSeq2SeqAgent(model, hparams)
    else:
        agent = Seq2SeqAgent(model, hparams)

    train(train_env, agent, model, optimizer, start_iter, end_iter,
          best_metrics, val_envs=val_envs)

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
    train_val()

