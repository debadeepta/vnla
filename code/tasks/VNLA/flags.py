# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

def make_parser():
   parser = argparse.ArgumentParser()

   parser.add_argument('-config_file', type=str,
        help='configuration file')
   parser.add_argument('-load_path', type=str,
        help='path to load a pretrained model')
   parser.add_argument('-start_path', type=str,
        help='path to load the starting model for rl')
   parser.add_argument('-exp_name', type=str,
        help='name of the experiment')
   parser.add_argument('-seed', type=int,
        help='random seed')
   parser.add_argument('-data_dir', type=str,
        help='data directory')
   parser.add_argument('-img_features', type=str,
        help='path to pretrained image embeddings')
   parser.add_argument('-img_feature_size', type=int, default=2048,
        help='image embedding size')
   parser.add_argument('-max_input_length', type=int,
        help='maximum input instruction length')
   parser.add_argument('-batch_size', type=int,
        help='batch size (both training and evaluation)')
   parser.add_argument('-max_episode_length', type=int,
        help='maximum number of actions per epsiode')
   parser.add_argument('-word_embed_size', type=int,
        help='word embedding size')
   parser.add_argument('-action_embed_size', type=int,
        help='navigation action embedding size')
   parser.add_argument('-ask_embed_size', type=int,
        help='ask action embedding size')
   parser.add_argument('-hidden_size', type=int,
        help='number of LSTM hidden units')
   parser.add_argument('-bidrectional', type=int,
        help='bidirectional encoder')
   parser.add_argument('-dropout_ratio', type=float,
        help='dropout probability')
   parser.add_argument('-nav_feedback', type=str,
        help='navigation training method (deprecated)')
   parser.add_argument('-ask_feedback', type=str,
        help='navigation training method (deprecated)')
   parser.add_argument('-lr', type=float,
        help='learning rate')
   parser.add_argument('-weight_decay', type=float,
        help='L2-regularization weight')
   parser.add_argument('-n_iters', type=int,
        help='number of training iterations (batches)')
   parser.add_argument('-min_word_count', type=int,
        help='minimum word count cutoff when building vocabulary')
   parser.add_argument('-split_by_spaces', type=int,
        help='split word by spaces (always set true)')
   parser.add_argument('-start_lr_decay', type=int,
        help='iteration to start decaying learning rate')
   parser.add_argument('-lr_decay_rate', type=float,
        help='learning rate decay rate')
   parser.add_argument('-decay_lr_every', type=int,
        help='number of iterations between learning rate decays')
   parser.add_argument('-save_every', type=int,
        help='number of iterations between model savings')
   parser.add_argument('-log_every', type=int,
        help='number of iterations between information loggings')
   parser.add_argument('-success_radius', type=float,
        help='success radius')

   parser.add_argument('-external_main_vocab', type=str,
        help='provide a different vocab file')

   # Advisor
   parser.add_argument('-advisor', type=str,
        help="type of advisor ('direct' or 'verbal'")
   parser.add_argument('-query_ratio', type=float,
        help='ratio between number of steps the agent is assisted and total number of steps (tau)')
   parser.add_argument('-n_subgoal_steps', type=int,
        help='number of next actions suggested by a subgoal')
   parser.add_argument('-subgoal_vocab', type=str,
        help='subgoal vocabulary')

   # Help-requesting teacher hyperparameters
   parser.add_argument('--deviate_threshold', type=float)
   parser.add_argument('--uncertain_threshold', type=float)
   parser.add_argument('--unmoved_threshold', type=int)


   # Baselines
   parser.add_argument('-random_ask', type=int)
   parser.add_argument('-ask_first', type=int)
   parser.add_argument('-teacher_ask', type=int)
   parser.add_argument('-no_ask', type=int)

   # Don't touch these ones
   parser.add_argument('-backprop_softmax', type=int, default=1)
   parser.add_argument('-backprop_ask_features', type=int)

   # Budget Features
   parser.add_argument('-max_ask_budget', type=int, default=20,
        help='budget upperbound')

   # Evaluation
   parser.add_argument('-eval_only', type=int,
        help='evaluation mode')
   parser.add_argument('-multi_seed_eval', type=int,
        help='evaluate with multiple seeds (automatically set -eval_only 1)')
   parser.add_argument('-teacher_interpret', type=int,
        help='0 = evaluate with indirect advisor    1 = evaluate with direct advisor')

   # Others
   parser.add_argument('-device_id', type=int, default=0,
        help='gpu id')
   parser.add_argument('-no_room', type=int,
        help='train or evaluate with the no_room dataset (when using this, set -data_dir noroom)')

   parser.add_argument('-rule_a_e', type=int,
        help='Use rules (a) and (e) only for help-requesting teacher')
   parser.add_argument('-rule_b_d', type=int,
        help='Use rules (b) to (d) only for help-requesting teacher')


   return parser
