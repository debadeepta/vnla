import argparse

def make_parser():
   parser = argparse.ArgumentParser()

   parser.add_argument('-config_file', type=str)
   parser.add_argument('-load_path', type=str)
   parser.add_argument('-exp_name', type=str)
   parser.add_argument('-seed', type=int)
   parser.add_argument('-data_dir', type=str)
   parser.add_argument('-img_features', type=str)
   parser.add_argument('-img_feature_size', type=int, default=2048)
   parser.add_argument('-max_input_length', type=int)
   parser.add_argument('-batch_size', type=int)
   parser.add_argument('-max_episode_length', type=int)
   parser.add_argument('-word_embed_size', type=int)
   parser.add_argument('-action_embed_size', type=int)
   parser.add_argument('-ask_embed_size', type=int)
   parser.add_argument('-hidden_size', type=int)
   parser.add_argument('-bidrectional', type=int)
   parser.add_argument('-dropout_ratio', type=float)
   parser.add_argument('-nav_feedback', type=str)
   parser.add_argument('-ask_feedback', type=str)
   parser.add_argument('-lr', type=float)
   parser.add_argument('-weight_decay', type=float)
   parser.add_argument('-n_iters', type=int)
   parser.add_argument('-min_word_count', type=int)
   parser.add_argument('-split_by_spaces', type=int)
   parser.add_argument('-start_lr_decay', type=int)
   parser.add_argument('-lr_decay_rate', type=float)
   parser.add_argument('-decay_lr_every', type=int)
   parser.add_argument('-save_every', type=int)
   parser.add_argument('-log_every', type=int)
   parser.add_argument('-error_margin', type=float)

   parser.add_argument('-external_main_vocab', type=str)

   ### ORACLE
   parser.add_argument('-advisor', type=str)
   parser.add_argument('-query_ratio', type=float)
   parser.add_argument('--deviate_threshold', type=float)
   parser.add_argument('--uncertain_threshold', type=float)
   parser.add_argument('--unmoved_threshold', type=int)
   parser.add_argument('-random_ask', type=int)
   parser.add_argument('-ask_first', type=int)
   parser.add_argument('-teacher_ask', type=int)
   parser.add_argument('-no_ask', type=int)

   ### VERBAL ORACLE
   parser.add_argument('-n_subgoal_steps', type=int)
   parser.add_argument('-subgoal_vocab', type=str)
   parser.add_argument('-backprop_softmax', type=int, default=1)
   parser.add_argument('-backprop_ask_features', type=int)
   parser.add_argument('-teacher_interpret', type=int)

   ### ADDITIONAL FEATURES
   parser.add_argument('-budget_feature', type=int, default=0)
   parser.add_argument('-max_ask_budget', type=int, default=20)

   parser.add_argument('-eval_only', type=int)
   parser.add_argument('-multi_seed_eval', type=int)

   #parser.add_argument('-load_nav_agent_path', type=str)
   #parser.add_argument('-load_subgoal_oracle_path', type=str)

   parser.add_argument('-device_id', type=int, default=0)
   parser.add_argument('-no_room', type=int)

   return parser
