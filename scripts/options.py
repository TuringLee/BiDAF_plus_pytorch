#!/usr/bin/enc python3
# Option script of Jarvis for SQuAD .
# Wirtten by Turing Lee in ColorfulClouds .

import argparse
import logging

logger = logging.getLogger(__name__)

def get_parser(desc):
    parser = argparse.ArgumentParser(description = 'Jarvis system -- ')
    parser.add_argument('--display_iter', type = int, default = 500, help = 'log progress every N updates.')
    parser.add_argument('--random_seed', default = 1, type = int, help='pseudo random number generator seed.')
    parser.add_argument('--log_file', type = str, help='Path to log file.')
    parser.add_argument('--save_dir', type = str, help='Path to save checkpoints.')
    return parser
    

def add_dataset_args(parser):
    group = parser.add_argument_group('Dataset and data loading.')
    group.add_argument('--embedding_file', type = str, help = 'Path to Space-separated \
                                               pretrained embedding file.')
    group.add_argument('--train_file', type = str, help = 'Path to preprocessed training file.')
    group.add_argument('--dev_file', type = str, help = 'Path to preprocessed develop file.')
    group.add_argument('--dev_json', type = str, help = 'Path to original develop file.')
    group.add_argument('--max_len', type = int, default = 17, help = 'max span length to consider' )
    return group
    

def add_trainer_args(parser):
    group = parser.add_argument_group('Trainer arguments')
    group.add_argument('--checkpoint', action = 'store_true', default = False)
    group.add_argument('--model_name', type = str, default = 'Jarvis_BiDAF' )
    group.add_argument('--skip_no_answer', action = 'store_true', default = True )
    group.add_argument('--no_cuda', action = 'store_true', default = False )
    group.add_argument('--parallel', action = 'store_true', default = False )
    group.add_argument('--sort_by_len', action = 'store_true', default = True )
    group.add_argument('--batch_size', type = int, default = 32 )
    group.add_argument('--data_workers', type = int, default = 1 )
    group.add_argument('--dev_batch_size', type = int, default = 32 )
    group.add_argument('--valid_metric', type = str, default = 'f1' )
    group.add_argument('--restrict_vocab', action = 'store_true', default = True)
    group.add_argument('--fixed_embedding', action = 'store_false', default = True)
    group.add_argument('--num_epochs', type = int, default = 25)
    return group    
    
def add_model_args(parser):
    group = parser.add_argument_group('Model arguments')
    group.add_argument('--embedding_dim', type = int, default = 100)
    group.add_argument('--char_emb_dim', type = int, default = 20)
    group.add_argument('--out_char_dim', type = int, default = 100)
    group.add_argument('--num_chars', type = int, default = 256)
    group.add_argument('--cnn_kernel_size', type = int, default = 5)
    group.add_argument('--num_highways', type = int, default = 2)
    group.add_argument('--model_lstm_layers', type = int, default = 1)
    group.add_argument('--output_lstm_layers', type = int, default = 1)
    group.add_argument('--model_dim', type = int, default = 100)
    group.add_argument('--dropout', type = float, default = 0.2)
    group.add_argument('--elmo_dropout', type = float, default = 0.2)
    group.add_argument('--multi_head_dropout', type = float, default = 0.2)
    group.add_argument('--uncased_question', action = 'store_true', default = True)
    group.add_argument('--uncased_doc', action = 'store_true', default = True)
    group.add_argument('--use_char_emb', action = 'store_true', default = True)
    group.add_argument('--use_in_question', action = 'store_true', default = True)
    group.add_argument('--use_lemma', action = 'store_true', default = True)
    group.add_argument('--use_pos', action = 'store_true', default = True)
    group.add_argument('--use_ner', action = 'store_true', default = True)
    group.add_argument('--use_tf', action = 'store_true', default = False)
    group.add_argument('--network_file', type = str, default = 'BiDAF')
    group.add_argument('--add_features', action = 'store_true', default = False)
    group.add_argument('--share_lstm_weight', action = 'store_false', default = True)
    group.add_argument('--residual', action = 'store_true', default = False)
    group.add_argument('--layer_norm', action = 'store_true', default = False)
    group.add_argument('--tune_unk', action = 'store_true', default = False)
    group.add_argument('--self_attn', action = 'store_true', default = False)
    group.add_argument('--num_heads', type = int, default = 4)
    group.add_argument('--rnn_type', type = str, default = 'lstm')
    group.add_argument('--add_elmo', action = 'store_true', default = False)
    group.add_argument('--elmo_options_file', type = str)
    group.add_argument('--elmo_weight_file', type = str)
    
    return group

def add_optimization_args(parser):
    group = parser.add_argument_group('Otimizer_arguments')
    group.add_argument('--optimizer', type = str, default = 'adam')
    group.add_argument('--learning_rate', type = float, default = 0.001)
    group.add_argument('--grad_clipping', type = float, default = 5)
    group.add_argument('--weight_decay', type = float, default = 0)
    group.add_argument('--loss_fn', type = str, default = 'nll')
    group.add_argument('--lrshrink', type = float, default = 0)
    group.add_argument('--tune_partial', type = int, default = 0)
    return group    

def override_model_args(old_args, new_args):
    old_args, new_args = vars(old_args), vars(new_args)
    for k in old_args.keys():
        if k in new_args and old_args[k] != new_args[k]:
            logger.info('Overriding saved %s: %s --> %s' % 
                        (k, old_args[k], new_args[k]))
            old_args[k] = new_args[k]
        else:
            logger.info('Keeping saved %s: %s' % (k, old_args[k]))
    return argparse.Namespace(**old_args)

def parse_ars_and_arch(parser):
    args = parser.parse_args()
    return args
    