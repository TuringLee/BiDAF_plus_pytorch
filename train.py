#!/usr/bin/enc python3
# Train script of Jarvis for SQuAD .
# Wirtten by Turing Lee in ColorfulClouds .

import argparse
import torch
import json
import logging
import os
import sys
import numpy as np

from scripts import utils, options, data, vector
from reader import DocReader

logger = logging.getLogger()

def init_from_scratch(args, train_exs, dev_exs):
    """ New model, new data, new dictionary."""
    # Create a feature dict out of the annotations in the data.
    logger.info('-' * 100)
    logger.info('Generate features')
    feature_dict = utils.build_feature_dict(args, train_exs + dev_exs)
    logger.info('Number feature = %d' % len(feature_dict))
    logger.info(feature_dict)

    # Build a dictionary from the data questions + words (train/dev splits)
    logger.info('-' * 100)
    logger.info('Build dictionary')
    word_dict, char_dict, word_sort_by_freq = utils.build_word_char_dict(args, train_exs + dev_exs) # Whether add dev_exs to build word dict.
    logger.info('Num words = %d' % len(word_dict))
    logger.info('Num characters = %d' % len(char_dict))

    #Initialize model
    model = DocReader(args, word_dict, char_dict, feature_dict, word_sort_by_freq)

    # # Load pretrained embedding for words in dictionary
    # if args.embedding_file:
    #     model.load_embeddings(word_dict.tokens(), args.embedding_file)

    return model

def train(args, data_loader, model, global_stats):
    """ Run througn one epoch of model training with the provided data loader """
    # Initialize meters + timers
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()

    # Run one epoch
    for idx, ex in enumerate(data_loader):
        train_loss.update(*model.update(ex))

        if idx > 0 and idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d|' % 
                (global_stats['epoch'], idx, len(data_loader)) + 
                'loss = %.2f | elapsed time = %.2f (s)' % 
                (train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()

    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' % 
        (global_stats['epoch'], epoch_time.time()))

    #Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_name + '.checkpoint', 
            global_stats['epoch'] + 1)


def validate_official(args, data_loader, model, global_stats, 
                      offsets, texts, answers):
    """ 
    Uses exact spans and same exact match/F1 score computation as in SQuAD script. 
    Extra arguments:
        offsets: The character start/end indices for the tokens in each context.
        texts: Map of qid --> raw text of examples context (matchs offsets).
        answers: Map of qid --> list of accepted answers.
    """
    eval_time = utils.Timer()
    f1 = utils.AverageMeter()
    exact_match = utils.AverageMeter()

    # Run through examples
    examples = 0
    for ex in data_loader:
        ex_id = ex[-1]
        batch_size = ex[0].size(0)
        pred_s, pred_e, _ = model.predict(ex)

        for i in range(batch_size):

            s_offset = offsets[ex_id[i]][pred_s[i][0]][0]
            e_offset = offsets[ex_id[i]][pred_e[i][0]][1]
            prediction = texts[ex_id[i]][s_offset:e_offset]

            # Compute metrics
            ground_truths = answers[ex_id[i]]
            exact_match.update(utils.metric_max_over_ground_truths(
                utils.exact_match_score, prediction, ground_truths))
            f1.update(utils.metric_max_over_ground_truths(
                utils.f1_score, prediction, ground_truths))

        examples += batch_size

    logger.info('dev valid official: Epoch = %d | EM = %.2f |' % 
        (global_stats['epoch'], exact_match.avg * 100) + 
        'F1 = %.2f | examples = %d | valid time = %.2f (s)' % 
        (f1.avg * 100, examples, eval_time.time()))

    return {'exact_match': exact_match.avg * 100, 'f1': f1.avg * 100}

def main(args):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load data files')
    train_exs = utils.load_data(args, args.train_file, args.skip_no_answer )
    logger.info('Num train examples = %d' % len(train_exs))
    dev_exs = utils.load_data(args, args.dev_file, args.skip_no_answer )
    logger.info('Num dev examples = %d' % len(dev_exs))

    # Doing Offician evals
    # 1) Load the original text to retrieve spans from offsets.
    # 2) Load the text answers for each question
    dev_texts = utils.load_text(args.dev_json)
    dev_offsets = {ex['id']: ex['offsets'] for ex in dev_exs}
    dev_answers = utils.load_answers(args.dev_json)

    # --------------------------------------------------------------------------
    # Model
    logger.info('-' * 100)
    start_epoch = 0
    if args.checkpoint and os.path.isfile(args.model_name + '.checkpoint'):
        pass
    else:
        logger.info('Training model from scratch ...')
        model = init_from_scratch(args, train_exs, dev_exs)
        model.init_optimizer()

    # if args.tune_partial:
    #     pass
    if args.cuda:
        model.cuda()

    if args.parallel:
        model.parallelize()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two dataset: train and dev. If sort by length it's faster
    logger.info('-' * 100)
    logger.info('Make data loaders')
    train_dataset = data.ReaderDataset(train_exs, model, single_answer=True)
    if args.sort_by_len:
        train_sampler = data.SortedBatchSampler(train_dataset.lengths(), args.batch_size, shuffle=True)
    else:
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        sampler = train_sampler,
        num_workers = args.data_workers,
        collate_fn = vector.batchify,
        pin_memory = args.cuda,
        )
    
    dev_dataset = data.ReaderDataset(dev_exs, model, single_answer=False)
    if args.sort_by_len:
        dev_sampler = data.SortedBatchSampler(dev_dataset.lengths(), args.dev_batch_size, shuffle=True)
    else:
        dev_sampler = torch.utils.data.sampler.RandomSampler(dev_dataset)
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size = args.dev_batch_size,
        sampler = dev_sampler,
        num_workers = args.data_workers,
        collate_fn = vector.batchify,
        pin_memory = args.cuda,
        )

    # --------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' % json.dumps(vars(args), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # TRAIN&VALID LOOP
    logger.info('-' * 100)
    logger.info('Starting training... ')
    stats = {'timer':utils.Timer(), 'epoch':0, 'best_valid':0}
    for epoch in range(start_epoch, args.num_epochs):
        stats['epoch'] = epoch

        #Train
        train(args, train_loader, model, stats)

        # Validate official (dev)
        result = validate_official(args, dev_loader, model, stats, dev_offsets, dev_texts, dev_answers)
        if args.lrshrink > 0:
            _lr = model.lr_step(result[args.valid_metric])
            logger.info('learning rate is %f' % _lr)

        # Save best valid
        if result[args.valid_metric] > stats['best_valid']:
            logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' % 
                (args.valid_metric, result[args.valid_metric],
                    stats['epoch'], model.updates))
            model_save_name = os.path.join(args.save_dir, args.model_name+str(stats['epoch'])+'.pt')
            model.save(model_save_name)
            stats['best_valid'] = result[args.valid_metric]
        logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' % 
                   (args.valid_metric, stats['best_valid'],
                    stats['epoch'], model.updates))

if __name__ == '__main__':
    #Parse cmdline args and setup environment
    parser = options.get_parser('Jarvis Document Reader')
    options.add_dataset_args(parser)
    options.add_trainer_args(parser)
    options.add_optimization_args(parser)
    options.add_model_args(parser)

    args = options.parse_ars_and_arch(parser)
    print(args)

    #Set CUDA
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # if args.cuda:
    #     # device_id = tuple(range(torch.cuda.device_count()))
    #     device_id = 0
    #     torch.cuda.set_device(device_id)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run !
    main(args)



    




