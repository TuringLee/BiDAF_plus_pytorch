#!/usr/bin/enc python3
# vector script of Jarvis for SQuAD .
# Wirtten by Turing Lee in ColorfulClouds .

from collections import Counter
import torch

def vectorize(ex, model, single_answer=False):
    """
    Torchify a single example.
    """
    args = model.args
    word_dict = model.word_dict
    feature_dict = model.feature_dict

    #Index words
    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    question = torch.LongTensor([word_dict[w] for w in ex['question']])

    # # Add character embedding
    # document = torch.LongTensor([w[:16] if len(w) > 16 else w for w in ex['document']])
    # question = torch.LongTensor([w[:16] if len(w) > 16 else w for w in ex['question']])
    
    # Create extra features vectors
    if len(feature_dict) > 0:
        features = torch.zeros(len(ex['document']), len(feature_dict))
    else:
        feature = None

    # f_{exact_match}
    if args.use_in_question:
        q_words_cased = {w for w in ex['question']}
        q_words_uncased = {w.lower() for w in ex['question']}
        q_lemma = {w for w in ex['qlemma']} if args.use_lemma else None
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                features[i][feature_dict['in_question']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                features[i][feature_dict['in_question_uncased']] = 1.0
            if q_lemma and ex['lemma'][i] in q_lemma:
                features[i][feature_dict['in_question_lemma']] = 1.0

    # f_{token} (POS)
    if args.use_pos:
        for i, w in enumerate(ex['pos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (NER)
    if args.use_ner:
        for i, w in enumerate(ex['ner']):
            f = 'ner=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (TF)
    if args.use_tf:
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    # Maybe return without target
    if 'answers' not in ex:
        return document, features, question, ex['id']

    # or with target(s) (might still be empty if answers is empty)
    if single_answer:
        assert(len(ex['answers']) > 0)
        start = torch.LongTensor(1).fill_(ex['answers'][0][0])
        end = torch.LongTensor(1).fill_(ex['answers'][0][1])
    else:
        start = [a[0] for a in ex['answers']]
        end = [a[1] for a in ex['answers']]

    return document, features, question, start, end, ex['id']


def batchify(batch):
    """
    Gather a batch of individual examples into one batch.
    """
    NUM_INPUTS = 3
    NUM_TARGETS = 2
    NUM_EXTRA = 1

    ids = [ex[-1] for ex in batch]
    docs = [ex[0] for ex in batch]
    features = [ex[1] for ex in batch]
    questions = [ex[2] for ex in batch]

    # Batch documents and features
    max_length = max([d.size(0) for d in docs])
    context = torch.LongTensor(len(docs), max_length).zero_()
    context_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
    if features[0] is None:
        context_feature = None
    else:
        context_feature = torch.zeros(len(docs), max_length, features[0].size(1))
    for i, d in enumerate(docs):
        context[i,:d.size(0)].copy_(d)
        context_mask[i, :d.size(0)].fill_(0)
        if context_feature is not None:
            context_feature[i, :d.size(0)].copy_(features[i])

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    query = torch.LongTensor(len(questions), max_length).zero_()
    query_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    for i, q in enumerate(questions):
        query[i, :q.size(0)].copy_(q)
        query_mask[i, :q.size(0)].fill_(0)

    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        return context, context_feature, context_mask, query, query_mask, ids
        # return context, context_feature, query, ids
    
    elif len(batch[0]) == NUM_INPUTS + NUM_TARGETS + NUM_EXTRA:
        # Otherwise add targets
        if torch.is_tensor(batch[0][3]):
            target_s = torch.cat([ex[3] for ex in batch])
            target_e = torch.cat([ex[4] for ex in batch])
        else:
            target_s = [ex[3] for ex in batch]
            target_e = [ex[4] for ex in batch]
    else:
        raise RuntimeError('Incorrect number of inputs per example.')
    return context, context_feature, context_mask, query, query_mask, target_s, target_e, ids



                