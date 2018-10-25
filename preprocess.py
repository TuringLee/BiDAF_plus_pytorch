#!/usr/bin/enc python3
# Preprocess script of Jarvis for SQuAD .
# Wirtten by Turing Lee in ColorfulClouds .

import argparse
import json
import time
import os
import sys

import Tokenizer
from multiprocessing import Pool
from multiprocessing.util import Finalize
from functools import partial

from allennlp.modules.elmo import Elmo, batch_to_ids

# Tokenize + annotate.

TOK = None

def init(tokenizer_class, options):
    global TOK
    TOK = tokenizer_class(**options)
    Finalize(TOK, TOK.shutdown, exitpriority=100)

def tokenize(text):
    # Global the global process tokenizer on the input text.input
    global TOK
    tokens = TOK.tokenize(text)
    output = {
            'words': tokens.words(),
            'offsets': tokens.offset(),
            'pos': tokens.pos(),
            'lemma': tokens.lemmas(),
            'ner': tokens.entities(),
    }
    return output

# Process dataset examples. 

def load_dataset(path):
    # load json file and store fields separately. 
    with open(path) as f:
        data = json.load(f)['data']
    output = {'qids': [], 'questions': [], 'answers': [],
              'contexts':[], 'qid2cid': []}
    for article in data:
        for paragraph in article['paragraphs']:
            # if len(paragraph['context'].split()) > 400:
            #     continue
            output['contexts'].append(paragraph['context'])
            for qa in paragraph['qas']:
                if 'answers' not in qa:
                    continue
                output['qids'].append(qa['id'])
                output['questions'].append(qa['question'])
                output['qid2cid'].append(len(output['contexts']) - 1)
                output['answers'].append(qa['answers'])
        break
    return output

def find_answer(offsets, begin_offset, end_offset):
    # Match token offsets with the char begin/end offsets of the answer. 
    start = [i for i, tok in enumerate(offsets) if tok[0] == begin_offset]
    end = [i for i, tok in enumerate(offsets) if tok[1] == end_offset]
    assert(len(start) <= 1)
    assert(len(end) <= 1)
    if len(start) == 1 and len(end) == 1:
        return start[0], end[0]

def process_dataset(data, tokenizer, workers=None):
    # Iterate processing (tokenize, parse, etc) dataset multithreaded.
    tokenizer_class = Tokenizer.get_class(tokenizer)
    make_pool = partial(Pool, workers, initializer = init)
    q_workers = make_pool(initargs = (tokenizer_class, {'annotators': {'lemma', 'pos', 'ner'}}))
    q_tokens = q_workers.map(tokenize, data['questions'])
    q_workers.close()
    q_workers.join()

    c_workers = make_pool(initargs = (tokenizer_class, {'annotators':{'lemma', 'pos','ner'}}))
    c_tokens = c_workers.map(tokenize, data['contexts'])
    c_workers.close()
    c_workers.join()

    if len(data['answers']) == 0:
        raise RuntimeError('There is no question with answer .')

    for idx in range(len(data['qids'])):
        question = q_tokens[idx]['words']
        qlemma = q_tokens[idx]['lemma']
        qpos = q_tokens[idx]['pos']
        qner = q_tokens[idx]['ner']
        document = c_tokens[data['qid2cid'][idx]]['words']
        offsets = c_tokens[data['qid2cid'][idx]]['offsets']
        lemma = c_tokens[data['qid2cid'][idx]]['lemma']
        pos = c_tokens[data['qid2cid'][idx]]['pos']
        ner = c_tokens[data['qid2cid'][idx]]['ner']

        q_elmo = {}
        q_elmo['activations'] = []
        q_char_ids = batch_to_ids([question]).cuda()
        q_ret = elmo(q_char_ids)
        for i in range(len(q_ret['activations'])):
            q_elmo['activations'].append(q_ret['activations'][i].cpu().data.numpy().tolist())
        q_elmo['mask'] = q_ret['mask'].cpu().data.numpy().tolist()

        d_elmo = {}
        d_elmo['activations'] = []
        d_char_ids = batch_to_ids([document]).cuda()
        d_ret = elmo(d_char_ids)
        for i in range(len(d_ret['activations'])):
            d_elmo['activations'].append(d_ret['activations'][i].cpu().data.numpy().tolist())
        d_elmo['mask'] = d_ret['mask'].cpu().data.numpy().tolist()

        ans_tokens = []
        for ans in data['answers'][idx]:
            found = find_answer(offsets,
                                ans['answer_start'],
                                ans['answer_start'] + len(ans['text']))
            if found:
                ans_tokens.append(found)
        yield {
            'id':data['qids'][idx],
            'question': question,
            'document': document,
            'offsets': offsets,
            'answers': ans_tokens,
            'qlemma': qlemma,
            'lemma': lemma,
            'qpos': qpos,
            'pos': pos,
            'qner': qner,
            'ner': ner,
            'q_elmo': q_elmo,
            'd_elmo': d_elmo,
        }

#Conmandline options

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Path to SQuAD data directory')
parser.add_argument('out_dir', type=str, help='Path to output file dir')
parser.add_argument('--split', type=str, help='Filename for train/dev split',
                    default='train-v1.1')
parser.add_argument('--elmo_options_file', type = str)
parser.add_argument('--elmo_weight_file', type = str)
parser.add_argument('--workers', type=int, default=None)
parser.add_argument('--tokenizer', type=str, default='spacy')

args = parser.parse_args()

elmo = Elmo(args.elmo_options_file, args.elmo_weight_file, 1, is_update = True, dropout = 0).cuda()

start_time = time.time()
in_file = os.path.join(args.data_dir, args.split+'.json')
print('Loading dataset %s' % in_file, file=sys.stderr)
dataset = load_dataset(in_file)

out_file = os.path.join(
    args.out_dir, '%s-processed-%s.txt' % (args.split, args.tokenizer)
    )
print('Will write to file %s' % out_file, file=sys.stderr)
with open(out_file, 'w') as f:
    for ex in process_dataset(dataset, args.tokenizer, args.workers):
        f.write(json.dumps(ex) + '\n')
print('Total time : %4f (s)' % (time.time()-start_time))




