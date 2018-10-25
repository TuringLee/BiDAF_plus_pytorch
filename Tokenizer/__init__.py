#!/usr/bin/env python3
# Wirtten by Turing Lee in ColorfulClouds .

import os
from .Spacy_tokenizer import SpacyTokenizer

def get_class(name):
    if name == 'spacy':
        return SpacyTokenizer
    else:
        raise RuntimeError('Only support Spacy tokenizer now , and will add other tokenizer soon .')

def get_annotators_for_args(args):
    annotators = set()
    if args.use_pos:
        annotators.add('pos')
    if args.use_lemma:
        annotators.add('lemma')
    if args.use_ner:
        annotators.add('ner')
    return annotators

def get_annotators_for_model(model):
    return get_annotators_for_args(model.args)

