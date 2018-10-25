#!/usr/bin/env python3
# Wirtten by Turing Lee in ColorfulClouds .

import spacy
import copy
from .tokenizer import Tokens, Tokenizer

class SpacyTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            model: spaCy model to use (either path, or keyword like 'en').like
        """
        model = kwargs.get('model', 'en')
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        nlp_kwargs = {'parser': False}
        if not {'lemma', 'pos'} & self.annotators:
            nlp_kwargs['tagger'] = False
        if not {'ner'} & self.annotators:
            nlp_kwargs['entity'] = False
        self.nlp = spacy.load(model, **nlp_kwargs)

    def tokenize(self, text):
        # We don't treat new lines as tokens
        clean_text = text.replace('\n', ' ')
        # Segment text into words, punctuations marks etc.
        tokens = self.nlp.tokenizer(clean_text)
        if {'lemma','pos'} & self.annotators:
            # Annotate part-of-speech tags and lemma on documents.
            self.nlp.tagger(tokens)
        if {'ner'} & self.annotators:
            # Annotate named entities on documents.
            self.nlp.entity(tokens)

        data = []
        for i in range(len(tokens)):
            # Get words with whitespace follow the word
            start_ws = tokens[i].idx
            if i + 1 < len(tokens):
                end_ws = tokens[i+1].idx
            else:
                end_ws = tokens[i].idx + len(tokens[i].text)

            data.append((
                tokens[i].text,
                text[start_ws: end_ws],
                (tokens[i].idx, tokens[i].idx+len(tokens[i].text)),
                tokens[i].tag_,
                tokens[i].lemma_,
                tokens[i].ent_type_,
                ))
        # Set special option for non-entity tag: '' vs '0' in spacy
        return Tokens(data, self.annotators, opts={'non_ent': ''})