#!/usr/bin/env python3
# Wirtten by Turing Lee in ColorfulClouds .

class Tokenizer(object):
    """
    Base tokenizer class .
    Tokenizer implement tokenize, which should return a Tokens class. 
    """
    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()

class Tokens(object):
    """A class to represent a list of tokenized text. """
    TEXT = 0
    TEXT_RAW = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        return len(self.data)

    def slice(self, i=None, j=None):
        # Return a view of the list of tokens from [i, j].
        new_tokens= copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self):
        # Return the original text (with whitespace reinserted).
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        # Return a list of the text of each token

        # Args : 
        #     uncased: lower cases text

        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offset(self):
        # Return a list of [start, end] character offsets of each token.
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        # Return a list of part-of-speech tags of each token.
        # Return None if this annotation was not included.
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        # Return a list of the lemmatized text of each token.
        # Return None if this annotation was not included.
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        # Return a list of named-entity-recognition tags of each token. 
        # Return None if this annotation was not included.
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        # Returns a list of all ngrams from length 1 to n.

        # Args:
        #     n: upper limit of ngram length
        #     uncased: lower cases text
        #     filter_fn: user function that takes in an ngram list and returns
        #       True or False to keep or not keep the ngram
        #     as_string: return the ngram as a string vs list

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s,e+1)
                    for s in range(len(words))
                    for e in range(s, min(s+n, len(words)))
                    if not _skip(words[s:e+1])]

        # Concatenate into string
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s: e])) for (s, e) in ngrams]
        return ngrams

    def entity_group(self):
        # Group consecutive entity tokens with the same NER tag.
        entities = self.entities()
        if not entities:
             return None
        non_ent = self.opts.get('non_ent', '0')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            if ner_tag != non_ent:
            # chmop the sequence
                start = idx 
                while(idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups

