#!/usr/bin/enc python3
# Document Reader of Jarvis for SQuAD .
# Wirtten by Turing Lee in ColorfulClouds .

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np 
import logging 
import copy

from torch.autograd import Variable
import models

# from allennlp.modules.elmo import Elmo, batch_to_ids

logger = logging.getLogger(__name__)

class DocReader(object):
    """
    High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # Initialization

    def __init__(self, args, word_dict, char_dict, feature_dict, word_sort_by_freq, 
                 state_dict = None, normalize = True):
        # Book-Keeping.
        self.args = args
        self.word_dict = word_dict
        self.char_dict = char_dict
        self.word_sort_by_freq = word_sort_by_freq
        self.args.vocab_size = len(word_dict)
        self.feature_dict = feature_dict
        self.args.num_features = len(feature_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False

        # Building network. If normalize is false, scores are not normalized
        # 0-1 per paragraph (no softmax).
        self.network = getattr(models, args.network_file).build_model(args, word_dict, char_dict, normalize)
        
        if args.embedding_file:
            self.load_embedding(word_dict, args.embedding_file)
        if args.tune_partial > 0 :
            if args.fixed_embedding:
                logger.warning("tune_partial > 0 while fixed_embedding is True, set fixed_embedding to Ture.")
                self.args.fixed_embedding = False
            top_word_list = word_sort_by_freq[:args.tune_partial]
            self.tune_embedding(top_word_list)

        # if args.add_elmo:
        #     self.elmo = Elmo(args.elmo_options_file, args.elmo_weight_file, 1, dropout = 0, is_update = True).cuda()

        # Load saved state
        if state_dict:
            # Load buffer separately
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')
                self.network.load_state_dict(state_dict)
                self.network.register_buffer('fixed_embedding', fixed_embedding)
            else:
                self.network.load_state_dict(state_dict)

    def init_optimizer(self, state_dict = None):
        """
        Initialize on aoptimizer for the free parameters of the network.

        Args:
            state_dict: network parameters.
        """
        if self.args.fixed_embedding:
            for p in self.network.encoding_layer.embedding.parameters():
                p.requires_grad = False
        
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                momentum = self.args.momentum,
                weight_decay = self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(parameters, self.args.learning_rate,
                betas = (0.9, 0.9), eps = 1e-08, weight_decay = self.args.weight_decay)
        elif self.args.optimizer == 'adadelta':
            self.optimizer = optim.Adadelta(parameters, self.args.learning_rate,
                rho = 0.9, eps = 1e-06, weight_decay = self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay = self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.args.optimizer)
        
        if self.args.lrshrink > 0:
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode = 'max', patience = 2, factor = self.args.lrshrink)

    def load_embedding(self, word_dict, embedding_file):
        """
        Loading pretrained embeddings for a given list of word, if they exist.
        Args:
            words: iterable of tokens. Only those that are indexed in the 
                dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in word_dict.tokens()}
        logger.info("Loading pre-trained embeddingd for %d words from %s" % 
            (len(words), embedding_file))
        embedding = self.network.encoding_layer.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embendding)
        vec_counts = {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning('WARN: Duplicate embedding found for %s' % w)
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)
        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' % 
            (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def tune_embedding(self, top_word_list):
        """
        Unfix the embedding of the top N words.

        Shuffles the N specified words to the front of the dictionary, and saves
        the original vectors of the other N + 1:vocab words in a fixed buffer.
        """

        embeddings = self.network.encoding_layer.embedding.weight.data

        for i, swap_word in enumerate(top_word_list, self.word_dict.START-1):
            cur_word = self.word_dict[i]
            cur_emb = embeddings[i].clone()
            old_idx = self.word_dict[swap_word]

            embeddings[i].copy_(embeddings[old_idx])
            embeddings[old_idx].copy_(cur_emb)

            self.word_dict[cur_word] = old_idx
            self.word_dict[old_idx] = cur_word
            self.word_dict[swap_word] = i
            self.word_dict[i] = swap_word

        # Save the original, fixed embeddings
        self.network.register_buffer(
            'fixed_embedding', embeddings[i + 1:].clone()
        )

    def reset_parameters(self):
        """
        Reset any partially parameters to original states.
        """
        if self.args.tune_partial > 0:
            offset = self.args.tune_partial + self.word_dict.START - 1

            embeddings = self.network.encoding_layer.embedding.weight.data
            fixed_embedding = self.network.fixed_embedding

            if offset < embeddings.size(0):
                embeddings[offset:] = fixed_embedding



    #--------------------------------------------------------------------------------
    # Training
    #--------------------------------------------------------------------------------
    
    def update(self, ex, toBePrint = False):
        """
        Forward a batch of examples; step the optimizer to update weights.
        """
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        # Train to GPU
        if self.use_cuda:

            # context_ids = batch_to_ids(ex[8]).cuda()
            # query_ids = batch_to_ids(ex[9]).cuda()

            # ex[8] = self.elmo(context_ids)
            # ex[9] = self.elmo(query_ids)

            # for i in range(len(ex[8]['activations'])):
            #     ex[8]['activations'][i] = Variable(torch.from_numpy(np.array(ex[8]['activations'][i]))).cuda()
            # ex[8]['mask'].cuda()

            # for i in range(len(ex[9]['activations'])):
            #     ex[9]['activations'][i] = Variable(torch.from_numpy(np.array(ex[9]['activations'][i]))).cuda()
            # ex[9]['mask'].cuda()

            inputs = [e if e is None else Variable(e.cuda(async = True)) 
                      for e in ex[:8]]
            inputs.append(ex[8])
            inputs.append(ex[9])
            target_s = Variable(ex[10].cuda(async = True))
            target_e = Variable(ex[11].cuda(async = True))
        else:
            inputs = [e if e is None else Variable(e) for e in ex[:8]]
            inputs.append(ex[8])
            inputs.append(ex[9])
            target_s = Variable(ex[10])
            target_e = Variable(ex[11])

        # Run forward
        pred_s, pred_e = self.network(*inputs) # The asterisk means unpack.

        # Compute loss and accuracies
        if self.args.loss_fn == 'nll':
            _loss_fn = nn.NLLLoss()        
        elif self.args.loss_fn == 'cel':
            _loss_fn = nn.CrossEntropyLoss()
        
        loss = _loss_fn(pred_s, target_s) + _loss_fn(pred_e, target_e)

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # print(self.network.encoding_layer.embedding_char.weight.grad)
        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                     self.args.grad_clipping) 

        # Update parameters
        self.optimizer.step()
        self.reset_parameters()
        self.updates += 1

        return loss.data[0], ex[0].size(0)

    #--------------------------------------------------------------------------------
    # Prediction
    #--------------------------------------------------------------------------------

    def predict(self, ex, candidates = None, top_n = 1, async_pool = None):
        """
        Forward a batch of examples only to get predictions.
        Args:
            ex: the batch
            candidates: batch * variable length list of string answer options.
                The model will only consider exact spans contained in this list.
            top_n: Number of predictions to return per batch element.
            async_pool: If provided, non-gpu post-precessing will be offloaded 
                to this CPU process pool.
        Output:
            pred_s: batch * top_n predicted start indices
            pred_e: batch * top_n predicted end indices
            pred_score: batch * top_n prediction scores

        If async_pool is given, these will be AsyncResult handles.
        """
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        # context_ids = batch_to_ids(ex[8]).cuda()
        # query_ids = batch_to_ids(ex[9]).cuda()

        # ex[8] = self.elmo(context_ids)
        # ex[9] = self.elmo(query_ids)

        # for i in range(len(ex[8]['activations'])):
        #     ex[8]['activations'][i] = Variable(torch.from_numpy(np.array(ex[8]['activations'][i]))).cuda()
        # ex[8]['mask'].cuda()

        # for i in range(len(ex[9]['activations'])):
        #     ex[9]['activations'][i] = Variable(torch.from_numpy(np.array(ex[9]['activations'][i]))).cuda()
        # ex[9]['mask'].cuda()
        
        if self.use_cuda:
            inputs = [e if e is None else
                      Variable(e.cuda(async = True), volatile = True)
                      for e in ex[:8]]
            inputs.append(ex[8])
            inputs.append(ex[9])
        else:
            inputs = [e if e is None else Variable(e, volatile = True)
                      for e in ex[:8]]
            inputs.append(ex[8])
            inputs.append(ex[9])

        # Run forward
        score_s, score_e = self.network(*inputs)

        #Decode predictions
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()

        if candidates:
            args = (score_s, score_e, candidates, top_n, self.args.max_len)
            if async_pool:
                return async_pool.apply_async(self.decode_candidates, args)
            else:
                return self.decode_candidates(*args)
        else:
            _args = (score_s, score_e, top_n, self.args.max_len)
            if async_pool:
                return async_pool.apply_async(self.decode, _args)
            else:
                return self.decode(*_args)

    def lr_step(self, val_em):
        """
        learing rate decay.
        """
        self.lr_scheduler.step(val_em)
        return self.optimizer.param_groups[0]['lr']

    @staticmethod
    def decode_candidates(score_s, score_e, candidates, top_n = 1, max_len = None):
        """
        Take argmax of constrained score_s * score_e. Except only conside
        spans that are in the candidates list.
        """
        pred_s = []
        pred_e = []
        pred_score = []

        for i in range(score_s.size(0)):
            # Extract original tokens stored with candidates
            tokens = candidates[i]['input']
            cands = candidates[i]['cands']

            # score all valid candidates found in text.
            # Brute force get all ngrams and compare against the candidate list.    
            max_len = max_len or len(tokens)
            scores, s_idx, e_idx = [], [], []
            for s, e in tokens.ngrams(n=max_len, as_string = False):
                span = tokens.slice(s, e).untokenize()
                if span in cands or span.lower() in cands:
                    # Match! Record its score.
                    score.append(score_s[i][s] * score_e[i][e-1])
                    s_idx.append(s)
                    e_idx.append( e - 1)
            if len(scores) == 0:
                # No candidates present
                pred_s.append([])
                pred_e.append([])
                pred_score.append([])
            else:
                # Rank found candidates
                scores = np.array(scores)
                s_idx = np.array(s_idx)
                e_idx = np.array(e_idx)

                idx_sort = np.argsort(-scores)[0:top_n]
                pred_s.append(s_idx[idx_sort])
                pred_e.append(e_idx[idx_sort])
                pred_score.append(scores[idx_sort])

        return pred_s, pred_e, pred_score

    @staticmethod
    def decode(score_s, score_e, top_n = 1, max_len = None):
        """
        Take argmax of constrained score_s * score_e.

        Args:
            score_s: independent  start predictions
            score_e: independent  end predictions
            top_n : number of top scored pairs to take
            max_len: max span length to consider
        """
        pred_s = []
        pred_e = []
        pred_score = []
        max_len = max_len or score_s.size(1)

        for i in range(score_s.size(0)):
            # Output product of scores to get full p_s * p_e matrix
            scores = torch.ger(score_s[i], score_e[i])

            # Zero out negative length and over-length span scores
            scores.triu_().tril_(max_len - 1)

            # Take argmax or top n
            scores = scores.numpy()
            scores_flat = scores.flatten()

            if top_n == 1:
                idx_sort = [np.argmax(scores_flat)]
            elif len(scores_flat) < top_n:
                idx_sort = np.argsort(-scores_flat)
            else:
                idx = np.argpartition(-scores_flat, top_n)[0:top_n]
                idx_sort = idx[np.argsort(-scores_flat[idx])]
            s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)
            pred_s.append(s_idx)
            pred_e.append(e_idx)
            pred_score.append(scores_flat[idx_sort])
        return pred_s, pred_e, pred_score

    #--------------------------------------------------------------------------------
    # Saving and loading
    #--------------------------------------------------------------------------------

    def save(self, filename):
        state_dict = copy.copy(self.network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'args': self.args
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saveing failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        params = {
            'state_dict': self.network.state_dict(),
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict()
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args = None, normalize = True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        args = saved_params['args']

        if new_args:
            args = override_model_args(args,new_args)

        return DocReader(args, word_dict, feature_dict, state_dict, normalize)

    @staticmethod
    def load_checkpoint(filename, normalize = True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        optimizer = saved_params['optimizer']
        args = saved_params['args']
        model = DocReader(args, word_dict, feature_dict, state_dict, normalize)
        model.init_optimizer(optimizer)

        return model, epoch


    #--------------------------------------------------------------------------------
    # Runtime
    #--------------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """
        Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)

