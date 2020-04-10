#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.utils.misc import warn_once
from parlai.core.torch_agent import Batch
from parlai.utils.misc import AttrDict, warn_once
from .modules import  opt_to_kwargs
from .hred_model import  HRED

import torch
import torch.nn as nn

class HREDBatch(AttrDict):
    """
    Batch is a namedtuple containing data being sent to an agent.

    This is the input type of the train_step and eval_step functions.
    Agents can override the batchify function to return an extended namedtuple
    with additional fields if they would like, though we recommend calling the
    parent function to set up these fields as a base.

    :param text_vec:
        bsz x seqlen tensor containing the parsed text data.

    :param text_lengths:
        list of length bsz containing the lengths of the text in same order as
        text_vec; necessary for pack_padded_sequence.

    :param label_vec:
        bsz x seqlen tensor containing the parsed label (one per batch row).

    :param label_lengths:
        list of length bsz containing the lengths of the labels in same order as
        label_vec.

    :param labels:
        list of length bsz containing the selected label for each batch row (some
        datasets have multiple labels per input example).

    :param valid_indices:
        list of length bsz containing the original indices of each example in the
        batch. we use these to map predictions back to their proper row, since e.g.
        we may sort examples by their length or some examples may be invalid.

    :param candidates:
        list of lists of text. outer list has size bsz, inner lists vary in size
        based on the number of candidates for each row in the batch.

    :param candidate_vecs:
        list of lists of tensors. outer list has size bsz, inner lists vary in size
        based on the number of candidates for each row in the batch.

    :param image:
        list of image features in the format specified by the --image-mode arg.

    :param observations:
        the original observations in the batched order
    """

    def __init__(
        self,
        text_1_vec=None,
        text_1_lengths=None,
        text_2_vec=None,
        text_2_lengths=None,
        labels_1_vec_1=None, 
        labels_1_lengths=None, 
        labels_2_vec_2=None, 
        labels_2_lengths=None, 
        valid_indices=None,
        candidates=None,
        candidate_vecs=None,
        image=None,
        observations=None,
        **kwargs,
    ):
        super().__init__(
            text_1_vec=text_1_vec,
            text_1_lengths=text_1_lengths,
            text_2_vec=text_2_vec,
            text_2_lengths=text_2_lengths,
            labels_1_vec_1=labels_1_vec_1,
            labels_1_lengths=labels_1_lengths,
            labels_2_vec_2=labels_2_vec_2,
            labels_2_lengths=labels_2_lengths,
            valid_indices=valid_indices,
            candidates=candidates,
            candidate_vecs=candidate_vecs,
            image=image,
            observations=observations,
            **kwargs,
        )


class HREDAgent(TorchGeneratorAgent):
    """
    Agent which takes an input sequence and produces an output sequence.

    This model supports encoding the input and decoding the output via one of
    several flavors of RNN. It then uses a linear layer (whose weights can
    be shared with the embedding layer) to convert RNN output states into
    output tokens. This model supports greedy decoding, selecting the
    highest probability token at each time step, as well as beam
    search.

    For more information, see the following papers:

    - Neural Machine Translation by Jointly Learning to Align and Translate
      `(Bahdanau et al. 2014) <arxiv.org/abs/1409.0473>`_
    - Sequence to Sequence Learning with Neural Networks
      `(Sutskever et al. 2014) <arxiv.org/abs/1409.3215>`_
    - Effective Approaches to Attention-based Neural Machine Translation
      `(Luong et al. 2015) <arxiv.org/abs/1508.04025>`_
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        agent = argparser.add_argument_group('Seq2Seq Arguments')
        agent.add_argument(
            '-hs',
            '--hiddensize',
            type=int,
            default=128,
            help='size of the hidden layers',
        )
        agent.add_argument(
            '-esz',
            '--embeddingsize',
            type=int,
            default=128,
            help='size of the token embeddings',
        )
        agent.add_argument(
            '-csz',
            '--contextsize',
            type=int,
            default=1000,
            help='size of the context vector',
        )
        agent.add_argument(
            '-nl', '--numlayers', type=int, default=2, help='number of hidden layers'
        )
        agent.add_argument(
            '-dr', '--dropout', type=float, default=0.1, help='dropout rate'
        )
        agent.add_argument(
            '-bi',
            '--bidirectional',
            type='bool',
            default=False,
            help='whether to encode the context with a ' 'bidirectional rnn',
        )
        agent.add_argument(
            '-att',
            '--attention',
            default='none',
            choices=['none', 'concat', 'general', 'dot', 'local'],
            help='Choices: none, concat, general, local. '
            'If set local, also set attention-length. '
            '(see arxiv.org/abs/1508.04025)',
        )
        agent.add_argument(
            '-attl',
            '--attention-length',
            default=48,
            type=int,
            help='Length of local attention.',
        )
        agent.add_argument(
            '--attention-time',
            default='post',
            choices=['pre', 'post'],
            help='Whether to apply attention before or after ' 'decoding.',
        )
        agent.add_argument(
            '-rnn',
            '--rnn-class',
            default='lstm',
            choices=HRED.RNN_OPTS.keys(),
            help='Choose between different types of RNNs.',
        )
        agent.add_argument(
            '-dec',
            '--decoder',
            default='same',
            choices=['same', 'shared'],
            help='Choose between different decoder modules. '
            'Default "same" uses same class as encoder, '
            'while "shared" also uses the same weights. '
            'Note that shared disabled some encoder '
            'options--in particular, bidirectionality.',
        )
        agent.add_argument(
            '-lt',
            '--lookuptable',
            default='unique',
            choices=['unique', 'enc_dec', 'dec_out', 'all'],
            help='The encoder, decoder, and output modules can '
            'share weights, or not. '
            'Unique has independent embeddings for each. '
            'Enc_dec shares the embedding for the encoder '
            'and decoder. '
            'Dec_out shares decoder embedding and output '
            'weights. '
            'All shares all three weights.',
        )
        agent.add_argument(
            '-soft',
            '--numsoftmax',
            default=1,
            type=int,
            help='default 1, if greater then uses mixture of '
            'softmax (see arxiv.org/abs/1711.03953).',
        )
        agent.add_argument(
            '-idr',
            '--input-dropout',
            type=float,
            default=0.0,
            help='Probability of replacing tokens with UNK in training.',
        )

        super(HREDAgent, cls).add_cmdline_args(argparser)
        return agent

    @staticmethod
    def model_version():
        """
        Return current version of this model, counting up from 0.

        Models may not be backwards-compatible with older versions. Version 1 split from
        version 0 on Aug 29, 2018. Version 2 split from version 1 on Nov 13, 2018 To use
        version 0, use --model legacy:seq2seq:0 To use version 1, use --model
        legacy:seq2seq:1 (legacy agent code is located in parlai/agents/legacy_agents).
        """
        return 2

    def __init__(self, opt, shared=None):
        """
        Set up model.
        """
        super().__init__(opt, shared)
        self.id = 'Hred'

    def build_model(self, states=None):
        """
        Initialize model, override to change model setup.
        """
        opt = self.opt
        if not states:
            states = {}

        kwargs = opt_to_kwargs(opt)
        model = HRED(
            len(self.dict),
            opt['contextsize'],
            opt['embeddingsize'],
            opt['hiddensize'],
            padding_idx=self.NULL_IDX,
            start_idx=self.START_IDX,
            end_idx=self.END_IDX,
            unknown_idx=self.dict[self.dict.unk_token],
            longest_label=states.get('longest_label', 1),
            **kwargs,
        )

        if opt.get('dict_tokenizer') == 'bpe' and opt['embedding_type'] != 'random':
            print('skipping preinitialization of embeddings for bpe')
        elif not states and opt['embedding_type'] != 'random':
            # `not states`: only set up embeddings if not loading model
            self._copy_embeddings(model.decoder.lt.weight, opt['embedding_type'])
            if opt['lookuptable'] in ['unique', 'dec_out']:
                # also set encoder lt, since it's not shared
                self._copy_embeddings(
                    model.encoder.lt.weight, opt['embedding_type'], log=False
                )

        if states:
            # set loaded states if applicable
            model.load_state_dict(states['model'])

        if opt['embedding_type'].endswith('fixed'):
            print('HRED: fixing embedding weights.')
            model.decoder.lt.weight.requires_grad = False
            model.encoder.lt.weight.requires_grad = False
            if opt['lookuptable'] in ['dec_out', 'all']:
                model.output.weight.requires_grad = False

        return model

    def build_criterion(self):
        # set up criteria
        if self.opt.get('numsoftmax', 1) > 1:
            return nn.NLLLoss(ignore_index=self.NULL_IDX, reduction='sum')
        else:
            return nn.CrossEntropyLoss(ignore_index=self.NULL_IDX, reduction='sum')

    def _set_label_vec(self, obs, add_start, add_end, truncate, index):
        """
        Set the 'labels_vec' field in the observation.

        Useful to override to change vectorization behavior
        """
        # convert 'labels' or 'eval_labels' into vectors
        if f'labels_{index}' in obs:
            label_type = f'labels_{index}'
        elif 'eval_labels' in obs:
            label_type = f'eval_labels_{index}'
        else:
            label_type = None

        if label_type is None:
            return

        elif label_type + f'_vec_{index}' in obs:
            # check truncation of pre-computed vector
            truncated_vec = self._check_truncate(obs[label_type + f'_vec_{index}'], truncate)
            obs.force_set(label_type + f'_vec_{index}', torch.LongTensor(truncated_vec))
        else:
            # pick one label if there are multiple
            lbls = obs[label_type]
            label = lbls[0] if len(lbls) == 1 else self.random.choice(lbls)
            vec_label = self._vectorize_text(label, add_start, add_end, truncate, False)
            obs[label_type + f'_vec_{index}'] = vec_label
            obs[label_type + f'_choice_{index}'] = label

        return obs


    def _set_text_vec(self, obs, history, truncate, index):
        """
        Override the _set_text_vec fxn. Set the 'text_vec' field in the observation.

        Useful to override to change vectorization behavior
        """

        history_string = history.get_history_str()

        if f'text_{index}' not in obs:
            return obs

        if 'text_vec_{index}' not in obs:
            history_string = history.get_history_str()
            if history_string is None:
                return obs
            obs[f'text_vec_{index}'] =  history.parse(obs[f'text_{index}'])

        # check truncation
        if obs.get(f'text_vec_{index}') is not None:
            truncated_vec = self._check_truncate(obs[f'text_vec_{index}'], truncate, True)
            obs.force_set(f'text_vec_{index}', torch.LongTensor(truncated_vec))
        return obs

    def _set_label_cands_vec(self, obs, add_start, add_end, truncate, index):
        """
        Set the 'label_candidates_vec' field in the observation.

        Useful to override to change vectorization behavior
        """
        if 'label_candidates_vecs' in obs:
            if truncate is not None:
                # check truncation of pre-computed vectors
                vecs = obs[f'label_candidates_vecs_{index}']
                for i, c in enumerate(vecs):
                    vecs[i] = self._check_truncate(c, truncate)
        elif self.rank_candidates and obs.get(f'label_candidates_{index}'):
            obs.force_set(f'label_candidates_{index}', list(obs[f'label_candidates_{index}']))
            obs[f'label_candidates_vecs_{index}'] = [
                self._vectorize_text(c, add_start, add_end, truncate, False)
                for c in obs[f'label_candidates_{index}']
            ]
        return obs

    def vectorize(
                  self,
                  obs,
                  history,
                  add_start=True,
                  add_end=True,
                  text_truncate=None,
                  label_truncate=None):
        """
        override core.torch_agent.vectorize 
        to add an extra utterance to the batch
        """
        
        self._set_text_vec(obs, history, text_truncate, 0)
        self._set_text_vec(obs, history, text_truncate, 1)
        self._set_label_vec(obs, add_start, add_end, label_truncate, 1)
        self._set_label_vec(obs, add_start, add_end, label_truncate, 2)
        self._set_label_cands_vec(obs, add_start, add_end, label_truncate, 1)
        self._set_label_cands_vec(obs, add_start, add_end, label_truncate, 2)
        return obs


    def batchify(self, *args, **kwargs):
        """
        Override batchify options for seq2seq.
        """
        obs_batch = args[0]
        sort = kwargs.get('sort', False)

        if len(obs_batch) == 0:
            return HREDBatch()
        
        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if self.is_valid(ex)]

        if len(valid_obs) == 0:
            return HREDBatch()

        valid_inds, exs = zip(*valid_obs)

        # TEXT
        xs_1, x_1_lens = None, None
        xs_2, x_2_lens = None, None

        if any(ex.get(f'text_vec_0') is not None for ex in exs):
            _xs_1 = [ex.get(f'text_vec_0', self.EMPTY) for ex in exs]
            xs_1, x_1_lens = self._pad_tensor(_xs_1)
            # TODO HRED: Implement this later doesn't matter rn
            if sort:
                sort = False  # now we won't sort on labels
                xs, x_lens, valid_inds, exs = argsort(
                    x_lens, xs, x_lens, valid_inds, exs, descending=True
                )

        if any(ex.get(f'text_vec_1') is not None for ex in exs):
            _xs_2 = [ex.get(f'text_vec_1', self.EMPTY) for ex in exs]
            xs_2, x_2_lens = self._pad_tensor(_xs_2)

        # LABELS
        labels_1_avail = any('labels_1_vec_1' in ex for ex in exs)
        some_labels_1_avail = labels_1_avail or any('eval_labels_1_vec_1' in ex for ex in exs)

        ys_1, y_1_lens, labels_1 = None, None, None
        if some_labels_1_avail:
            field = 'labels_1' if labels_1_avail else 'eval_labels_1'

            label_1_vecs = [ex.get(field + '_1_vec_1', self.EMPTY) for ex in exs]
            labels_1 = [ex.get(field + '_1_choice_1') for ex in exs]
            y_lens_1 = [y.shape[0] for y in label_1_vecs]

            ys_1, y_1_lens = self._pad_tensor(label_1_vecs)

            if sort and xs is None:
                ys, valid_inds, label_vecs, labels, y_lens = argsort(
                    y_lens, ys, valid_inds, label_vecs, labels, y_lens, descending=True
                )

        labels_2_avail = any('labels_2_vec_2' in ex for ex in exs)
        some_labels_2_avail = labels_2_avail or any('eval_labels_2_vec_2' in ex for ex in exs)

        ys_2, y_2_lens, labels_2 = None, None, None
        if some_labels_2_avail:
            field = 'labels_2' if labels_2_avail else 'eval_labels_2'

            label_2_vecs = [ex.get(field + '_2_vec_2', self.EMPTY) for ex in exs]
            labels_2 = [ex.get(field + '_2_choice_2') for ex in exs]
            y_lens_2 = [y.shape[0] for y in label_2_vecs]

            ys_2, y_2_lens = self._pad_tensor(label_2_vecs)

        # LABEL_CANDIDATES
        cands, cand_vecs = None, None
        if any('label_candidates_vecs' in ex for ex in exs):
            cands = [ex.get('label_candidates', None) for ex in exs]
            cand_vecs = [ex.get('label_candidates_vecs', None) for ex in exs]

        # IMAGE
        imgs = None
        if any('image' in ex for ex in exs):
            imgs = [ex.get('image', None) for ex in exs]

        return HREDBatch(
            text_1_vec=xs_1,
            text_1_lengths=x_1_lens,
            text_2_vec=xs_2,
            text_2_lengths=x_2_lens,
            label_1_vec=ys_1,
            label_1_lengths=y_1_lens,
            label_2_vec=ys_2,
            label_2_lengths=y_2_lens,
            labels_1=labels_1,
            labels_2=labels_2,
            valid_indices=valid_inds,
            candidates=cands,
            candidate_vecs=cand_vecs,
            image=imgs,
            observations=exs,
        )

    def self_observe(self): 

        
    def state_dict(self):
        """
        Get the model states for saving.

        Overriden to include longest_label
        """
        states = super().state_dict()
        if hasattr(self.model, 'module'):
            states['longest_label'] = self.model.module.longest_label
        else:
            states['longest_label'] = self.model.longest_label

        return states

    def load(self, path):
        """
        Return opt and model states.
        """
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        # set loaded states if applicable
        self.model.load_state_dict(states['model'])
        if 'longest_label' in states:
            self.model.longest_label = states['longest_label']
        return states

    def is_valid(self, obs):
        contains_empties = []
        for i in range(2):
            contains_empties.append(obs[f'text_vec_{i}'].shape[0] == 0)
        if self.is_training and any(contains_empties):
            warn_once(
                'seq2seq got an empty input sequence (text_vec) during training. '
                'Skipping this example, but you should check your dataset and '
                'preprocessing.'
            )
        elif not self.is_training and contains_empties:
            warn_once(
                'seq2seq got an empty input sequence (text_vec) in an '
                'evaluation example! This may affect your metrics!'
            )
        return [not x for x in contains_empties]

    def compute_loss(self, batch, return_output=False):
        """
        Compute and return the loss for the given batch.

        Easily overridable for customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch['label_1_vec'] is None or batch['label_2_vec'] is None:
            raise ValueError('Cannot compute loss without a label.')
        #TODO HRED: change model forward function to implement HRED 
        # model output: (output_1, output_2)
        print(self.model)
        model_output = self.model(*self._model_input(batch), ys1=batch.label_1_vec, ys2 = batch.label_2_vec)
        scores_1, preds_1, scores_2, preds_2,  *_ = model_output
        score_1_view = scores_1.view(-1, scores_1.size(-1))
        score_1_view = scores_1.view(-1, scores_1.size(-1))
        loss = self.criterion(score_1_view, batch.label_1_vec.view(-1)).sum()
        score_2_view = scores_2.view(-1, scores_1.size(-1))
        score_2_view = scores_2.view(-1, scores_1.size(-1))
        loss += self.criterion(score_2_view, batch.label_2_vec.view(-1)).sum()

        # save loss to metrics
        notnull_1 = batch.label_1_vec.ne(self.NULL_IDX)
        target_tokens_1 = notnull_1.long().sum().item()
        correct = ((batch.label_1_vec == preds_1) * notnull_1).sum().item()
        notnull_2 = batch.label_2_vec.ne(self.NULL_IDX)
        target_tokens_2 = notnull_2.long().sum().item()
        correct += ((batch.label_2_vec == preds_2) * notnull_2).sum().item()
        target_tokens = target_tokens_1 + target_tokens_2

        self.metrics['correct_tokens'] += correct
        self.metrics['nll_loss'] += loss.item()
        self.metrics['num_tokens'] += target_tokens
        loss /= target_tokens  # average loss per token
        if return_output:
            return (loss, model_output)
        else:
            return loss

    def _model_input(self, batch):
        return (batch.text_1_vec, batch.text_2_vec)
