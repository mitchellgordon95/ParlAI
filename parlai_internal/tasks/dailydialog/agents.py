#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Daily Dialog https://arxiv.org/abs/1710.03957.

Original data is copyright by the owners of the paper, and free for use in research.

Every conversation contains entries with special fields (see the paper):

- emotion
- act_type
- topic

This teacher plays both sides of the conversation, once acting as Speaker 1, and
once acting as Speaker 2.
"""

import os
import json
from parlai.core.teachers import FixedDialogTeacher
from parlai.core.message import Message
from .build import build


START_ENTRY = {'text': '__SILENCE__', 'emotion': 'no_emotion', 'act': 'no_act'}


class Convai2Teacher(FixedDialogTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt
        if shared:
            self.data = shared['data']
        else:
            build(opt)
            fold = opt.get('datatype', 'train').split(':')[0]
            self._setup_data(fold)

        self.num_exs = sum(len(d['dialogue']) for d in self.data)

        # we learn from both sides of every conversation
        self.num_eps = 2 * len(self.data)
        self.reset()

    def num_episodes(self):
        return self.num_eps

    def num_examples(self):
        return self.num_exs

    def _setup_data(self, fold):
        self.data = []
        fpath = os.path.join(self.opt['datapath'], 'dailydialog', fold + '.json')
        with open(fpath) as f:
            for line in f:
                self.data.append(json.loads(line))

    def get(self, episode_idx, entry_idx=0):
        # Sometimes we're speaker 1 and sometimes we're speaker 2
        speaker_id = episode_idx % 2
        full_eps = self.data[episode_idx // 2]

        entries = [START_ENTRY] + full_eps['dialogue']
        their_turn = entries[speaker_id + 2 * entry_idx]
        my_turn = entries[1 + speaker_id + 2 * entry_idx]
        try:
            their_turn_2 = entries[2 + speaker_id + 2 * entry_idx]
        except IndexError:
            their_turn_2 = {'emotion': 'no_emotion', 'act': 'directive', 'text': "__SILENCE__"}
        episode_done = 2 * entry_idx + speaker_id + 2 >= len(full_eps['dialogue']) - 1

        action = {
            'topic': full_eps['topic'],
            'text_0': their_turn['text'],
            'text_1': my_turn['text'],
            'emotion': their_turn['emotion'],
            'act_type': their_turn['act'],
            'labels_1': [my_turn['text']],
            'labels_2': [their_turn_2['text']],
            'episode_done': episode_done,
        }
        return action

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared

    def act(self):
        """
        Send new dialog message.
        """
        if not hasattr(self, 'epochDone'):
            # reset if haven't yet
            self.reset()

        # get next example, action is episode_done dict if already out of exs
        action, self.epochDone = self.next_example()
        # TODO: all teachers should eventually create messages
        # while setting up the data, so this won't be necessary
        action = Message(action)
        action.force_set('id', self.getID())

        # remember correct answer if available
        self.lastY = action.get('labels_1', action.get('eval_labels_1', None))
        if (
            not self.datatype.startswith('train') or 'evalmode' in self.datatype
        ) and 'labels' in action:
            # move labels to eval field so not used for training
            # but this way the model can use the labels for perplexity or loss
            action = action.copy()
            labels = action.pop('labels')
            if not self.opt.get('hide_labels', False):
                action['eval_labels'] = labels

        return action


class NoStartTeacher(Convai2Teacher):
    """
    Same as default teacher, but it doesn't contain __SILENCE__ entries.

    If we are the first speaker, then the first utterance is skipped.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        # Calculate the correct number of examples.
        self.num_exs = sum(len(d['dialogue']) - 1 for d in self.data)

        # Store all episodes separately, so we can deal with 2-turn dialogs.
        self.all_eps = self.data + [d for d in self.data if len(d['dialogue']) > 2]
        self.num_eps = len(self.all_eps)

    def get(self, episode_idx, entry_idx=0):
        full_eps = self.all_eps[episode_idx]
        entries = full_eps['dialogue']

        # Sometimes we're speaker 1 and sometimes we're speaker 2.
        # We can't be speaker 1 if dialog has only 2 turns.
        speaker_id = int(episode_idx >= len(self.data))

        their_turn = entries[speaker_id + 2 * entry_idx]
        my_turn = entries[1 + speaker_id + 2 * entry_idx]
        their_turn_2 = entries[1 + speaker_id + 2 * entry_idx]
        episode_done = 2 * entry_idx + speaker_id + 1 >= len(entries) - 2

        action = {
            'topic': full_eps['topic'],
            'text': their_turn['text'],
            'emotion': their_turn['emotion'],
            'act_type': their_turn['act'],
            'labels': [my_turn['text']],
            'episode_done': episode_done,
        }
        return action


class DefaultTeacher(Convai2Teacher):
    pass
