#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Basic example which iterates through the tasks specified and prints them out. Used for
verification of data loading and iteration.

For example, to make sure that bAbI task 1 (1k exs) loads one can run and to
see a few of them:

Examples
--------

.. code-block:: shell

  python display_data.py -t babi:task1k:1
"""

from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.strings import colorize

import random
import sys


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Display data from a task')
    # Get command line arguments
    parser.add_argument('-n', '-ne', '--num-examples', type=int, default=10)
    parser.add_argument('-mdl', '--max-display-len', type=int, default=1000)
    parser.add_argument('--display-ignore-fields', type=str, default='agent_reply')
    parser.add_argument(
        '-v',
        '--display-verbose',
        default=False,
        action='store_true',
        help='If false, simple converational view, does not show other message fields.',
    )

    parser.set_defaults(datatype='train:stream')
    return parser


def display_data(opt):
    # create repeat label agent and assign it to the specified task
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)

    # Show some example dialogs.
    episode_done = True
    count = 0
    for _ in range(world.num_examples()):
        world.parley()

        # NOTE: If you want to look at the data from here rather than calling
        # world.display() you could access world.acts[0] directly
        # print(world.display() + '\n~~')

        # First act is special cause it has the persona
        text = world.acts[0]['text'].split('\n')
        if episode_done:
            assert text
            print('[your persona]')
            for idx, line in enumerate(text):
                if line.startswith('your persona: '):
                    print(line[14:])
                else:
                    break

            assert idx == len(text) - 1
            print()
            print(text[-1])
            episode_done = False
        else:
            assert len(text) == 1
            print(text[0])

        print(world.acts[0]['labels'][0])

        if world.acts[0]['episode_done']:
            count += 1
            if count % 100 == 0:
                print(count, file=sys.stderr)
            episode_done = True
            print()
            print()

    try:
        # print dataset size if available
        print(
            '[ loaded {} episodes with a total of {} examples ]'.format(
                world.num_episodes(), world.num_examples()
            )
        )
    except Exception:
        pass


if __name__ == '__main__':
    random.seed(42)

    # Get command line arguments
    parser = setup_args()
    opt = parser.parse_args()
    display_data(opt)
