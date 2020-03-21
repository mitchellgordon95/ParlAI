import fileinput
import fire
from collections import Counter

def print_progress(progress, text):
    if progress % 10000 == 0:
        print(f"{text}: {progress}")
    return progress + 1

# Data format: see parlai/core/teachers.py:FbDialogTeacher
def filter_data(input_file, output_file):
    # First pass: trigrams
    # Count the trigrams seen more than 1000 times (Counter + Set?)
    trigrams = Counter()
    progress = 0
    for line in fileinput.input(input_file):
        num, parts = parse_line(line)
        for part in parts:
            words = part.split(' ')
            for trigram in get_trigrams(words):
                trigrams[trigram] += 1
        progress = print_progress(progress, "Lines")

    progress = 0
    common_trigrams = set()
    for trigram, count in trigrams.items():
        if count > 1000:
            common_trigrams.add(trigram)
        progress = print_progress(progress, "Trigrams")

    # Second pass: filtering
    with open(output_file, 'w+') as out_f:
        skip_rest_eps = False
        for line in fileinput.input(input_file):
            num, parts = parse_line(line)

            if skip_rest_eps and num != '1':
                continue
            if skip_rest_eps and num == '1':
                skip_rest_eps = False
            else:
                if should_filter(parts, common_trigrams):
                    # If an episode has an example which should be filtered, skip the rest of the episode
                    skip_rest_eps = True
                    continue
                print(line.strip('\n'), file=out_f)

def get_trigrams(words):
    return zip(words, words[1:], words[2:])

def parse_line(line):
    parts = line.strip("\n").split("\t")
    num = parts[0][0]
    parts[0] = parts[0][2:]
    return num, parts

def should_filter(parts, common_trigrams):
    all_words = []
    for part in parts:
        words = part.split(' ')
        all_words += words
        trigrams = list(get_trigrams(words))
        num_common = len([trigram for trigram in trigrams if trigram in common_trigrams ]) # Lol
        # 90% of trigams seen more than 1000 times
        if len(trigrams) > 0 and num_common / len(trigrams) > 0.9:
            return True
        # word repitition > 3 words
        if len(words) > 3:
            for i in range(3, len(words)):
                if words[i] == words[i-1] == words[i-2] == words[i-3]:
                    return True

    # Source and target > 200 words
    if len(all_words) > 200:
        return True

    return False

# Things we didn't do:
# URL matcher? (Might already be done by a good tokenizer) (Not necessary for open subtitles)
# Contains "[" or "]" (Might already be done by a good tokenizer) (Not necessary for open subtitles)
# does not contain top 50 English words (where are these?) (for language ID) (not necessary for open subtitles, already translated to English)
# Offensive language (phrase matching N/A) (Sub-reddits N/A)

if __name__ == "__main__":
    fire.Fire(filter_data)
