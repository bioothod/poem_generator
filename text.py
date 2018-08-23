import json
import logging
import re
import random

from collections import defaultdict
from collections import deque
from itertools import islice

space_char_id = ' '
eos_char_id = '\n'
pad_char_id = '+'

unknown_word_id = '<unknown>'

def pad(word, max_len, pad_char):
    return word + pad_char*(max_len - len(word))

def word_to_seq(word, char_idx_map):
    return [char_idx_map[l] for l in word]

def seq_to_word(num, char_idx):
    chars = []
    for idx in num:
        char = char_idx[idx]
        if char == pad_char_id:
            break
        chars.append(char)

    return ''.join(chars)


class Poem(object):
    def __init__(self, title, content):
        self.encoded_lines = []
        self.encoded_char_lines = []

        self.title = title
        content = content.lower()
        content = re.sub(r'[^\w\d\n\-\s]+', '', content).lower()
        content = content.split('\n')
        self.content = []
        for l in content:
            l = l.strip()
            if len(l) == 0:
                continue

            self.content.append(l)

    def collect_stats(self, word_dict, char_dict):
        for content in self.content:
            for word in content.split():
                word_dict[word] += 1

                for char in word:
                    char_dict[char] += 1

        return word_dict, char_dict

    def random_line(self):
        return random.choice(self.content)

    def lines(self):
        for line in self.content:
            yield line
        return None

    def push_encoded_line(self, line):
        self.encoded_lines.append(line)

    def push_encoded_char_line(self, line):
        self.encoded_char_lines.append(line)

class Poet(object):
    def __init__(self):
        self.poems = []
        self.words = 0
        self.poet_id = None

        self.word_num = []
        self.char_num = []
        
    def add_text(self, poet_id, title, content):
        self.poet_id = poet_id

        self.poems.append(Poem(title, content))

    def collect_stats(self, word_dict, char_dict):
        for poem in self.poems:
            word_dict, char_dict = poem.collect_stats(word_dict, char_dict)

        return word_dict, char_dict

    def random_word(self):
        line = random.choice(self.poems).random_line()
        return random.choice(line.split())

    def endings(self, num):
        for poem in self.poems:
            for line in poem.lines():
                yield line[-num:]

    def prepare_word_char_batches(self, word_idx_map, char_idx_map):
        unknown_word_num = word_idx_map[unknown_word_id]
        eol_num = char_idx_map[eos_char_id]
        space_num = char_idx_map[space_char_id]

        for poem in self.poems:
            for line in poem.lines():
                words = line.split()

                encoded_line = []
                encoded_char_line = []

                for w in words:
                    widx = word_idx_map.get(w, unknown_word_num)
                    encoded_line.append(widx)

                    seq = word_to_seq(w, char_idx_map)
                    encoded_char_line.append(seq)

                    if w == words[-1]:
                        encoded_char_line.append(eol_num)
                    else:
                        encoded_char_line.append(space_num)
                
                poem.push_encoded_line(encoded_line)
                poem.push_encoded_char_line(encoded_char_line)

def import_poems(path):
    poets = defaultdict(Poet)

    with open(path, 'r') as poems_file:
        poems = json.load(poems_file)
        for poem in poems:
            poet_id = poem['poet_id']
            title = poem['title']
            content = poem['content']

            poets[poet_id].add_text(poet_id, title, content)

    return poets

def prepare_dicts(poets):
    word_dict = defaultdict(int)
    char_dict = defaultdict(int)
    for poet_id, poet in poets.items():
        word_dict, char_dict = poet.collect_stats(word_dict, char_dict)

    char_dict[space_char_id] += 1
    char_dict[eos_char_id] += 1
    char_dict[pad_char_id] += 1

    word_dict[unknown_word_id] += 1

    word_idx = list(word_dict.keys())
    char_idx = list(char_dict.keys())

    char_idx_map = {}
    for idx, char in enumerate(char_idx):
        char_idx_map[char] = idx

    word_idx_map = {}
    for idx, word in enumerate(word_idx):
        word_idx_map[word] = idx

    return word_idx, word_idx_map, char_idx, char_idx_map

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def prepare_rhyme_dataset(poets, cf, char_idx_map):
    target_words = []
    target_words_len = []
    context_words = []
    context_words_len = []

    prev_window = deque(maxlen=cf.rm_num_context)
    prev_window_lens = deque(maxlen=cf.rm_num_context)
    for poet_id, poet in poets.items():
        for win in window(poet.endings(cf.rm_num_last_symbols), cf.rm_num_context+1):
            win = [w.split()[-1] for w in win]

            target_words.append(win[0])
            target_words_len.append(len(win[0]))

            cwords = []
            clens = []

            for c in win[1:]:
                cwords.append(c)
                clens.append(len(c))

            cwords += list(prev_window)
            clens += list(prev_window_lens)

            while len(cwords) < cf.rm_num_context*2:
                random_word = poet.random_word()
                cwords.append(random_word)
                clens.append(len(random_word))

            context_words += cwords
            context_words_len += clens

            prev_window.append(win[0])
            prev_window_lens.append(len(win[0]))

    max_word_len = max(max(target_words_len), max(context_words_len))
    target_words = [pad(w, max_word_len, pad_char_id) for w in target_words]
    context_words = [pad(w, max_word_len, pad_char_id) for w in context_words]

    target_words_numeric = [word_to_seq(w, char_idx_map) for w in target_words]
    context_words_numeric = [word_to_seq(w, char_idx_map) for w in context_words]

    batch_start = 0
    ret_batch_words = []
    ret_batch_lens = []
    num_context = cf.rm_num_context * 2 # forward and backward rhyme words
    while batch_start < len(target_words):
            tb = []
            tbl = []
            cb = []
            cbl = []
            for i in range(cf.batch_size):
                idx = (i + batch_start) % len(target_words)

                tb.append(target_words_numeric[idx])
                tbl.append(target_words_len[idx])

                for j in range(num_context):
                    idx = ((batch_start + i) * num_context + j) % len(context_words)

                    cb.append(context_words_numeric[idx])
                    cbl.append(context_words_len[idx])

            batch_start += cf.batch_size

            batch_words = tb + cb
            batch_lens = tbl + cbl

            ret_batch_words.append(batch_words)
            ret_batch_lens.append(batch_lens)

    return ret_batch_words, ret_batch_lens

