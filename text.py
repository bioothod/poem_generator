import json
import logging
import re
import random

import numpy as np

from collections import defaultdict
from collections import deque
from itertools import islice

# no vowels in these char/word placeholders
space_char_id = ' '
eos_char_id = '\n'
pad_char_id = '+'

unknown_word_id = '<nkn>'
eol_word_id = '<l>'
pad_word_id = pad_char_id*5

vowels = set([l for l in 'aeiouауеэоаыяию'])

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

def vowels_mask(word):
    return [float(c in vowels) for c in word]

class Poem(object):
    def __init__(self, title, content):
        self.encoded_lines = []
        self.encoded_char_lines = []

        self.title = title
        content = content.lower()
        content = re.sub(r'[^\w\d\n\-\s]+', '', content).lower()
        content = re.sub(r'\-', ' ', content)
        content = content.split('\n')
        self.content = []
        for l in content:
            l = l.strip()
            if len(l) == 0:
                continue

            if len(l) > 100:
                continue

            self.content.append(l)

    def collect_word_stats(self, word_dict):
        for content in self.content:
            for word in content.split():
                word_dict[word] += 1

        return word_dict

    def random_line(self):
        return random.choice(self.content)

    def lines(self):
        for line in self.content:
            yield line
        return None

    def push_encoded_line(self, line, line_len):
        self.encoded_lines.append((line, line_len))

    def push_encoded_char_line(self, chars, vowels_mask, word_lens):
        self.encoded_char_lines.append((chars, vowels_mask, word_lens))

    def generate_pentameter_batches(self, batch_size):
        batch_words = []
        batch_lens = []
        batch_masks = []

        flat_words = []
        flat_lens = []
        flat_masks = []

        for char_line, vmask, lens in self.encoded_char_lines:
            flat_words += char_line
            flat_lens += lens
            flat_masks += vmask

        for w in window(flat_words, batch_size):
            batch_words.append(w)
        for l in window(flat_lens, batch_size):
            batch_lens.append(l)
        for m in window(flat_masks, batch_size):
            batch_masks.append(m)

        #logging.info('flat_words: {}, flat_lens: {}, flat_masks: {}'.format(
        #    np.array(flat_words).shape, np.array(flat_lens).shape, np.array(flat_masks).shape))

        #logging.info('batches: {}, lens: {}, masks: {}'.format(np.array(batch_words).shape, np.array(batch_lens).shape, np.array(batch_masks).shape))
        return batch_words, batch_lens, batch_masks

    def generate_lm_batches(self, batch_size, word_idx_map):
        batch_words = []
        batch_lens = []
        batch_chars = []
        batch_clens = []
        batch_vmasks = []
        batch_history = []
        batch_hlens = []
        batch_x = []
        batch_y = []

        unknown_word_enc = word_idx_map[unknown_word_id]
        eol_word_enc = word_idx_map[eol_word_id]
        pad_word_enc = word_idx_map[pad_word_id]

        for _ in range(10):
            words, lens, chars, clens, vmasks, history, hlens, x, y = [], [], [], [], [], [], [], [], []

            for _ in range(batch_size):
                line_idx = random.randint(0, len(self.encoded_lines) - 1)

                word_line, word_len = self.encoded_lines[line_idx]
                char_line, vmask, char_len = self.encoded_char_lines[line_idx]

                seq_idx = random.randint(0, len(word_line) - 1)

                hist = []
                start = 0
                if line_idx > 4:
                    start = line_idx - 4
                for lidx in range(start, line_idx):
                    w, l = self.encoded_lines[lidx]
                    hist += w

                if seq_idx > 0:
                    hist += word_line[:seq_idx-1]

                if len(hist) == 0:
                    hist = [unknown_word_enc]

                words.append(word_line)
                lens.append(word_len)
                chars += char_line
                clens += char_len
                vmasks += vmask
                history.append(hist)
                hlens.append(len(hist))

                x.append(word_line[:-1] + [eol_word_enc])
                y.append(word_line)

            hist_max_len = max(hlens)
            history = [pad(h, hist_max_len, [pad_word_enc]) for h in history]

            batch_words.append(words)
            batch_lens.append(lens)
            batch_chars.append(chars)
            batch_clens.append(clens)
            batch_vmasks.append(vmasks)
            batch_history.append(history)
            batch_hlens.append(hlens)
            batch_x.append(x)
            batch_y.append(y)

        #logging.info('flat_words: {}, batch_x: {}, batch_history: {}'.format(
        #    np.array(flat_words).shape, np.array(batch_x).shape, np.array(batch_history).shape))
        return batch_words, batch_lens, batch_chars, batch_clens, batch_vmasks, batch_history, batch_hlens, batch_x, batch_y

class Poet(object):
    def __init__(self):
        self.poems = []
        self.words = 0
        self.poet_id = None

        self.word_num = []
        self.char_num = []
        
    def add_text(self, poet_id, title, content):
        self.poet_id = poet_id

        if title == "Песни западных славян":
            return

        self.poems.append(Poem(title, content))

    def collect_word_stats(self, word_dict):
        for poem in self.poems:
            word_dict = poem.collect_word_stats(word_dict)

        return word_dict

    def random_word(self):
        line = random.choice(self.poems).random_line()
        return random.choice(line.split())

    def endings(self, num):
        for poem in self.poems:
            for line in poem.lines():
                yield line[-num:]

    def pad_params(self):
        max_words = 0
        max_word_len = 0
        max_len_str = None
        max_word = None
        max_word_line = None

        for poem in self.poems:
            for line in poem.lines():
                words = line.split()
                if len(words) > max_words:
                    max_words = len(words)
                    max_len_str = ' '.join(words)

                for w in words:
                    if len(w) > max_word_len:
                        max_word_len = len(w)
                        max_word = w
                        max_word_line = line

        logging.info('{}: the longest string: "{}", the longest word: "{}", its line: "{}"'.format(self.poet_id, max_len_str, max_word, max_word_line))
        self.max_word_len = max_word_len
        return max_words, max_word_len

    def prepare_word_char_batches(self, word_idx_map, char_idx_map):
        pad_word_enc = word_idx_map[pad_word_id]
        unknown_word_enc = word_idx_map[unknown_word_id]
        eol_word_enc = word_idx_map[eol_word_id]

        eol_char_enc = char_idx_map[eos_char_id]
        space_char_enc = char_idx_map[space_char_id]
        pad_char_enc = char_idx_map[pad_char_id]

        max_words, max_word_len = self.pad_params()
        logging.info('{}: max_words: {}, max_word_len: {}'.format(self.poet_id, max_words, max_word_len))

        encoded_lines_num = 0

        for poem in self.poems:
            for line in poem.lines():
                words = line.split()

                encoded_line = []

                encoded_char_line = []
                vowels_mask_line = []
                word_lens = []

                line_len = len(words)

                words = pad(words, max_words, [pad_word_id]) + [eol_word_id]
                encoded_line = [word_idx_map.get(w, unknown_word_enc) for w in words]

                for w in words:
                    if w == pad_word_id or w == eol_word_id:
                        word_lens.append(0)
                    else:
                        word_lens.append(len(w))

                    wpad = pad(w, max_word_len, pad_char_id)

                    vmask = vowels_mask(wpad)
                    vowels_mask_line.append(vmask)

                    seq = word_to_seq(wpad, char_idx_map)
                    encoded_char_line.append(seq)

                poem.push_encoded_line(encoded_line, line_len)
                poem.push_encoded_char_line(encoded_char_line, vowels_mask_line, word_lens)
                encoded_lines_num += 1

        logging.info('{}: encoded lines: {}'.format(self.poet_id, encoded_lines_num))

    def generate_pentameter_batches(self, batch_size):
        batch_words = []
        batch_lens = []
        batch_masks = []

        for poem in self.poems:
            w, l, m = poem.generate_pentameter_batches(batch_size)

            batch_words += w
            batch_lens += l
            batch_masks += m

        logging.info('{}: poems: {}, pentameter batches: {}'.format(self.poet_id, len(self.poems), len(batch_words)))
        return batch_words, batch_lens, batch_masks

    def generate_language_model_batches(self, batch_size, word_idx_map):
        batch_words, batch_lens, batch_history, batch_hlens, batch_x, batch_y = [], [], [], [], [], []
        batch_chars = []
        batch_clens = []
        batch_vmasks = []

        for poem in self.poems:
            w, bl, c, cl, v, h, hl, x, y = poem.generate_lm_batches(batch_size, word_idx_map)

            batch_words += w
            batch_lens += bl

            batch_chars += c
            batch_clens += cl
            batch_vmasks += v

            batch_history += h
            batch_hlens += hl

            batch_x += x
            batch_y += y

        logging.info('{}: poems: {}, language model batches: {}/{}'.format(self.poet_id, len(self.poems), len(batch_x), len(batch_words)))
        return batch_words, batch_lens, batch_chars, batch_clens, batch_vmasks, batch_history, batch_hlens, batch_x, batch_y

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

    char_dict[space_char_id] += 1
    char_dict[eos_char_id] += 1
    char_dict[pad_char_id] += 1

    word_dict[unknown_word_id] += 1
    word_dict[eol_word_id] += 1
    word_dict[pad_word_id] += 1

    for poet_id, poet in poets.items():
        word_dict = poet.collect_word_stats(word_dict)

    for word in word_dict.keys():
        for char in word:
            char_dict[char] += 1

    logging.info('poets: {}, words: {}, chars: {}'.format(len(poets), len(word_dict), len(char_dict)))

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
    result = list(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + [elem]
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

    logging.info('rhyme dataset has been created, batches generated: {}'.format(len(ret_batch_words)))
    return ret_batch_words, ret_batch_lens
