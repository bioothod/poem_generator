import tensorflow as tf
import numpy as np

import argparse
import imp
import json
import logging
import os
import re

from collections import defaultdict
from scipy.spatial.distance import cosine
from itertools import islice

space_char_id = ' '
eos_char_id = '\n'
pad_char_id = '+'

class Poem(object):
    def __init__(self, title, content):
        self.title = title
        content = content.lower()
        content = re.sub(r'[^\w\d\n\'`-]+', ' ', content).lower()
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

    def lines(self):
        for line in self.content:
            yield line
        return None

class Poet(object):
    def __init__(self):
        self.poems = []
        self.words = 0
        self.poet_id = None
        
    def add_text(self, poet_id, title, content):
        self.poet_id = poet_id

        self.poems.append(Poem(title, content))

    def collect_stats(self, word_dict, char_dict):
        for poem in self.poems:
            word_dict, char_dict = poem.collect_stats(word_dict, char_dict)

        return word_dict, char_dict

    def endings(self, num):
        for poem in self.poems:
            for line in poem.lines():
                yield line[-num:]

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

class Model(object):
    def __init__(self, config):
        self.config = config

        #language model placeholders
        self.lm_x    = tf.placeholder(tf.int32, [None, None])
        self.lm_xlen = tf.placeholder(tf.int32, [None])
        self.lm_y    = tf.placeholder(tf.int32, [None, None])
        self.lm_hist = tf.placeholder(tf.int32, [None, None])
        self.lm_hlen = tf.placeholder(tf.int32, [None])

        #pentameter model placeholders
        self.pm_enc_x    = tf.placeholder(tf.int32, [None, None])
        self.pm_enc_xlen = tf.placeholder(tf.int32, [None])
        self.pm_cov_mask = tf.placeholder(tf.float32, [None, None])

        #rhyme model placeholders
        self.rm_num_context = tf.placeholder(tf.int32)


    def get_last_hidden(self, h, xlen):
        ids = tf.range(tf.shape(xlen)[0])
        gather_ids = tf.concat([tf.expand_dims(ids, 1), tf.expand_dims(xlen-1, 1)], 1)
        return tf.gather_nd(h, gather_ids)

    def init_rhyme(self, is_training, batch_size, char_idx_size):
        cf = self.config

        self.char_embedding = tf.get_variable("char_embedding", [char_idx_size, cf.char_embedding_dim],
            initializer=tf.random_uniform_initializer(-0.05/cf.char_embedding_dim, 0.05/cf.char_embedding_dim))

        rnn_cell = tf.nn.rnn_cell.LSTMCell(cf.rm_dim)
        if is_training and cf.keep_prob < 1.0:
            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=cf.keep_prob)

        initial_state = rnn_cell.zero_state(tf.shape(self.pm_enc_x)[0], tf.float32)
        char_enc, _   = tf.nn.dynamic_rnn(rnn_cell, tf.nn.embedding_lookup(self.char_embedding, self.pm_enc_x),
            sequence_length=self.pm_enc_xlen, dtype=tf.float32, initial_state=initial_state)

        #get last hidden states
        char_enc = self.get_last_hidden(char_enc, self.pm_enc_xlen)
        
        #slice it into target_words and context words
        target  = char_enc[:batch_size,:]
        context = char_enc[batch_size:,:]

        target_tiled   = tf.reshape(tf.tile(target, [1, self.rm_num_context]), [-1, cf.rm_dim])
        target_context = tf.concat([target_tiled, context], 1)

        #cosine similarity
        e = tf.reduce_sum(tf.nn.l2_normalize(target_tiled, 1) * tf.nn.l2_normalize(context, 1), 1)
        e = tf.reshape(e, [batch_size, -1])

        #save the attentions
        self.rm_attentions = e

        #max margin loss
        min_cos = tf.nn.top_k(e, 2)[0][:, -1] #second highest cos similarity
        max_cos = tf.reduce_max(e, 1)
        self.rm_cost = tf.reduce_mean(tf.maximum(0.0, cf.rm_delta - max_cos + min_cos))

        if not is_training:
            return
        
        self.rm_train_op = tf.train.AdamOptimizer(cf.rm_learning_rate).minimize(self.rm_cost)


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

def sound_distance(word1, word2):
    """Фонетическое растояние на основе расстояния Левенштейна по окончаниям
    (число несовпадающих символов на соответствующих позициях)"""
    suffix_len = 3
    suffix1 = (' ' * suffix_len + word1)[-suffix_len:]
    suffix2 = (' ' * suffix_len + word2)[-suffix_len:]

    distance = sum((ch1 != ch2) for ch1, ch2 in zip(suffix1, suffix2))
    return distance

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--poems', required=True, type=str, help='Path to poems database')
    parser.add_argument('--config', default='config.py', type=str, help='Config python file')

    FLAGS = parser.parse_args()
    cf = imp.load_source('config', FLAGS.config)

    if not os.path.exists(cf.output_dir):
        os.makedirs(cf.output_dir)

    logging.basicConfig(filename=os.path.join(cf.output_dir, 'train.log'), filemode='a', level=logging.INFO,
            format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')

    poets = import_poems(FLAGS.poems)

    word_dict = defaultdict(int)
    char_dict = defaultdict(int)
    for poet_id, poet in poets.items():
        word_dict, char_dict = poet.collect_stats(word_dict, char_dict)

    char_dict[space_char_id] += 1
    char_dict[eos_char_id] += 1
    char_dict[pad_char_id] += 1

    word_idx = list(word_dict.keys())
    char_idx = list(char_dict.keys())

    char_idx_map = {}
    for idx, char in enumerate(char_idx):
        char_idx_map[char] = idx

    m = Model(cf)
    m.init_rhyme(True, cf.batch_size, len(char_idx))

    target_words = []
    target_words_len = []
    context_words = []
    context_words_len = []

    num_context = 6
    num_last_symbols = 10
    for poet_id, poet in poets.items():
        for win in window(poet.endings(num_last_symbols), num_context+1):
            win = [w.split()[-1] for w in win]

            target_words.append(win[0])
            target_words_len.append(len(win[0]))

            for c in win[1:]:
                context_words.append(c)
                context_words_len.append(len(c))

    max_word_len = max(max(target_words_len), max(context_words_len))
    target_words = [pad(w, max_word_len, pad_char_id) for w in target_words]
    context_words = [pad(w, max_word_len, pad_char_id) for w in context_words]

    target_words_numeric = [word_to_seq(w, char_idx_map) for w in target_words]
    context_words_numeric = [word_to_seq(w, char_idx_map) for w in context_words]

    if cf.save_model:
        saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        batch_start = 0
        step = 0
        num_epochs = 0

        while num_epochs < cf.num_epochs:
            tb = []
            tbl = []
            cb = []
            cbl = []
            for i in range(cf.batch_size):
                idx = (i + batch_start) % len(target_words)

                tb.append(target_words_numeric[idx])
                tbl.append(target_words_len[idx])

                for j in range(num_context):
                    idx = (i*num_context + j + batch_start*num_context) % len(context_words)

                    cb.append(context_words_numeric[idx])
                    cbl.append(context_words_len[idx])

            batch_start += cf.batch_size
            num_epochs = batch_start / len(target_words)

            batch_words = tb + cb
            batch_lens = tbl + cbl

            feed_dict = {
                    m.pm_enc_x: batch_words,
                    m.pm_enc_xlen: batch_lens,
                    m.rm_num_context: num_context,
            }

            cost, attns, _  = sess.run([m.rm_cost, m.rm_attentions, m.rm_train_op], feed_dict=feed_dict)
            step += 1

            def print_stats():
                max_pos = np.argmax(attns, 1)
                rhymes = []
                for idx, word in enumerate(tb):
                    candidates = []
                    for i in range(num_context):
                        candidate = cb[idx*num_context+i]
                        cword = seq_to_word(candidate, char_idx)

                        if i == max_pos[idx]:
                            cword = '*{}*'.format(cword)
                        candidates.append(cword)

                    rhymes.append((seq_to_word(word, char_idx), candidates))

                logging.info('{}: cost: {}, rhymes: {}'.format(step, cost, rhymes[:5]))
            
            if step % 100 == 0:
                print_stats()

            if step % 1000 == 0:
                if cf.save_model:
                    saver.save(sess, os.path.join(cf.output_dir, "model-{}.ckpt".format(step)))
        
        print_stats()
        if cf.save_model:
            saver.save(sess, os.path.join(cf.output_dir, "model-{}.ckpt".format(step)))
