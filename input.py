import tensorflow as tf
import numpy as np

import argparse
import imp
import json
import logging
import os
import random
import re

from collections import defaultdict, deque
from scipy.spatial.distance import cosine
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

class Model(object):
    def __init__(self, config, poet_id, poet, word_idx, word_idx_map, char_idx, char_idx_map):
        self.config = config

        self.graph = tf.Graph()

        self.poet = poet
        self.poet_id = poet_id
        self.char_idx_map = char_idx_map
        self.word_idx_map = word_idx_map

        with self.graph.as_default():
            self.pentameter = [0,1]*5
            self.pentameter_len = len(self.pentameter)

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

            is_training = True

            ##################
            #pentameter model#
            ##################
            with tf.variable_scope("pentameter_model"):
                self.init_pentameter(is_training, config.batch_size, len(char_idx), char_idx_map[space_char_id], char_idx_map[pad_char_id])

            ################
            #language model#
            ################
            with tf.variable_scope("language_model"):
                self.init_language_model(is_training, config.batch_size, len(word_idx))

            #############
            #rhyme model#
            #############
            with tf.variable_scope("rhyme_model"):
                self.init_rhyme(is_training, config.batch_size)

    def get_last_hidden(self, h, xlen):
        ids = tf.range(tf.shape(xlen)[0])
        gather_ids = tf.concat([tf.expand_dims(ids, 1), tf.expand_dims(xlen-1, 1)], 1)
        return tf.gather_nd(h, gather_ids)

    # -- sample a word given probability distribution (with option to normalise the distribution with temperature)
    # -- temperature = 0 means argmax
    def sample_word(self, sess, probs, temperature, unk_symbol_id, pad_symbol_id, wordxchar, idxword,
        rm_target_pos, rm_target_neg, rm_threshold_pos, avoid_words):

        def rhyme_pair_to_char(rhyme_pair):
            char_ids, char_id_len = [], []

            for word in rhyme_pair:
                char_ids.append(wordxchar[word])
                char_id_len.append(len(char_ids[-1]))

            #pad char_ids
            for ci, c in enumerate(char_ids):
                char_ids[ci] = pad(c, max(char_id_len), pad_symbol_id)

            return char_ids, char_id_len

        def rhyme_cos(x, y):
            rm_char, rm_char_len = rhyme_pair_to_char([x, y])

            feed_dict = {self.pm_enc_x: rm_char, self.pm_enc_xlen: rm_char_len, self.rm_num_context: 1}
            rm_attns  = sess.run(self.rm_attentions, feed_dict)

            return rm_attns[0][0]

        rm_threshold_neg = 0.7 #non-rhyming words A and B shouldn't have similarity larger than this threshold

        if temperature == 0:
            return np.argmax(probs)

        probs = probs.astype(np.float64) #convert to float64 for higher precision
        probs = np.log(probs) / temperature
        probs = np.exp(probs) / math.fsum(np.exp(probs))

        #avoid unk_symbol_id if possible
        sampled = None
        pw      = idxword[rm_target_pos] if rm_target_pos != None else "None"
        nw      = idxword[rm_target_neg] if rm_target_neg != None else "None"
        for i in range(1000):
            sampled = np.argmax(np.random.multinomial(1, probs, 1))

            #resample if it's a word to avoid
            if sampled in avoid_words:
                continue

            #if it needs to rhyme, resample until we find a rhyming word
            if rm_threshold_pos != 0.0 and rm_target_pos != None and rhyme_cos(sampled, rm_target_pos) < rm_threshold_pos:
                continue

            if rm_target_neg != None and rhyme_cos(sampled, rm_target_neg) > rm_threshold_neg:
                continue

            return sampled

        return None


    # -- generate a sentence by sampling one word at a time
    def sample_sent(self, sess, state, x, hist, hlen, xchar, xchar_len, avoid_symbols, stopwords,temp_min, temp_max,
        unk_symbol_id, pad_symbol_id, end_symbol_id, space_id, idxchar, charxid, idxword, wordxchar,
        rm_target_pos, rm_target_neg, rm_threshold, last_words, max_words):

        def filter_stop_symbol(word_ids):
            cleaned = set([])
            for w in word_ids:
                if w not in (stopwords | set([pad_symbol_id, end_symbol_id])) and not only_symbol(idxword[w]):
                    cleaned.add(w)
            return cleaned

        def get_freq_words(word_ids, freq_threshold):
            words     = []
            word_freq = Counter(word_ids)
            for k, v in list(word_freq.items()):
                #if v >= freq_threshold and not only_symbol(idxword[k]) and k != end_symbol_id:
                if v >= freq_threshold and k != end_symbol_id:
                    words.append(k)
            return set(words)

        sent   = []

        while True:
            probs, state = sess.run([self.lm_probs, self.lm_final_state],
                {self.lm_x: x, self.lm_initial_state: state, self.lm_xlen: [1],
                self.lm_hist: hist, self.lm_hlen: hlen,
                self.pm_enc_x: xchar, self.pm_enc_xlen: xchar_len})

            #avoid words previously generated            
            avoid_words = filter_stop_symbol(sent + hist[0])
            freq_words  = get_freq_words(sent + hist[0], 2) #avoid any words that occur >= N times
            avoid_words = avoid_words | freq_words | set(sent[-3:] + last_words + avoid_symbols + [unk_symbol_id])
            #avoid_words = set(sent[-3:] + last_words + avoid_symbols + [unk_symbol_id])

            word = self.sample_word(sess, probs[0], np.random.uniform(temp_min, temp_max), unk_symbol_id,
                pad_symbol_id, wordxchar, idxword, rm_target_pos, rm_target_neg, rm_threshold, avoid_words)

            if word != None:
                sent.append(word)
                x             = [[ sent[-1] ]]
                xchar         = [wordxchar[sent[-1]]]
                xchar_len     = [len(xchar[0])]
                rm_target_pos = None
                rm_target_neg = None
            else:
                return None, None, None

            if sent[-1] == end_symbol_id or len(sent) >= max_words:

                if len(sent) > 1:
                    pm_loss  = self.eval_pm_loss(sess, sent, end_symbol_id, space_id, idxchar, charxid, idxword, wordxchar)
                    return sent, state, pm_loss
                else:
                    return None, None, None

    def compute_pm_loss(self, is_training, batch_size, enc_hiddens, dec_cell, space_id, pad_id):
        cf             = self.config

        xlen_max       = tf.reduce_max(self.pm_enc_xlen)

        #use decoder hidden states to select encoder hidden states to predict stress for next time step
        repeat_loss    = tf.zeros([batch_size])
        attentions     = tf.zeros([batch_size, xlen_max]) #historical attention weights
        prev_miu       = tf.zeros([batch_size,1])
        outputs        = []
        attention_list = []
        miu_list       = []

        #initial inputs (learnable) and state
        initial_inputs = tf.get_variable("dec_init_input", [cf.pm_enc_dim*2])
        inputs         = tf.reshape(tf.tile(initial_inputs, [batch_size]), [batch_size, -1])
        state          = dec_cell.zero_state(batch_size, tf.float32)

        #manual unroll of time steps because attention depends on previous attention weights
        with tf.variable_scope("RNN"):
            for time_step in range(self.pentameter_len):

                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                def attend(enc_hiddens, dec_hidden, attn_hist, prev_miu):
                    with tf.variable_scope("pm_attention"):
                        attend_w = tf.get_variable("attend_w", [cf.pm_enc_dim*2+cf.pm_dec_dim, cf.pm_attend_dim])
                        attend_b = tf.get_variable("attend_b", [cf.pm_attend_dim], initializer=tf.constant_initializer())
                        attend_v = tf.get_variable("attend_v", [cf.pm_attend_dim, 1])
                        miu_w    = tf.get_variable("miu_w", [cf.pm_dec_dim+1, cf.pm_attend_dim])
                        miu_b    = tf.get_variable("miu_b", [cf.pm_attend_dim], initializer=tf.constant_initializer())
                        miu_v    = tf.get_variable("miu_v", [cf.pm_attend_dim, 1])
                    
                    #position attention
                    miu     = tf.minimum(tf.sigmoid(tf.matmul(tf.tanh(tf.matmul(tf.concat(
                        [dec_hidden, prev_miu], 1), miu_w) + miu_b), miu_v)) + prev_miu, tf.ones([batch_size, 1]))
                    miu_p   = miu * tf.reshape(tf.cast(self.pm_enc_xlen-1, tf.float32), [-1, 1])
                    pos     = tf.cast(tf.reshape(tf.tile(tf.range(xlen_max), [batch_size]), [batch_size, -1]),
                        dtype=tf.float32)
                    pos_lp  = -(pos - miu_p)**2 / (2 * tf.reshape(tf.tile([tf.square(cf.sigma)], [batch_size]),
                        [batch_size,-1]))
            
                    #char encoding attention
                    pos_weight = tf.reshape(tf.exp(pos_lp), [-1, 1])
                    inp_concat = tf.concat([enc_hiddens * pos_weight,
                        tf.reshape(tf.tile(dec_hidden, [1,xlen_max]), [-1,cf.pm_dec_dim])], 1)
                    x       = self.pm_enc_x
                    e       = tf.matmul(tf.tanh(tf.matmul(inp_concat, attend_w) + attend_b), attend_v)
                    e       = tf.reshape(e, [batch_size, xlen_max])
                    mask1   = tf.cast(tf.equal(x, tf.fill(tf.shape(x), pad_id)), dtype=tf.float32)
                    mask2   = tf.cast(tf.equal(x, tf.fill(tf.shape(x), space_id)), dtype=tf.float32)
                    e_mask  = tf.maximum(mask1, mask2)
                    e_mask *= tf.constant(-1e20)

                    #combine alpha with position probability
                    alpha   = tf.nn.softmax(e + e_mask + pos_lp)
                    #alpha   = tf.nn.softmax(e + e_mask)

                    #weighted sum
                    c       = tf.reduce_sum(tf.expand_dims(alpha, 2)*tf.reshape(enc_hiddens,
                        [batch_size, xlen_max, cf.pm_enc_dim*2]), 1)

                    return c, alpha, miu

                dec_hidden, state               = dec_cell(inputs, state)
                enc_hiddens_sum, attn, prev_miu = attend(enc_hiddens, dec_hidden, attentions, prev_miu)

                repeat_loss += tf.reduce_sum(tf.minimum(attentions, attn), 1)
                attentions  += attn
                inputs       = enc_hiddens_sum

                attention_list.append(attn)
                miu_list.append(prev_miu)
                outputs.append(enc_hiddens_sum)

        #reshape output into [batch_size*num_steps,hidden_size]
        outputs = tf.reshape(tf.concat(outputs, 1), [-1, cf.pm_enc_dim*2])

        #compute loss
        pm_softmax_w = tf.get_variable("pm_softmax_w", [cf.pm_enc_dim*2, 1])
        pm_softmax_b = tf.get_variable("pm_softmax_b", [1], initializer=tf.constant_initializer())
        pm_logit     = tf.squeeze(tf.matmul(outputs, pm_softmax_w) + pm_softmax_b)
        pm_crossent  = tf.nn.sigmoid_cross_entropy_with_logits(logits=pm_logit,
            labels=tf.tile(tf.cast(self.pentameter, tf.float32), [batch_size]))
        cov_loss     = tf.reduce_sum(tf.nn.relu(self.pm_cov_mask*cf.cov_loss_threshold - attentions), 1)
        pm_cost      = tf.reduce_sum(tf.reshape(pm_crossent, [batch_size, -1]), 1) + \
            cf.repeat_loss_scale*repeat_loss + cf.cov_loss_scale*cov_loss

        #save some variables
        self.pm_logits     = tf.sigmoid(tf.reshape(pm_logit, [batch_size, -1]))
        self.pm_attentions = attention_list
        self.mius          = miu_list

        return pm_cost

    def gated_layer(self, s, h, sdim, hdim):

        update_w = tf.get_variable("update_w", [sdim+hdim, hdim])
        update_b = tf.get_variable("update_b", [hdim], initializer=tf.constant_initializer(1.0))
        reset_w  = tf.get_variable("reset_w", [sdim+hdim, hdim])
        reset_b  = tf.get_variable("reset_b", [hdim], initializer=tf.constant_initializer(1.0))
        c_w      = tf.get_variable("c_w", [sdim+hdim, hdim])
        c_b      = tf.get_variable("c_b", [hdim], initializer=tf.constant_initializer())

        z = tf.sigmoid(tf.matmul(tf.concat([s, h], 1), update_w) + update_b)
        r = tf.sigmoid(tf.matmul(tf.concat([s, h], 1), reset_w) + reset_b)
        c = tf.tanh(tf.matmul(tf.concat([s, r*h], 1), c_w) + c_b)
        
        return (1-z)*h + z*c

    def selective_encoding(self, h, s, hdim):

        h1 = tf.shape(h)[0]
        h2 = tf.shape(h)[1]
        h_ = tf.reshape(h, [-1, hdim])
        s_ = tf.reshape(tf.tile(s, [1, h2]), [-1, hdim])

        attend_w = tf.get_variable("attend_w", [hdim*2, hdim])
        attend_b = tf.get_variable("attend_b", [hdim], initializer=tf.constant_initializer())

        g = tf.sigmoid(tf.matmul(tf.concat([h_, s_], 1), attend_w) + attend_b)

        return tf.reshape(h_* g, [h1, h2, -1])

    # -- language model network
    def init_language_model(self, is_training, batch_size, word_type_size):

        cf = self.config

        #shared word embeddings (used by encoder and decoder)
        self.word_embedding = tf.get_variable("word_embedding", [word_type_size, cf.word_embedding_dim],
            initializer=tf.random_uniform_initializer(-0.05/cf.word_embedding_dim, 0.05/cf.word_embedding_dim))
    
        #########
        #decoder#
        #########

        #define lstm cells
        lm_dec_cell = tf.nn.rnn_cell.LSTMCell(cf.lm_dec_dim, use_peepholes=True, forget_bias=1.0)
        if is_training and cf.keep_prob < 1.0:
            lm_dec_cell = tf.nn.rnn_cell.DropoutWrapper(lm_dec_cell, output_keep_prob=cf.keep_prob)
        self.lm_dec_cell = tf.nn.rnn_cell.MultiRNNCell([lm_dec_cell] * cf.lm_dec_layer_size)

        #initial states
        self.lm_initial_state = self.lm_dec_cell.zero_state(batch_size, tf.float32)
        state = self.lm_initial_state

        #pad symbol vocab ID = 0; create mask = 1.0 where vocab ID > 0 else 0.0
        lm_mask = tf.cast(tf.greater(self.lm_x, tf.zeros(tf.shape(self.lm_x), dtype=tf.int32)), dtype=tf.float32)

        #embedding lookup
        word_inputs = tf.nn.embedding_lookup(self.word_embedding, self.lm_x)
        if is_training and cf.keep_prob < 1.0:
            word_inputs = tf.nn.dropout(word_inputs, cf.keep_prob)

        #process character encodings
        #concat last hidden state of fw RNN with first hidden state of bw RNN
        fw_hidden = self.get_last_hidden(self.char_encodings[0], self.pm_enc_xlen)
        char_inputs = tf.concat([fw_hidden, self.char_encodings[1][:,0,:]], 1)
        char_inputs = tf.reshape(char_inputs, [batch_size, -1, cf.pm_enc_dim*2]) #reshape into same dimension as inputs
        
        #concat word and char encodings
        inputs = tf.concat([word_inputs, char_inputs], 2)
        #inputs = word_inputs

        #apply mask to zero out pad embeddings
        inputs = inputs * tf.expand_dims(lm_mask, -1)

        #dynamic rnn
        dec_outputs, final_state = tf.nn.dynamic_rnn(self.lm_dec_cell, inputs, sequence_length=self.lm_xlen, \
            dtype=tf.float32, initial_state=self.lm_initial_state)
        self.lm_final_state = final_state

        #########################
        #encoder (history words)#
        #########################

        #embedding lookup
        hist_inputs = tf.nn.embedding_lookup(self.word_embedding, self.lm_hist)
        if is_training and cf.keep_prob < 1.0:
            hist_inputs = tf.nn.dropout(hist_inputs, cf.keep_prob)

        #encoder lstm cell
        lm_enc_cell = tf.nn.rnn_cell.LSTMCell(cf.lm_enc_dim, forget_bias=1.0)
        if is_training and cf.keep_prob < 1.0:
            lm_enc_cell = tf.nn.rnn_cell.DropoutWrapper(lm_enc_cell, output_keep_prob=cf.keep_prob)

        #history word encodings
        hist_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lm_enc_cell, cell_bw=lm_enc_cell,
            inputs=hist_inputs, sequence_length=self.lm_hlen, dtype=tf.float32)

        #full history encoding
        full_encoding = tf.concat([hist_outputs[0][:,-1,:], hist_outputs[1][:,0,:]], 1)

        #concat fw and bw hidden states
        hist_outputs = tf.concat(hist_outputs, 2)

        #selective encoding
        with tf.variable_scope("selective_encoding"):
            hist_outputs = self.selective_encoding(hist_outputs, full_encoding, cf.lm_enc_dim*2)

        #attention (concat)
        with tf.variable_scope("lm_attention"):
            attend_w = tf.get_variable("attend_w", [cf.lm_enc_dim*2+cf.lm_dec_dim, cf.lm_attend_dim])
            attend_b = tf.get_variable("attend_b", [cf.lm_attend_dim], initializer=tf.constant_initializer())
            attend_v = tf.get_variable("attend_v", [cf.lm_attend_dim, 1])

        enc_steps = tf.shape(hist_outputs)[1]
        dec_steps = tf.shape(dec_outputs)[1]

        #prepare encoder and decoder
        hist_outputs_t = tf.tile(hist_outputs, [1, dec_steps, 1])
        dec_outputs_t  = tf.reshape(tf.tile(dec_outputs, [1, 1, enc_steps]),
            [batch_size, -1, cf.lm_dec_dim])

        #compute e
        hist_dec_concat = tf.concat([tf.reshape(hist_outputs_t, [-1, cf.lm_enc_dim*2]),
            tf.reshape(dec_outputs_t, [-1, cf.lm_dec_dim])], 1)
        e = tf.matmul(tf.tanh(tf.matmul(hist_dec_concat, attend_w) + attend_b), attend_v)
        e = tf.reshape(e, [-1, enc_steps])

        #mask out pad symbols to compute alpha and weighted sum of history words
        alpha    = tf.reshape(tf.nn.softmax(e), [batch_size, -1, 1])
        context  = tf.reduce_sum(tf.reshape(alpha * hist_outputs_t,
            [batch_size,dec_steps,enc_steps,-1]), 2)

        #save attention weights
        self.lm_attentions = tf.reshape(alpha, [batch_size, dec_steps, enc_steps])

        ##############
        #output layer#
        ##############

        #reshape both into [batch_size*len, hidden_dim]
        dec_outputs = tf.reshape(dec_outputs, [-1, cf.lm_dec_dim])
        context     = tf.reshape(context, [-1, cf.lm_enc_dim*2])
        
        #combine context and decoder hidden state with a gated unit
        with tf.variable_scope("gated_unit"):
            hidden = self.gated_layer(context, dec_outputs, cf.lm_enc_dim*2, cf.lm_dec_dim)
            #hidden = dec_outputs

        #output embeddings
        lm_output_proj = tf.get_variable("lm_output_proj", [cf.word_embedding_dim, cf.lm_dec_dim])
        lm_softmax_b   = tf.get_variable("lm_softmax_b", [word_type_size], initializer=tf.constant_initializer())
        lm_softmax_w   = tf.transpose(tf.tanh(tf.matmul(self.word_embedding, lm_output_proj)))

        #compute logits and cost
        lm_logits     = tf.matmul(hidden, lm_softmax_w) + lm_softmax_b
        lm_crossent   = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits, labels=tf.reshape(self.lm_y, [-1]))
        lm_crossent_m = lm_crossent * tf.reshape(lm_mask, [-1])
        self.lm_cost  = tf.reduce_sum(lm_crossent_m) / batch_size

        if not is_training:
            self.lm_probs = tf.nn.softmax(lm_logits)
            return

        #run optimiser and backpropagate (clipped) gradients for lm loss
        lm_tvars = tf.trainable_variables()
        lm_grads, _ = tf.clip_by_global_norm(tf.gradients(self.lm_cost, lm_tvars), cf.max_grad_norm)
        self.lm_train_op = tf.train.AdagradOptimizer(cf.lm_learning_rate).apply_gradients(list(zip(lm_grads, lm_tvars)))

    
    def init_pentameter(self, is_training, batch_size, char_idx_size, space_id, pad_id):
        cf = self.config

        #character embeddings
        self.char_embedding = tf.get_variable("char_embedding", [char_idx_size, cf.char_embedding_dim],
            initializer=tf.random_uniform_initializer(-0.05/cf.char_embedding_dim, 0.05/cf.char_embedding_dim))

        #get bidirectional rnn states of the encoder
        enc_cell = tf.nn.rnn_cell.LSTMCell(cf.pm_enc_dim)#, forget_bias=1.0)
        if is_training and cf.keep_prob < 1.0:
            enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=cf.keep_prob)

        char_inputs    = tf.nn.embedding_lookup(self.char_embedding, self.pm_enc_x)
        enc_hiddens, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=enc_cell, cell_bw=enc_cell, inputs=char_inputs,
            sequence_length=self.pm_enc_xlen, dtype=tf.float32)

        #save enc_hiddens
        self.char_encodings = enc_hiddens

        #reshape enc_hiddens
        x = tf.concat(enc_hiddens, 2)
        enc_hiddens  = tf.reshape(x, [-1, cf.pm_enc_dim*2]) #[batch_size*num_steps, hidden]

        #get decoder hidden states
        dec_cell = tf.nn.rnn_cell.LSTMCell(cf.pm_dec_dim)#, forget_bias=1.0)

        if is_training and cf.keep_prob < 1.0:
            dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=cf.keep_prob)

        #compute loss for pentameter
        self.pm_costs     = self.compute_pm_loss(is_training, batch_size, enc_hiddens, dec_cell, space_id, pad_id)
        self.pm_mean_cost = tf.reduce_sum(self.pm_costs) / batch_size

        if not is_training:
            return

        #run optimiser and backpropagate (clipped) gradients for pm loss
        pm_tvars         = tf.trainable_variables()
        pm_grads, _      = tf.clip_by_global_norm(tf.gradients(self.pm_mean_cost, pm_tvars), cf.max_grad_norm)
        self.pm_train_op = tf.train.AdamOptimizer(cf.pm_learning_rate).apply_gradients(list(zip(pm_grads, pm_tvars)))

    def init_rhyme(self, is_training, batch_size):
        cf = self.config

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

    def train(self, rm_batch_words, rm_batch_lens):
        cf = self.config

        with self.graph.as_default(), tf.Session(graph=self.graph) as sess:
            if cf.save_model or cf.restore_model_step or cf.restore_model_latest:
                saver = tf.train.Saver()

            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            zero_state  = sess.run(m.lm_initial_state)

            step = 0

            if cf.restore_model_step:
                model_file = "{}-{}.ckpt".format(self.poet_id, cf.restore_model_step)
                saver.restore(sess, os.path.join(cf.output_dir, model_file))
                logging.info('{}: restored {} step from {}'.format(self.poet_id, cf.restore_model_step, model_file))
                step = cf.restore_model_step
            elif cf.restore_model_latest:
                model_file = tf.train.latest_checkpoint(cf.output_dir)
                if model_file:
                    saver.restore(sess, model_file)
                    step = int(model_file.split('/')[-1].split('-')[-1].split('.')[0])
                    logging.info('{}: restored latest step {} from {}'.format(self.poet_id, step, model_file))
                else:
                    logging.info('{}: could not restore latest checkpoint, since it is empty'.format(self.poet_id))

            num_epochs = 0

            rm_batch_idx = 0

            while num_epochs < cf.num_epochs:

                #feed_dict        = {model.pm_enc_x: b[0], model.pm_enc_xlen: b[1], model.pm_cov_mask: b[2]}
                #cost, attns, _,  = sess.run([model.pm_mean_cost, model.pm_attentions, pm_train_op], feed_dict)


                rm_feed_dict = {
                        self.pm_enc_x: rm_batch_words[rm_batch_idx],
                        self.pm_enc_xlen: rm_batch_lens[rm_batch_idx],
                        self.rm_num_context: cf.rm_num_context * 2,
                }
                rm_batch_idx += 1
                if rm_batch_idx == len(rm_batch_words):
                    rm_batch_idx = 0

                rm_cost, rm_attns, _  = sess.run([self.rm_cost, self.rm_attentions, self.rm_train_op], feed_dict=rm_feed_dict)
                step += 1

                def print_stats():
                    max_pos = np.argmax(rm_attns, 1)
                    rhymes = []
                    batch = rm_feed_dict[self.pm_enc_x]
                    tb = batch[:cf.batch_size]
                    cb = batch[cf.batch_size:]
                    num_context = cf.rm_num_context * 2

                    for idx, word in enumerate(tb):
                        candidates = []
                        for i in range(num_context):
                            candidate = cb[idx*num_context+i]
                            cword = seq_to_word(candidate, char_idx)

                            if i == max_pos[idx]:
                                cword = '*{}*'.format(cword)
                            candidates.append(cword)

                        rhymes.append((seq_to_word(word, char_idx), candidates))

                    logging.info('{}: cost: {}, rhymes: {}'.format(step, rm_cost, rhymes[:5]))
                
                if step % 100 == 0:
                    print_stats()

                if step % cf.save_model_steps == 0:
                    if cf.save_model:
                        model_file = "model-{}.ckpt".format(step)
                        saver.save(sess, os.path.join(cf.output_dir, model_file))
                        logging.info('saving {} step into {}'.format(step, model_file))
            
            print_stats()
            if cf.save_model:
                model_file = "model-{}.ckpt".format(step)
                saver.save(sess, os.path.join(cf.output_dir, model_file))
                logging.info('saving {} step into {}'.format(step, model_file))


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.py', type=str, help='Config python file')

    FLAGS = parser.parse_args()
    cf = imp.load_source('config', FLAGS.config)

    if not os.path.exists(cf.output_dir):
        os.makedirs(cf.output_dir)

    logging.basicConfig(filename=os.path.join(cf.output_dir, 'train.log'), filemode='a', level=logging.INFO,
            format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')

    poets = import_poems(cf.poems_input_file)

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

    rm_batch_words, rm_batch_lens = prepare_rhyme_dataset(poets, cf, char_idx_map)

    for poet_id, poet in poets.items():
        poet.prepare_word_char_batches(word_idx_map, char_idx_map)

        m = Model(cf, poet_id, poet, word_idx, word_idx_map, char_idx, char_idx_map)
        m.train(rm_batch_words, rm_batch_lens)


