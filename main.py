import argparse
import imp
import logging
import os

import model
import text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.py', type=str, help='Config python file')

    FLAGS = parser.parse_args()
    cf = imp.load_source('config', FLAGS.config)

    if not os.path.exists(cf.output_dir):
        os.makedirs(cf.output_dir)

    logging.basicConfig(filename=os.path.join(cf.output_dir, 'train.log'), filemode='a', level=logging.INFO,
            format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')

    poets = text.import_poems(cf.poems_input_file)

    word_idx, word_idx_map, char_idx, char_idx_map = text.prepare_dicts(poets)

    rm_batch_words, rm_batch_lens = text.prepare_rhyme_dataset(poets, cf, char_idx_map)

    for poet_id, poet in poets.items():
        poet.prepare_word_char_batches(word_idx_map, char_idx_map)

        m = model.Model(cf, poet_id, poet, word_idx, word_idx_map, char_idx, char_idx_map)
        m.train(rm_batch_words, rm_batch_lens)

