import argparse
import imp
import logging
import os

import model
import text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.py', type=str, help='Config python file')
    parser.add_argument('--poet', type=str, required=True, help='Use data for only this poet')

    FLAGS = parser.parse_args()
    cf = imp.load_source('config', FLAGS.config)

    cf.output_dir = 'train_{}'.format(FLAGS.poet)

    if not os.path.exists(cf.output_dir):
        os.makedirs(cf.output_dir)

    logging.basicConfig(filename=os.path.join(cf.output_dir, 'train.log'), filemode='a', level=logging.INFO,
            format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')

    poets = text.import_poems(cf.poems_input_file)

    word_idx, word_idx_map, char_idx, char_idx_map = text.prepare_dicts(poets)

    batch_size = 32
    rm_batch_words, rm_batch_lens = text.prepare_rhyme_dataset(poets, cf, char_idx_map, batch_size)

    for poet_id, poet in poets.items():
        if FLAGS.poet and poet_id != FLAGS.poet:
            continue

        logging.info('Generating data and training model for poet {}'.format(poet_id))
        poet.prepare_word_char_batches(word_idx_map, char_idx_map)

        m = model.Model(cf, poet_id, poet, word_idx, word_idx_map, char_idx, char_idx_map, True, batch_size)
        m.train(rm_batch_words, rm_batch_lens)

