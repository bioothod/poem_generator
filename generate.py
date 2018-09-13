import tensorflow as tf

import argparse
import imp
import logging
import os

import model
import text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.py', type=str, help='Config python file')
    parser.add_argument('--train_dir', type=str, required=True, help='Use data from this directory')

    FLAGS = parser.parse_args()
    cf = imp.load_source('config', FLAGS.config)

    #logging.basicConfig(filename=os.path.join(FLAGS.train_dir, 'train.log'), filemode='a', level=logging.INFO,
    logging.basicConfig(filename=None, filemode='a', level=logging.INFO,
            format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')

    poets = text.import_poems(cf.poems_input_file)

    word_idx, word_idx_map, char_idx, char_idx_map = text.prepare_dicts(poets)

    m = model.Model(cf, '', None, word_idx, word_idx_map, char_idx, char_idx_map, False, 1)

    with m.graph.as_default(), tf.Session(graph=m.graph) as sess:
        saver = tf.train.Saver()

        model_file = tf.train.latest_checkpoint(FLAGS.train_dir)
        if model_file:
            saver.restore(sess, model_file)
            step = int(model_file.split('/')[-1].split('-')[-1].split('.')[0])
            logging.info('restored latest step {} from {}/{}'.format(step, FLAGS.train_dir, model_file))
        else:
            logging.error('Could not restore latest checkpoint from {}, since it is empty'.format(FLAGS.train_dir))
            exit(-1)

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        for _ in range(10):
            m.generate_poem(sess)
