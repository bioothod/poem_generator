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
    parser.add_argument('--poet', type=str, required=True, help='Use data for this poet')
    parser.add_argument('--train_dir', type=str, required=True, help='Use data from this directory')

    FLAGS = parser.parse_args()
    cf = imp.load_source('config', FLAGS.config)

    logging.basicConfig(filename=os.path.join(FLAGS.train_dir, 'train.log'), filemode='a', level=logging.INFO,
            format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')

    poets = text.import_poems(cf.poems_input_file)

    word_idx, word_idx_map, char_idx, char_idx_map = text.prepare_dicts(poets)

    for poet_id, poet in poets.items():
        if FLAGS.poet and poet_id != FLAGS.poet:
            continue

        logging.info('Using trained model for poet {}'.format(poet_id))
        poet.prepare_word_char_batches(word_idx_map, char_idx_map)

        m = model.Model(cf, poet_id, poet, word_idx, word_idx_map, char_idx, char_idx_map, False, 1)

        with m.graph.as_default(), tf.Session(graph=m.graph) as sess:
            if not cf.restore_model_step and not cf.restore_model_latest:
                logging.error('Neither model step nor latest model has been specified')
                exit(-1)

            saver = tf.train.Saver()

            if cf.restore_model_step:
                model_file = "model-{}-{}.ckpt".format(poet_id, cf.restore_model_step)
                saver.restore(sess, os.path.join(FLAGS.train_dir, model_file))
                logging.info('{}: restored {} step from {}'.format(poet_id, cf.restore_model_step, model_file))
                step = cf.restore_model_step
            elif cf.restore_model_latest:
                model_file = tf.train.latest_checkpoint(FLAGS.train_dir)
                if model_file:
                    saver.restore(sess, model_file)
                    step = int(model_file.split('/')[-1].split('-')[-1].split('.')[0])
                    logging.info('{}: restored latest step {} from {}'.format(poet_id, step, model_file))
                else:
                    logging.error('{}: could not restore latest checkpoint, since it is empty'.format(poet_id))
                    exit(-1)

            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            for _ in range(10):
                m.generate_poem(sess)
                logging.info('\n')
