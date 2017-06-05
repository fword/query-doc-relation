import os, sys, time

import tensorflow as tf
import numpy as np
import time
from conf import *
from datetime import datetime
import input_app as InputApp
import traceback

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train_input', '../data/train', '')
flags.DEFINE_string('board', '../board/', '')
flags.DEFINE_string('model', '../model/', '')
flags.DEFINE_string('train_data_name', 'tf_train', '')
flags.DEFINE_integer('batch_size', 6, 'Batch size.')
flags.DEFINE_integer('num_epochs', '10', '')
flags.DEFINE_integer('min_records', '100', '')
flags.DEFINE_integer('num_preprocess_threads', '1', '')

flags.DEFINE_boolean('shuffle_then_decode', True, '')
flags.DEFINE_string('name', 'train', 'records name')
flags.DEFINE_boolean('dynamic_batch_length', True, '')
flags.DEFINE_boolean('is_sequence_example', False, '')
flags.DEFINE_integer('num_negs', 1, '')

flags.DEFINE_integer('num_threads', 12, '')
flags.DEFINE_boolean('batch_join', True, '')
flags.DEFINE_boolean('shuffle_batch', True, '')
flags.DEFINE_boolean('shuffle_files', True, '')
flags.DEFINE_boolean('fix_sequence', True, '')
flags.DEFINE_integer('num_prefetch_batches', 500, '')
flags.DEFINE_integer('min_after_dequeue', 100000, '')




def train():
    with tf.Session() as sess:
            input_app = InputApp.InputApp()
            input_results = input_app.gen_input(train_only=True)
            query, query_str, text, text_str = input_results[input_app.input_train_name]
            neg_text, neg_text_str = input_results[input_app.input_train_neg_name]
            #print input_results[input_app.input_train_neg_name]
            init = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            step = 0
            try:
                while not coord.should_stop():
                    start_time = time.time()
                    step = step + 1
                    #a,b,c,d,e,f = sess.run([query_str[0],text_str[0],neg_text_str[0],query[0],text[0],neg_text[0]])
                    a,b,c,d,e,f,g,h,i,j,k,l = sess.run([query_str[0],text_str[0],query_str[1],text_str[1], \
query_str[2],text_str[2],query_str[3],text_str[3], \
query_str[4],text_str[4],query_str[5],text_str[5]
])
                    print a
                    print b
                    print c
                    print d
                    print e
                    print f
                    print g
                    print h
                    print i
                    print j
                    print k
                    print l
                    print '-----------------------'
                    duration = time.time() - start_time
                    #if step%10 == 0:
                        #saver.save(sess, os.path.join(FLAGS.model, 'model_%d.ckpt'%step))
            except :
                traceback.print_exc()
                pass
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
                coord.join(threads)
            sess.close()
def main(_):
    with tf.variable_scope("read"):
        train()
if __name__ == '__main__':
  tf.app.run()
