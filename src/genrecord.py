#!/usr/bin/env python
  
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import tfrecords_tools

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('vocab', '../data/vocab.npy', 'vocabulary binary file')
flags.DEFINE_boolean('pad',True , 'wether to pad to pad 0 to make fixed length text ids')
flags.DEFINE_string('output_directory', '../data/train',
                         'Directory to download data files and write the '
                         'converted result')

flags.DEFINE_string('input', '../data/train', 'input pattern')
flags.DEFINE_string('trainname', 'train', '')
flags.DEFINE_string('vaildinput', '../data/vaild', '')
flags.DEFINE_string('vaildname', 'tf_vaild', '')
flags.DEFINE_integer('threads', 3, 'Number of threads for dealing')



import sys,os
import multiprocessing
from multiprocessing import Process, Manager, Value

import numpy as np
import jieba

import conf  
from conf import IMAGE_FEATURE_LEN, TEXT_MAX_WORDS, NUM_RESERVED_IDS, ENCODE_UNK


vocab = np.load(FLAGS.vocab)
vocabulary = vocab[()]

def pad(query_ids):
  qnum_words = len(query_ids)
  if qnum_words < TEXT_MAX_WORDS:
      query_ids += [0] * (TEXT_MAX_WORDS - qnum_words)
  else:
      qnum_words = TEXT_MAX_WORDS
      query_ids = query_ids[:qnum_words]
  return query_ids

def gen_id_from_text(text):
  word_list = jieba.cut(text)
  word_ids = [vocabulary[word] for word in word_list if word in vocabulary or ENCODE_UNK]
  word_ids_length = len(word_ids)
  if len(word_ids) == 0:
     return
  if FLAGS.pad:
     word_ids = pad(word_ids)

  return word_ids
  
def deal_file(file, thread_index, out_file):
  with tf.python_io.TFRecordWriter(out_file) as writer:
    num = 0
    for line in open(file):
      if num % 1000 == 0:
        print num
      num+=1
      l = line.strip().split('\1')
      if len(l)!=3:
        continue
      query = l[0]
      title = l[1]
      query_ids = gen_id_from_text(query)
      title_ids = gen_id_from_text(title)
      if not query_ids or not title_ids:
          continue
      #query_ids = np.array(query_ids)
      #title_ids = np.array(title_ids)
      example = tf.train.Example(features=tf.train.Features(feature={
              'query_str': tfrecords_tools._bytes_feature(query),
              'title_str': tfrecords_tools._bytes_feature(title),
              'query': tfrecords_tools._int64_feature(query_ids),
              'title': tfrecords_tools._int64_feature(title_ids)
            }))



      #example = tfrecords_tools.write_to_records_txtonly(query, title, query_ids, title_ids)
      writer.write(example.SerializeToString())
record = []
for thread_index in xrange(FLAGS.threads):
  in_file = '{}_{}'.format(FLAGS.input, thread_index) if FLAGS.threads > 1 else FLAGS.input
  out_file = '{}/{}_{}'.format(FLAGS.output_directory, FLAGS.trainname, thread_index) if FLAGS.threads > 1 else '{}/{}'.format(FLAGS.output_directory, FLAGS.trainname)
  args = (in_file, thread_index, out_file)
  process = multiprocessing.Process(target=deal_file,args=args)
  process.start()
  record.append(process)

for process in record:
  process.join()
"""
in_file =  FLAGS.vaildinput
out_file = '{}/{}'.format(FLAGS.output_directory, FLAGS.vaildname)
args = (in_file, 0, out_file)
print args
deal_file(in_file, 0, out_file)
"""
