import tensorflow as tf
import numpy as np
import os,sys
from conf import *

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_to_records_txtonly(query, title, query_ids, title_ids):
  example = tf.train.Example(features=tf.train.Features(feature={
      'query': _bytes_feature(query),
      'title': _bytes_feature(title),
      'query_ids': _bytes_feature(query_ids.tostring()),
      'title_ids': _bytes_feature(title_ids.tostring())
    }))

  return example

def neg_read_and_decode_txtonly(filename_queue,IMAGE_FEATURE_LEN, TEXT_MAX_WORDS):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'title': tf.FixedLenFeature([], tf.string),
          'title_ids': tf.FixedLenFeature([], tf.string)
      })
  title_feat = tf.decode_raw(features['title_ids'], tf.int64)
  title_feat.set_shape([TEXT_MAX_WORDS])
  return features['title'],title_feat

def read_and_decode_txtonly(sess, filename_queue,IMAGE_FEATURE_LEN, TEXT_MAX_WORDS):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'query': tf.FixedLenFeature([], tf.string),
          'title': tf.FixedLenFeature([], tf.string),
          'query_ids': tf.FixedLenFeature([], tf.string),
          'title_ids': tf.FixedLenFeature([], tf.string)
      })
  title_feat = tf.decode_raw(features['title_ids'], tf.int64)
  query_feat = tf.decode_raw(features['query_ids'], tf.int64)
  title_feat.set_shape([TEXT_MAX_WORDS])
  query_feat.set_shape([TEXT_MAX_WORDS])

  return features['query'],features['title'],query_feat, title_feat

def decode(comment):
    return '/'.join([vocab.key(id) for id in comment if id != 0])


def batch_inputs_txtonly(sess, files, IMAGE_FEATURE_LEN, TEXT_MAX_WORDS, batch_size=256, shuffle=True, num_epochs=None, num_preprocess_threads=1):
    if not num_epochs: num_epochs = None
    print files
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            files, 
            num_epochs=num_epochs,
            shuffle=True)
    query, title, query_feat, title_feat = read_and_decode_txtonly(sess, filename_queue, IMAGE_FEATURE_LEN, TEXT_MAX_WORDS)
    if shuffle:
        querys, titles, query_feats, title_feats = tf.train.shuffle_batch(
                [query, title, query_feat, title_feat], 
                batch_size=batch_size, 
                num_threads=num_preprocess_threads,
                capacity=100000 + 3 * batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=100000)
    else:
        querys, titles, query_feats, title_feats = tf.train.batch(
                [query, title, query_feat, title_feat], 
                batch_size=batch_size, 
                num_threads=num_preprocess_threads, 
                capacity=100000 + 3 * batch_size,)      
    return querys, titles, query_feats, title_feats
def neg_batch_inputs_txtonly(files, IMAGE_FEATURE_LEN, TEXT_MAX_WORDS, batch_size=256, shuffle=True, num_epochs=None, num_preprocess_threads=1):
    if not num_epochs: num_epochs = None
    with tf.name_scope('neg_input'):
        filename_queue = tf.train.string_input_producer(
            files, 
            num_epochs=num_epochs,
            shuffle=True)
    title, title_feat = neg_read_and_decode_txtonly(filename_queue, IMAGE_FEATURE_LEN, TEXT_MAX_WORDS)
    if shuffle:
        titles, title_feats = tf.train.shuffle_batch(
                [title, title_feat], 
                batch_size=batch_size, 
                num_threads=num_preprocess_threads,
                capacity=100000 + 3 * batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=100000)
    else:
        titles, title_feats = tf.train.batch(
                [title, title_feat], 
                batch_size=batch_size, 
                num_threads=num_preprocess_threads, 
                capacity=100000 + 3 * batch_size,)      
    return titles, title_feats

def read_dataset(sess, data_path, data_name, IMAGE_FEATURE_LEN, TEXT_MAX_WORDS, batch_size, \
         num_epochs, num_preprocess_threads):
    tf_record_pattern = os.path.join(data_path, '%s_*' % data_name)
    print tf_record_pattern
    data_files = tf.gfile.Glob(tf_record_pattern)
    with tf.variable_scope("train"):
        querys, titles, query_feats, title_feats = batch_inputs_txtonly(sess, files=data_files, 
                IMAGE_FEATURE_LEN=IMAGE_FEATURE_LEN,
                TEXT_MAX_WORDS=TEXT_MAX_WORDS,
                shuffle=True, 
                batch_size=batch_size, 
                num_epochs=num_epochs, 
                num_preprocess_threads = num_preprocess_threads)
        #negtitles, negtitle_feats = titles, title_feats
        negtitles, negtitle_feats = neg_batch_inputs_txtonly(files=data_files, 
                IMAGE_FEATURE_LEN=IMAGE_FEATURE_LEN,
                TEXT_MAX_WORDS=TEXT_MAX_WORDS,
                shuffle=True, 
                batch_size=batch_size*5, 
                num_epochs=num_epochs, 
                num_preprocess_threads = num_preprocess_threads)
        negtitles, negtitle_feats = reshape_neg_tensors([negtitles, negtitle_feats], batch_size, 5)
    return querys, titles, negtitles, query_feats, title_feats, negtitle_feats
        
def reshape_neg_tensors(neg_ops, batch_size, num_negs):
  neg_ops = list(neg_ops)
  for i in xrange(len(neg_ops)):
    #notice for strs will get [batch_size, num_negs, 1], will squeeze later
    neg_ops[i] = tf.reshape(neg_ops[i], [batch_size, num_negs,-1])
  return neg_ops        

def get_model_path(model_path, model_name=None):
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
      model_path = ckpt.model_checkpoint_path if model_name is None else os.path.join(model_path, model_name)
    return model_path
