#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import functools
import optools
import decode_then_shuffle
import shuffle_then_decode
from conf import IMAGE_FEATURE_LEN,TEXT_MAX_WORDS

flags = tf.app.flags
FLAGS = flags.FLAGS


def _decode(example, parse):
  features_dict = {
     'query': tf.VarLenFeature(tf.int64),
     'query_str': tf.FixedLenFeature([], tf.string),
     'title': tf.VarLenFeature(tf.int64),
     'title_str': tf.FixedLenFeature([], tf.string),
    }

  features = parse(example, features=features_dict)

  text_str = features['title_str']
  text = features['title']
  query_str = features['query_str']
  query = features['query']

  maxlen = 0 if FLAGS.dynamic_batch_length else TEXT_MAX_WORDS

  text = optools.sparse_tensor_to_dense(text, maxlen)
  query = optools.sparse_tensor_to_dense(query, maxlen)
  
  return query, query_str, text, text_str

def decode_examples(examples):
  return _decode(examples, tf.parse_example)

def decode_example(example):
  return _decode(example, tf.parse_single_example)

def decode_sequence_example(example):
  context_features_dict = {
     'title_str': tf.FixedLenFeature([], tf.string),
     'query_str': tf.FixedLenFeature([], tf.string),
    }

  features, sequence_features = tf.parse_single_sequence_example(example, 
                                              context_features=context_features_dict,
                                              sequence_features={
                                                 'title': tf.FixedLenSequenceFeature([], dtype=tf.int64),
                                                 'query': tf.FixedLenSequenceFeature([], dtype=tf.int64)
                                                })

  text_str = features['title_str']
  query_str = features['query_str']
  text = sequence_features['title']
  query = sequence_features['query']
  
  return query, query_str, text, text_str

#---------------for negative sampling using tfrecords
def _decode_neg(example, parse):
  features = parse(
      example,
      features={
          'title': tf.VarLenFeature(tf.int64),
          'title_str': tf.FixedLenFeature([], tf.string),
      })

  text = features['title']
  maxlen = 0 if FLAGS.dynamic_batch_length else TEXT_MAX_WORDS
  text = optools.sparse_tensor_to_dense(text, maxlen)
  text_str = features['title_str']
  
  return text, text_str

def decode_neg_examples(examples):
  return _decode_neg(examples, tf.parse_example)

def decode_neg_example(example):
  return _decode_neg(example, tf.parse_single_example)

def decode_neg_sequence_example(example):
  features, sequence_features = tf.parse_single_sequence_example(
      example,
      context_features={
          'title_str': tf.FixedLenFeature([], tf.string),
      },
      sequence_features={
          'title': tf.FixedLenSequenceFeature([], dtype=tf.int64),
        })

  text_str = features['title_str']
  text = sequence_features['title']
  
  return text, text_str

#-----------utils
def get_decodes(use_neg=True):
  if FLAGS.is_sequence_example:
    assert FLAGS.dynamic_batch_length, 'sequence example must be dyanmic batch length for fixed input'
    inputs = functools.partial(decode_then_shuffle.inputs,
                               dynamic_pad=True,
                               bucket_boundaries=FLAGS.buckets,
                               length_fn=lambda x: tf.shape(x[-2])[-1])
    decode = lambda x: decode_sequence_example(x)
    assert not (use_neg and FLAGS.buckets), 'if use neg, discriminant method do not use buckets'
    decode_neg = (lambda x: decode_neg_sequence_example(x)) if use_neg else None
  else:
    if FLAGS.shuffle_then_decode:
      inputs = shuffle_then_decode.inputs
      decode = lambda x: decode_examples(x)
      decode_neg = (lambda x: decode_neg_examples(x)) if use_neg else None
      print('decode_neg', decode_neg)

    else:
      #assert False, 'since have sparse data must use shuffle_then_decode'
      inputs = decode_then_shuffle.inputs
      decode = lambda x: decode_example(x)
      decode_neg = (lambda x: decode_neg_example(x)) if use_neg else None

  return inputs, decode, decode_neg

def reshape_neg_tensors(neg_ops, batch_size, num_negs):
  neg_ops = list(neg_ops)
  for i in xrange(len(neg_ops)):
    #notice for strs will get [batch_size, num_negs, 1], will squeeze later
    neg_ops[i] = tf.reshape(neg_ops[i], [batch_size, num_negs,-1])
  return neg_ops
