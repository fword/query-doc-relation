#!/usr/bin/env python
# -*- coding: utf-8 -*-
  
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
import input
import logging
import util




def print_input_results(input_results):
  print('input_results:')
  for name, tensors in input_results.items():
    print(name)
    if tensors:
      for tensor in tensors:
        print(tensor)

class InputApp(object):
  def __init__(self):
    self.input_train_name = 'input_train'
    self.input_train_neg_name = 'input_train_neg'
    self.input_valid_name = 'input_valid'
    self.fixed_input_valid_name = 'fixed_input_valid'
    self.input_valid_neg_name = 'input_valid_neg'
    #-----------common for all app inputs may be 

    # self.step = 0

  def gen_train_input(self, inputs, decode_fn):
     #--------------------- train
    logging.info('train_input: %s'%FLAGS.train_input)
    print 'train_input: %s'%FLAGS.train_input
    trainset = util.list_files(FLAGS.train_input)
    logging.info('trainset:{} {}'.format(len(trainset), trainset[:2]))
    print 'trainset:{} {}'.format(len(trainset), trainset[:2])
    
    
    query, query_str, text, text_str = inputs(
      trainset, 
      decode_fn=decode_fn,
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.num_epochs, 
      #seed=seed,
      num_threads=FLAGS.num_threads,
      batch_join=FLAGS.batch_join,
      shuffle_files=FLAGS.shuffle_files,
      fix_sequence=FLAGS.fix_sequence,
      num_prefetch_batches=FLAGS.num_prefetch_batches,
      min_after_dequeue=FLAGS.min_after_dequeue,
      name=self.input_train_name)

    return (query, query_str, text, text_str), trainset

  def gen_train_neg_input(self, inputs, decode_neg_fn, trainset):
    assert FLAGS.num_negs > 0
    neg_text, neg_text_str = inputs(
      trainset, 
      decode_fn=decode_neg_fn,
      batch_size=FLAGS.batch_size * FLAGS.num_negs,
      num_epochs=0, 
      num_threads=FLAGS.num_threads,
      batch_join=FLAGS.batch_join,
      shuffle_files=FLAGS.shuffle_files,
      num_prefetch_batches=FLAGS.num_prefetch_batches,
      min_after_dequeue=FLAGS.min_after_dequeue,
      fix_sequence=FLAGS.fix_sequence,
      name=self.input_train_neg_name)
    return neg_text, neg_text_str


  def gen_input(self, train_only=False):

    input_results = {}

    input_name_list = [self.input_train_name, self.input_train_neg_name, self.input_valid_name]

    for name in input_name_list:
      input_results[name] = None

    assert FLAGS.shuffle_then_decode, "since use sparse data for text, must shuffle then decode"

    inputs, decode_fn, decode_neg_fn = \
     input.get_decodes(use_neg=(FLAGS.num_negs > 0))

    input_results[self.input_train_name], trainset = self.gen_train_input(inputs, decode_fn)

    if decode_neg_fn is not None:
      input_results[self.input_train_neg_name] = self.gen_train_neg_input(inputs, decode_neg_fn, trainset)
    
    if not train_only:
      #---------------------- valid
      train_with_validation = bool(FLAGS.valid_input) 
      self.train_with_validation = train_with_validation
      print('train_with_validation:', train_with_validation)
      if train_with_validation:
        input_results[self.input_valid_name], \
        input_results[self.fixed_input_valid_name], \
        eval_batch_size = self.gen_valid_input(inputs, decode_fn)

        if decode_neg_fn is not None:
          input_results[self.input_valid_neg_name] = self.gen_valid_neg_input(inputs, decode_neg_fn, trainset, eval_batch_size)

    print_input_results(input_results)


    return input_results
