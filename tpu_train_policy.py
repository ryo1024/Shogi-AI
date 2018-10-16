import numpy as np
import tensorflow as tf

from pydlshogi.network.policy import cnn_model
from pydlshogi.common import *
from pydlshogi.features import *
from pydlshogi.read_kifu import *
from tensorflow.python.client import device_lib

import argparse
import random
import pickle
import os
import re
import logging

parser = argparse.ArgumentParser()
parser.add_argument('kifulist_train', type=str, help='train kifu list')
parser.add_argument('kifulist_test', type=str, help='test kifu list')
parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of positions in each mini-batch')
parser.add_argument('--test_batchsize', type=int, default=512, help='Number of positions in each test mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=10, help='Number of epoch times')
parser.add_argument('--model', type=str, default='model/model_policy', help='model file name')
parser.add_argument('--state', type=str, default='model/state_policy', help='state file name')
parser.add_argument('--initmodel', '-m', default='', help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='', help='Resume the optimization from snapspot')
parser.add_argument('--log', default=None, help='log file path')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--eval_interval', '-i', type=int, default=1000, help='eval interval')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)

logging.info('read kifu start')

train_pickle_filename = re.sub(r'\..*?$', '', args.kifulist_train) + '.pickle'

if os.path.exists(train_pickle_filename):
    with open(train_pickle_filename, 'rb') as f:
        positions_train = pickle.load(f)
    logging.info('load train pickle')
else:
    positions_train = read_kifu(args.kifulist_train)

test_pickle_filename = re.sub(r'\..*?$', '', args.kifulist_test) + '.pickle'
if os.path.exists(test_pickle_filename):
    with open(test_pickle_filename, 'rb') as f:
        positions_test = pickle.load(f)
    logging.info('load test pickle')
else:
    positions_test = read_kifu(args.kifulist_test)

if not os.path.exists(train_pickle_filename):
    with open(train_pickle_filename, 'wb') as f:
        pickle.dump(positions_train, f, pickle.HIGHEST_PROTOCOL)
    logging.info('save train pickle')
if not os.path.exists(test_pickle_filename):
    with open(test_pickle_filename, 'wb') as f:
        pickle.dump(positions_test, f, pickle.HIGHEST_PROTOCOL)
    logging.info('save test pickle')
logging.info('read kifu end')

logging.info('train position num = {}'.format(len(positions_train)))
logging.info('test position num = {}'.format(len(positions_test)))


def mini_batch(positions, i, batchsize):
    mini_batch_data = []
    mini_batch_move = []
    for b in range(batchsize):
        features, move, win = make_features(positions[i + b])
        mini_batch_data.append(features)
        mini_batch_move.append(move)
    return np.array(mini_batch_data, dtype=np.float32), np.array(mini_batch_move, dtype=np.int32)


def mini_batch_for_test(positions, batchsize):
    mini_batch_data = []
    mini_batch_move = []
    for b in range(batchsize):
        features, move, win = make_features(random.choice(positions))
        mini_batch_data.append(features)
        mini_batch_move.append(move)

    return np.array(mini_batch_data, dtype=np.float32), np.array(mini_batch_move, dtype=np.int32)


tf.estimator.RunConfig(device_fn=lambda op: "/device:GPU:0")
mnist_classifier = tf.contrib.tpu.TPUEstimator(
    model_fn=cnn_model, model_dir="./model/model_policy")

print(device_lib.list_local_devices())

tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

for e in range(args.epoch):
    print("Training number " + str(e) + " start")
    positions_train_shuffled = random.sample(positions_train, len(positions_train))
    n = 1000*args.batchsize
    x, t = mini_batch(positions_train_shuffled, e * n, n)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=x,
        y=t,
        batch_size=args.batchsize,
        num_epochs=args.epoch,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1000,
        hooks=[logging_hook])
    p, q = mini_batch_for_test(positions_test, args.test_batchsize)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=p,
        y=q,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    print("Training number " + str(e) + " done")
