from __future__ import division, print_function, absolute_import
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib import data_util
from lib.config import params_setup
from lib.googlenet import GoogLeNet
from datetime import datetime

import pickle, gzip
import numpy as np
import tflearn.datasets.oxflower17 as oxflower17

# import AI Vision train service module
from dnn_train import train_service

#-------------------------------
#   Training
#-------------------------------

# init AI Vision train service
train_service = train_service.TrainService()

args = params_setup()
gnet = GoogLeNet(args, train_service)

# go to pre-processing stage
train_service.sendStatusMessagePreproccess()

# go to training stage
train_service.sendStatusMessageTrain()
print(pkl_files)
for f in pkl_files:
    X, Y = pickle.load(gzip.open(f, 'rb'))
    gnet.fit(X, Y, n_epoch=100)
    gnet.save()
    print('[pkl_files] done with %s @ %s' % (f, datetime.now()))

# go to complete stage
train_service.sendStatusMessageComplete()
