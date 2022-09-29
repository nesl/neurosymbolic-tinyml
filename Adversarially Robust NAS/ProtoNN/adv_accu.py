import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import pandas as pd
import numpy as np
from gtda.time_series import SlidingWindow
import matplotlib.pyplot as plt
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True  
config.log_device_placement = True  
sess2 = tf.compat.v1.Session(config=config)
set_session(sess2)  
import csv
import random
import itertools
import glob
import time
import pickle
from data_utils import *
np.random.seed(42)
random.seed(1)

from tensorflow import keras
from tqdm import tqdm
from tensorflow.keras.losses import MSE
from edgeml_tf.tflite.protoNNLayer import ProtoNNLayer


def fgsm_attack(model, image, label, eps):
    image = tf.cast(image, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        pred = model(image)
        loss = MSE(label, pred)
        gradient = tape.gradient(loss, image)
        signedGrad = tf.sign(gradient)
        adversary = (image + (signedGrad * eps)).numpy()
        return adversary
        
def perform_fgsm_attack(data,lab,eps,my_model):
    countadv = 0
    for i in tqdm(range(len(data))):
        act = data[i,:].reshape(1,data.shape[1])
        label = lab[i,:]
        actPred = my_model.predict(act)
        actPred = actPred.argmax()
        adversary = fgsm_attack(my_model,act, label, eps=eps)
        pred = my_model.predict(adversary)
        adversaryPred = pred[0].argmax()
        if actPred == adversaryPred:
            countadv += 1
        
 
    adv_accu = countadv / len(data)
    return adv_accu

def get_adv_accu(epsilon,model_dir,data_dir,window_size,stride,val=False):
	f = data_dir
	X_tr, Y_tr, X_test, Y_test, X_val,Y_val = import_auritus_activity_dataset(dataset_folder = f, 
                                use_timestamp=False, 
                                shuffle=True, 
                                window_size = window_size, stride = stride, 
                                return_test_set = True, return_val_set = True, test_set_size = 300,channels=2)
                                
	feat_size = 10
	featX_tr = np.zeros((X_tr.shape[0],feat_size))
	featX_test = np.zeros((X_test.shape[0],feat_size))
	featX_val = np.zeros((X_val.shape[0],feat_size))
	for i in range(X_tr.shape[0]):
	    cur_win = X_tr[i]
	    featX_tr[i,0] = np.min(cur_win[:,0])
	    featX_tr[i,1] = np.min(cur_win[:,1])
	    featX_tr[i,2] = np.max(cur_win[:,0])
	    featX_tr[i,3] = np.max(cur_win[:,1])
	    featX_tr[i,4] = featX_tr[i,2]-featX_tr[i,0]
	    featX_tr[i,5] = featX_tr[i,3]-featX_tr[i,1]
	    featX_tr[i,6] = np.var(cur_win[:,0])
	    featX_tr[i,7] = np.var(cur_win[:,1])
	    featX_tr[i,8] = np.sqrt(featX_tr[i,6])
	    featX_tr[i,9] = np.sqrt(featX_tr[i,7])  
	    
	for i in range(X_test.shape[0]):
	    cur_win = X_test[i]
	    featX_test[i,0] = np.min(cur_win[:,0])
	    featX_test[i,1] = np.min(cur_win[:,1])
	    featX_test[i,2] = np.max(cur_win[:,0])
	    featX_test[i,3] = np.max(cur_win[:,1])
	    featX_test[i,4] = featX_test[i,2]-featX_test[i,0]
	    featX_test[i,5] = featX_test[i,3]-featX_test[i,1]
	    featX_test[i,6] = np.var(cur_win[:,0])
	    featX_test[i,7] = np.var(cur_win[:,1])
	    featX_test[i,8] = np.sqrt(featX_test[i,6])
	    featX_test[i,9] = np.sqrt(featX_test[i,7])
	    
	for i in range(X_val.shape[0]):
	    cur_win = X_val[i]
	    featX_val[i,0] = np.min(cur_win[:,0])
	    featX_val[i,1] = np.min(cur_win[:,1])
	    featX_val[i,2] = np.max(cur_win[:,0])
	    featX_val[i,3] = np.max(cur_win[:,1])
	    featX_val[i,4] = featX_val[i,2]-featX_val[i,0]
	    featX_val[i,5] = featX_val[i,3]-featX_val[i,1]
	    featX_val[i,6] = np.var(cur_win[:,0])
	    featX_val[i,7] = np.var(cur_win[:,1])
	    featX_val[i,8] = np.sqrt(featX_val[i,6])
	    featX_val[i,9] = np.sqrt(featX_val[i,7])
	    
	x_train = featX_tr
	y_train = Y_tr
	x_test = featX_test
	y_test = Y_test
	x_val = featX_val
	y_val = Y_val
	numClasses = Y_tr.shape[1]
	dataDimension = x_train.shape[1]

	mean = np.mean(x_train, 0)
	std = np.std(x_train, 0)
	std[std[:] < 0.000001] = 1
	x_train = (x_train - mean) / std
	x_test = (x_test - mean) / std
	x_val = (x_val-mean)/std
	Y_tr_int = np.argmax(Y_tr, axis=1)
	Y_test_int = np.argmax(Y_test,axis=1)
	Y_val_int = np.argmax(Y_val,axis=1)
	
	Z = np.load( glob.glob(model_dir + "/**/Z.npy", recursive = True)[0], allow_pickle=True )
	W = np.load( glob.glob(model_dir + "/**/W.npy", recursive = True)[0], allow_pickle=True )
	B = np.load( glob.glob(model_dir + "/**/B.npy", recursive = True)[0], allow_pickle=True )
	gamma = np.load( glob.glob(model_dir + "/**/gamma.npy", recursive = True)[0], allow_pickle=True )


	n_dim = inputDimension = W.shape[0]
	projectionDimension = W.shape[1]
	numPrototypes = B.shape[1]
	numOutputLabels = Z.shape[0]

	dense = ProtoNNLayer( inputDimension, projectionDimension, numPrototypes, numOutputLabels, gamma )

	model = keras.Sequential([
	keras.layers.InputLayer(n_dim),
	dense
	])

	dummy_tensor = tf.convert_to_tensor( np.zeros((1,n_dim), np.float32) )
	out_tensor = model( dummy_tensor )

	model.summary()

	dense.set_weights( [W, B, Z] )
	model.compile() 
	if(val==True):
		adv_accu = perform_fgsm_attack(data=x_val, lab=y_val,eps=epsilon,my_model=model)
	else:
		adv_accu = perform_fgsm_attack(data=x_test, lab=y_test,eps=epsilon,my_model=model)
	
	print('Adversarial Accuracy:',adv_accu)
