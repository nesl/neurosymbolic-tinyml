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
from edgeml_tf.tflite.bonsaiLayerMod import BonsaiLayerMod as BonsaiLayer 


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
	X_tr, Y_tr, X_test, Y_test = import_auritus_activity_dataset(dataset_folder = f, 
		                        use_timestamp=False, 
		                        shuffle=True, 
		                        window_size = window_size, stride = stride, 
		                        return_test_set = True, test_set_size = 300,channels=2)


	random_indices = np.random.choice(X_tr.shape[0], size=1000, replace=False)
	X_val = X_tr[random_indices,:,:]
	Y_val = Y_tr[random_indices,:]
	X_tr = np.delete(X_tr,random_indices,axis=0)
	Y_tr = np.delete(Y_tr,random_indices,axis=0)
                                
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
	    

	dataDimension = featX_tr.shape[1]
	numClasses = Y_tr.shape[1]
	Xtrain = featX_tr
	Ytrain = Y_tr
	Xtest = featX_test
	Ytest = Y_test
	Xval = featX_val
	Yval = Y_val
	
	Z = np.load( glob.glob(model_dir + "/**/Z.npy", recursive = True)[0], allow_pickle=True )
	W = np.load( glob.glob(model_dir + "/**/W.npy", recursive = True)[0], allow_pickle=True )
	V = np.load( glob.glob(model_dir + "/**/V.npy", recursive = True)[0], allow_pickle=True )
	T = np.load( glob.glob(model_dir + "/**/T.npy", recursive = True)[0], allow_pickle=True )
	hyperparams = np.load(glob.glob(model_dir + "/**/hyperParam.npy", 
		                recursive = True)[0], allow_pickle=True ).item()

	n_dim = hyperparams['dataDim']
	projectionDimension = hyperparams['projDim']
	numClasses = hyperparams['numClasses']
	depth = hyperparams['depth']
	sigma = hyperparams['sigma']

	dense = BonsaiLayer( numClasses, n_dim, projectionDimension, depth, sigma )
	new_model = keras.Sequential([
	keras.layers.InputLayer(n_dim),
	dense
	])

	dummy_tensor = tf.convert_to_tensor( np.zeros((1,n_dim), np.float32) )
	out_tensor = new_model(dummy_tensor)
	new_model.summary()
	dense.set_weights( [Z, W, V, T] )
	new_model.compile(loss=tf.keras.losses.categorical_crossentropy)  	

	
	
	if(val==True):
		adv_accu = perform_fgsm_attack(data=Xval, lab=Yval,eps=epsilon,my_model=new_model)
	else:
		adv_accu = perform_fgsm_attack(data=Xtest, lab=Ytest,eps=epsilon,my_model=new_model)
	
	print('Adversarial Accuracy:',adv_accu)
