import os
import csv 
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import keras
from keras.utils import np_utils 
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l1, l2,l1l2
import matplotlib.pyplot as plt

class Data:
	'Class for data object'
	
	def __init__(self,trainFile,trainLabelFile,valRatio):
		self.trainFile = trainFile;
		self.trainLabelFile = trainLabelFile;
# 		self.testFile = testFile;
		self.valRatio = valRatio;
		
	def inputPreprocess(self):
		trainData = pd.read_csv(self.trainFile);
		trainLabel = pd.read_table(self.trainLabelFile,header = None);
		# The number of input features
		
		trainData = trainData.replace(-1,np.nan);
		trainData = trainData.fillna(trainData.mean())
		self.features = trainData.shape[1];
		print self.features
		trainData = trainData.as_matrix();
		trainLabel = trainLabel.as_matrix();
		# [self.trainInputs, self.valInputs, self.trainTargets, self.valTargets] = train_test_split(trainData,trainLabel ,test_size = self.valRatio)	
# 		

		self.trainInputs = trainData;
		self.trainTargets = trainLabel;
		print self.trainInputs
		self.trainMean = np.mean(self.trainInputs,axis=0)
		tnStd = np.std(self.trainInputs,axis=0,dtype = np.float32)
		self.trainStd = np.array(list(tnStd))
		self.trainInputs = z_score_inputs(self.trainInputs,self.trainMean,self.trainStd);
# 		self.valInputs = z_score_inputs(self.valInputs,self.trainMean,self.trainStd);
		print self.trainInputs
		
		
def z_score_inputs(Inputs, Mean, StdDev):
	""" 
		Pre-process inputs by making it mean zero and unit standard deviation
	"""
# 	for i in range(StdDev.size):
		# if (StdDev[i] < 0.02):
# 			StdDev[i] = 0.02
	
	Inputs = np.divide((Inputs-Mean),StdDev)
	return Inputs
	
	
class Model:

	def __init__(self,neurons,activation,optimizer,errFunc = 'mse',epochs=200, wd= 0.01,dropout = 0.0):
		self.neurons = neurons;
		self.errFunc = errFunc;
		self.activation = activation;
		self.optimizer = optimizer;
		self.epochs = epochs;
		self.wd = wd;
		self.dropout = dropout;
		
		
	def modelTrainValidate(self,data):
		
		# Callback used to track crossentropy loss during training
	
		class LossHistory(keras.callbacks.Callback):
			def on_train_begin(self,logs={}):
				self.losses = []
		
			def on_batch_end(self, batch, logs={}):
				self.losses.append(logs.get('loss'))
		
		# Build the NN model
		
		model = Sequential();
		model.add(Dense(self.neurons,input_dim = data.features,init = 'glorot_uniform', W_regularizer = l2(self.wd),activation=self.activation[0]));
		model.add(Dropout(self.dropout));
		model.add(Dense(1,activation = self.activation[1]));
		model.add(Dropout(self.dropout));
		model.compile(loss = self.errFunc, optimizer = self.optimizer);
		
		history = LossHistory()
		
		# Train the model
		
		model.fit(data.trainInputs, data.trainTargets, nb_epoch = self.epochs, batch_size = 100, callbacks = [history]);
		predictions = model.predict(data.trainInputs);
		plt.figure()
		plt.plot(history.losses)
		plt.show()
		
		
def main():
	
	activation = np.array(['sigmoid','relu'],dtype=object)
	
	error_function = 'mse'
	optimizer_model = 'sgd'
	
	data1 = Data('train.csv','train_labels.txt',0.30);
	
	data1.inputPreprocess();
	
	model1 = Model(500,activation,optimizer_model);
	
	model1.modelTrainValidate(data1);
	
	
if __name__ == "__main__":
	main()
	
