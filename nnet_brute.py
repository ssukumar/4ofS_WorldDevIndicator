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
	
    def __init__(self,trainFile,trainLabelFile,valRatio,testFile=None):
        self.trainFile = trainFile
        self.testFile = testFile
        self.trainLabelFile = trainLabelFile
        self.valRatio = valRatio
		
    def inputPreprocess(self):
        trainData = pd.read_csv(self.trainFile)
        trainLabel = pd.read_table(self.trainLabelFile,header = None)
        testData = pd.read_csv(self.testFile)
        # The number of input features	

        trainData = trainData.replace(-1,np.nan)
        testData = testData.replace(-1,np.nan)
        self.features = trainData.shape[1];

        ##self.randfill(trainData)
        ##self.randfill(testData)
        testData = testData.fillna(testData.mean())
        trainData = trainData.fillna(trainData.mean())
		# [self.trainInputs, self.valInputs, self.trainTargets, self.valTargets] = train_test_split(trainData,trainLabel ,test_size = self.valRatio)

        self.trainInputs = trainData.as_matrix()
        self.testInputs = testData.as_matrix()
        self.trainTargets = trainLabel.as_matrix()
        self.trainMean = np.mean(self.trainInputs,axis=0)
        #self.testMean = np.mean(self.trainInputs_t,axis=0)
        tnStd = np.std(self.trainInputs,axis=0,dtype = np.float32)
        self.trainStd = np.array(list(tnStd))
        self.trainInputs = z_score_inputs(self.trainInputs,self.trainMean,self.trainStd);
        self.testInputs = z_score_inputs(self.testInputs,self.trainMean,self.trainStd)
# 		self.valInputs = z_score_inputs(self.valInputs,self.trainMean,self.trainStd);

    def randfill(self,data):
        data1 = data.as_matrix()
        r_c = np.shape(data1)
        for col in range(r_c[1]):
            col_idx = np.isnan(data1[:,col])
            mu = np.nanmean(data1[:,col])
            sigma = np.nanstd(data1[:,col])
            n = sum(col_idx)
            fill_values = np.random.normal(mu,sigma,n)
            data1[:,col][col_idx] = fill_values[:]
            #data.loc[col] = data1[col]
            #data.ix[:,col] = data.ix[:,col]
        print(data1)
        return(data1)
		
def z_score_inputs(Inputs, Mean, StdDev):
    """ 
		Pre-process inputs by making it mean zero and unit standard deviation
    """
    Inputs = np.divide((Inputs-Mean),StdDev)
    return Inputs
	
	
class Model:

    def __init__(self,neurons,activation,
                 optimizer,errFunc = 'mse',
                 epochs=500, wd= 0.01,dropout = 0.0):
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
        model.add(Dense(self.neurons[0],input_dim = data.features,init = 'glorot_uniform', W_regularizer = l2(self.wd),activation=self.activation[0]));
        model.add(Dropout(self.dropout));
        model.add(Dense(self.neurons[1], init = 'glorot_uniform', activation = self.activation[0]));
        model.add(Dropout(self.dropout))
        model.add(Dense(1,activation = self.activation[1]));
        model.compile(loss = self.errFunc, optimizer = self.optimizer);
		
        history = LossHistory()
		
		# Train the model
		
        model.fit(data.trainInputs, data.trainTargets, nb_epoch = self.epochs,validation_split = 0.00, batch_size = 32, callbacks = [history]);
        #predictions = 
        return(model.predict(data.testInputs))

        print('*****PREDICTIONS*****')
		
        print(predictions)
        np.savetxt('./predictions.csv', predictions,delimiter=',')
        plt.figure()
        plt.plot(history.losses)
        plt.show()
		
		
def main():
	
    activation = np.array(['tanh','relu'],dtype=object)
	
    error_function = 'mse'
    optimizer_model = 'RMSprop'
    neurons = [250, 50]

    data1 = Data('train.csv','train_labels.txt',0.30,'test.csv');
	
    data1.inputPreprocess();
	
    model1 = Model(neurons,activation,optimizer_model);
	
    return(model1.modelTrainValidate(data1))	

if __name__ == "__main__":
    results = pd.DataFrame(main(),columns=['run_0'],index=[range(1,4305)])
    for i in range(100):
        results['run_' + str(i+1)] = main()
        print(str(i) + ' % done')

    results.T.mean().to_csv('./ensemble.csv', sep=',')
    results.T.std().to_csv('./ensemle_std.csv',sep=',')
    results.to_hdf('full_results.h5','data')

