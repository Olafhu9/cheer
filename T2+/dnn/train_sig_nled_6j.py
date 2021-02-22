from __future__ import division
import sys, os

#3os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from sklearn.utils import shuffle
import re
import string
import math
from ROOT import TFile, TTree
from ROOT import *
import ROOT
import numpy as np
from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation, Dropout, add
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from array import array
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.callbacks import Callback, ModelCheckpoint

timer = ROOT.TStopwatch()
timer.Start()

node = int(sys.argv[1]) #Node
layer = int(sys.argv[2]) #Layer
epochs = int(sys.argv[3]) #Epoch
dropout = float(sys.argv[4]) #Dropout

resultTxt = "result_1122_6j.txt"
resultDir = "/home/juhee5819/T2+/result/1122_6j/sig"
trainInput = "/home/juhee5819/T2+/array/ttbb_2018_4f_pt20_6j.h5"

data = pd.read_hdf(trainInput)
print len(data)
# remove background event (cat 6)
data = data.drop(data[data['category'] == 15].index)


# 6j variables
#variables = ["nbjets_m", "lepton_pt", "lepton_eta", "lepton_e", "MET", "MET_phi", "jet1_pt", "jet1_eta", "jet1_e", "jet1_m", "jet1_btag", "jet1_CvsB", "jet1_CvsL", "jet2_pt", "jet2_eta", "jet2_e", "jet2_m", "jet2_btag", "jet2_CvsB", "jet2_CvsL", "jet3_pt", "jet3_eta", "jet3_e", "jet3_m", "jet3_btag", "jet3_CvsB", "jet3_CvsL", "jet4_pt", "jet4_eta", "jet4_e", "jet4_m", "jet4_btag", "jet4_CvsB", "jet4_CvsL", "jet5_pt", "jet5_eta", "jet5_e", "jet5_m", "jet5_btag", "jet5_CvsB", "jet5_CvsL", "jet6_pt", "jet6_eta", "jet6_e", "jet6_m", "jet6_btag", "jet6_CvsB", "jet6_CvsL", "dR12", "dR13", "dR14", "dR15", "dR16", "dR23", "dR24", "dR25", "dR26", "dR34", "dR35", "dR36", "dR45", "dR46", "dR56", "dEta12", "dEta13", "dEta14", "dEta15", "dEta16", "dEta23", "dEta24", "dEta25", "dEta26", "dEta34", "dEta35", "dEta36", "dEta45", "dEta46", "dEta56", "dPhi12", "dPhi13", "dPhi14", "dPhi15", "dPhi16", "dPhi23", "dPhi24", "dPhi25", "dPhi26", "dPhi34", "dPhi35", "dPhi36", "dPhi45", "dPhi46", "dPhi56", "invm12", "invm13", "invm14", "invm15", "invm16", "invm23", "invm24", "invm25", "invm26", "invm34", "invm35", "invm36", "invm45", "invm46", "invm56", "dRlep1", "dRlep2", "dRlep3", "dRlep4", "dRlep5", "dRlep6", "dRnu1", "dRnu2", "dRnu3", "dRnu4", "dRnu5", "dRnu6", "dRnulep1", "dRnulep2", "dRnulep3", "dRnulep4", "dRnulep5", "dRnulep6", "dRnulep12", "dRnulep13", "dRnulep14", "dRnulep15", "dRnulep16", "dRnulep23", "dRnulep24", "dRnulep25", "dRnulep26", "dRnulep34", "dRnulep35", "dRnulep36", "dRnulep45", "dRnulep46", "dRnulep56", "invmlep1", "invmlep2", "invmlep3", "invmlep4", "invmlep5", "invmlep6", "invmnu1", "invmnu2", "invmnu3", "invmnu4", "invmnu5", "invmnu6"]
# 6j variables + global
variables = ["ngoodjets", "nulep_pt", "St", "Ht", "nbjets_m", "lepton_pt", "lepton_eta", "lepton_e", "MET", "MET_phi", "jet1_pt", "jet1_eta", "jet1_e", "jet1_m", "jet1_btag", "jet1_CvsB", "jet1_CvsL", "jet2_pt", "jet2_eta", "jet2_e", "jet2_m", "jet2_btag", "jet2_CvsB", "jet2_CvsL", "jet3_pt", "jet3_eta", "jet3_e", "jet3_m", "jet3_btag", "jet3_CvsB", "jet3_CvsL", "jet4_pt", "jet4_eta", "jet4_e", "jet4_m", "jet4_btag", "jet4_CvsB", "jet4_CvsL", "jet5_pt", "jet5_eta", "jet5_e", "jet5_m", "jet5_btag", "jet5_CvsB", "jet5_CvsL", "jet6_pt", "jet6_eta", "jet6_e", "jet6_m", "jet6_btag", "jet6_CvsB", "jet6_CvsL", "dR12", "dR13", "dR14", "dR15", "dR16", "dR23", "dR24", "dR25", "dR26", "dR34", "dR35", "dR36", "dR45", "dR46", "dR56", "dEta12", "dEta13", "dEta14", "dEta15", "dEta16", "dEta23", "dEta24", "dEta25", "dEta26", "dEta34", "dEta35", "dEta36", "dEta45", "dEta46", "dEta56", "dPhi12", "dPhi13", "dPhi14", "dPhi15", "dPhi16", "dPhi23", "dPhi24", "dPhi25", "dPhi26", "dPhi34", "dPhi35", "dPhi36", "dPhi45", "dPhi46", "dPhi56", "invm12", "invm13", "invm14", "invm15", "invm16", "invm23", "invm24", "invm25", "invm26", "invm34", "invm35", "invm36", "invm45", "invm46", "invm56", "dRlep1", "dRlep2", "dRlep3", "dRlep4", "dRlep5", "dRlep6", "dRnu1", "dRnu2", "dRnu3", "dRnu4", "dRnu5", "dRnu6", "dRnulep1", "dRnulep2", "dRnulep3", "dRnulep4", "dRnulep5", "dRnulep6", "dRnulep12", "dRnulep13", "dRnulep14", "dRnulep15", "dRnulep16", "dRnulep23", "dRnulep24", "dRnulep25", "dRnulep26", "dRnulep34", "dRnulep35", "dRnulep36", "dRnulep45", "dRnulep46", "dRnulep56", "invmlep1", "invmlep2", "invmlep3", "invmlep4", "invmlep5", "invmlep6", "invmnu1", "invmnu2", "invmnu3", "invmnu4", "invmnu5", "invmnu6"]
numbertr=len(data)

trainnb=0.7 # Fraction used for training

#Splitting between training set and cross-validation set
pd_valid = data[int(trainnb*numbertr):numbertr]
pd_train = data[0:int(trainnb*numbertr)]

#number of categories
pd_valid_out = pd_valid.filter(items = ['category'])
pd_valid_data = pd_valid.filter(items = variables)

catlist = pd_valid_out.apply(set)
numcat = catlist.str.len() 
print 'numcat ', numcat

#print 'train data not droped: ',  len(pd_train)
#print 'valid data: ', len(pd_valid)

#make the number of events uniformly
cat_num_list = [(len(pd_train.loc[pd_train['category'] == i])) for i in range(numcat)]
smallest = min(cat_num_list)
print 'number of each category = ',cat_num_list
print 'the smallest one = ', smallest

pd_train_droped = pd.DataFrame()
np.random.seed(10)
for i in range(numcat):
	pd_cat = pd_train.loc[pd_train['category'] == i]
	remove_cat = len(pd_train.loc[pd_train['category'] == i]) - smallest
	cat_drop_indices = np.random.choice(pd_cat.index, remove_cat, replace=False)
	pd_cat_droped = pd_cat.drop(cat_drop_indices)
	pd_train_droped = pd_train_droped.append(pd_cat_droped)
	#print 're pd_train_droped', len(pd_train_droped)

# droped data
#pd_train_out  = pd_train_droped.filter(items = ['category'])
#pd_train_data = pd_train_droped.filter(items = variables)

# not droped data
pd_train_out = pd_train.filter(items = ['category'])
pd_train_data = pd_train.filter(items = variables)

#covert from pandas to array
train_data_out = np.array( pd_train_out )
train_data1 = np.array( pd_train_data )

valid_data_out = np.array( pd_valid_out )
valid_data = np.array( pd_valid_data )


numbertr1 = len(train_data1)
numbertr2 = len(valid_data)

#Shuffling
order=shuffle(range(numbertr1),random_state=200)
train_data_out=train_data_out[order]
train_data1=train_data1[order,0::]
train_data_out = to_categorical( train_data_out )

order=shuffle(range(numbertr2),random_state=200)
valid_data_out=valid_data_out[order]
valid_data=valid_data[order,0::]
valid_data_out = to_categorical( valid_data_out )

classweight = np.array([1, 1.09, 1.76, 0.99, 1.6, 1.75, 5.22, 4.81, 5.88, 17.05, 11.54, 10.64, 13.96, 44.85, 77.81])
def weighted_categorical_crossentropy(weights):
	weights = K.variable(weights)
	def loss(valid_data, valid_data_out):
		valid_data_out = K.clip(valid_data_out, K.epsilon(), 1 - K.epsilon())
		loss = valid_data * K.log(valid_data_out) * (weights)
		loss = -K.sum(loss, -1)
		return loss
	return loss

import tensorflow as tf

nvar = len(variables)
#dropout = 0.15

inputs = Input(shape = (nvar,))
x = Dense(node, activation=tf.nn.relu)(inputs)
x = Dropout(dropout)(x)

for i in range(layer):
	x = Dense(node, activation=tf.nn.relu)(x)
	x = Dropout(dropout)(x)

predictions = Dense(int(numcat), activation=tf.nn.softmax)(x)
model = Model(inputs=inputs, outputs=predictions)

#modelshape = "10l_300n"
batch_size = 1024
model_output_name = 'model_ttbar_%de' %(epochs)

model.compile(loss=weighted_categorical_crossentropy(classweight),
#model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy', 'categorical_accuracy'])
#hist = model.fit(train_data1, train_data_out, batch_size=batch_size, epochs=epochs, validation_data=(valid_data,valid_data_out))
hist = model.fit(train_data1, train_data_out, batch_size=batch_size, epochs=epochs, validation_data=(valid_data,valid_data_out), class_weight = classweight)

model.summary()

timer.Stop()
rtime = timer.RealTime(); # Real time (or "wall time")
ctime = timer.CpuTime(); # CPU time
print("RealTime={0:6.2f} seconds, CpuTime={1:6.2f} seconds").format(rtime,ctime)

pred = model.predict(valid_data)
pred = np.argmax(pred, axis=1)
comp = np.argmax(valid_data_out, axis=1)

print len(valid_data)

result = pd.DataFrame({"real":comp, "pred":pred})

result_array = []
correct = 0

for i in range(numcat):
    result_real = result.loc[result['real']==i]
    temp = [(len(result_real.loc[result_real["pred"]==j])) for j in range(numcat)]
    result_array.append(temp)
    correct = correct + temp[i]
    print temp, len(result_real), temp[i]

result_array_prob = []
for i in range(numcat):
    result_real = result.loc[result['real']==i]
    temp = [(len(result_real.loc[result_real["pred"]==j])) / len(result_real) for j in range(numcat)]
    result_array_prob.append(temp)
    print temp, len(result_real), temp[i]
#print result_array_prob

#print result_array
#reconstruction efficiency
print 'calculate reco eff...'

print len(valid_data), 'correct = ', correct
#recoeff = correct/(len(valid_data))*100
recoeff = correct/(len(valid_data))*100
print 'reco eff = ', recoeff 

with open(resultTxt, "a") as f_log:
	print 'writing results...'
	f_log.write("\ntrainInput "+trainInput+'\n')
	f_log.write('Nodes: '+str(node)+'   Layers: '+str(layer)+'\nEpochs '+str(epochs)+'\nDropout '+str(dropout)+'\n')
	f_log.write('nvar: '+str(nvar)+'\n')
	f_log.write('reco eff: '+str(correct)+' / '+str(len(valid_data))+' = '+str(recoeff)+'\n')
	f_log.write('training samples '+str(len(pd_train_out))+'   validation samples '+str(len(pd_valid))+'\n')
	f_log.write('the number of each category '+str(cat_num_list)+'\n')
	f_log.write('reco eff: '+str(recoeff)+'\n')
#	f_log.write('class_weight: '+class_weight+'\n')
 
print("Plotting scores")
plt.plot(hist.history['categorical_accuracy'])
plt.plot(hist.history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train','Test'], loc='lower right')
#plt.savefig(os.path.join('fig_score_acc.pdf'))
#plt.savefig(os.path.join(resultDir,'Acc_pt20_exweight_N'+str(node)+'L'+str(layer)+'E'+str(epochs)+'D'+str(dropout)+'.pdf'))
plt.gcf().clear()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train','Test'],loc='upper right')
plt.savefig(os.path.join(resultDir,'Loss_weight_global_pt20_N'+str(node)+'L'+str(layer)+'E'+str(epochs)+'D'+str(dropout)+'.pdf'))
plt.gcf().clear()

#Heatmap
plt.rcParams['figure.figsize'] = [12, 10]
cfmt = lambda x,pos: '{:.0%}'.format(x)
heatmap = sns.heatmap(result_array_prob, annot=True, cmap='YlGnBu', fmt='.1%', annot_kws={"size":6}, vmax=1, cbar_kws={'format': FuncFormatter(cfmt)} )
#heatmap = sns.heatmap(result_array_prob, annot=True, cmap='YlGnBu', fmt='.1%', vmax=1, cbar_kws={'format': FuncFormatter(cfmt)} )
#plt.title('Heatmap', fontsize=15)
plt.xlabel('pred.', fontsize=12)
plt.ylabel('real', fontsize=12)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
plt.savefig(os.path.join(resultDir, 'HM_weight_global_pt20_N'+str(node)+'L'+str(layer)+'E'+str(epochs)+'D'+str(dropout)+'.pdf'))
plt.gcf().clear()

#timer.Stop()
#rtime = timer.RealTime(); # Real time (or "wall time")
#ctime = timer.CpuTime(); # CPU time
#print("RealTime={0:6.2f} seconds, CpuTime={1:6.2f} seconds").format(rtime,ctime)
