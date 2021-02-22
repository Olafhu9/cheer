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
from keras.layers import Input, Dense, Activation, Dropout, add, LSTM, Concatenate, MaxPooling2D
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D, Conv2D
from array import array
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers.core import Reshape
from keras.utils.vis_utils import plot_model
import tensorflow as tf

nodes = 50
dropout = 0.1
epochs = 5

resultTxt = "result_0111_v1.txt"
resultDir = "/home/juhee5819/T2+/result/0111_v1/"
plotName = 'N'+str(nodes)+'E'+str(epochs)+'D'+str(dropout)

# input files
input_ttbb = "/home/juhee5819/T2+/array/ttbb_2018_4f_pt20_fin.h5"
input_ttHbb = "/home/juhee5819/T2+/array/ttbb_2018_4f_pt20_ttHbb.h5"
input_ttcc = "/home/juhee5819/T2+/array/ttbb_2018_4f_pt20_ttcc.h5"
input_ttLF = "/home/juhee5819/T2+/array/ttbb_2018_4f_pt20_ttLF.h5"

data_ttbb = pd.read_hdf( input_ttbb )
data_ttHbb = pd.read_hdf( input_ttHbb )
data_ttcc = pd.read_hdf( input_ttcc )
data_ttLF = pd.read_hdf( input_ttLF )

data = data_ttbb.append( data_ttHbb ).append( data_ttcc ).append( data_ttLF )
#data = data.drop(data[data['category']==6].index)

# shuffle
data = data.sample(frac=1).reset_index(drop=True)

num_data = len( data )

# Split between training set and validation set
train_set = 0.7
for_valid = data[int( train_set * num_data ) : num_data]
for_train = data[0 : int( train_set * num_data )]

# event variables
var_event = ['nbjets_m', 'ngoodjets', 'St', 'Ht', 'lepton_pt', 'lepton_eta', 'lepton_e', 'MET', 'MET_phi', 'nulep_pt', "dRnulep12", "dRnulep13", "dRnulep14", "dRnulep23", "dRnulep24", "dRnulep34", "dEta12", "dEta13", "dEta14", "dEta23", "dEta24", "dEta34", "dPhi12", "dPhi13", "dPhi14", "dPhi23", "dPhi24", "dPhi34", "invm12", "invm13", "invm14", "invm23", "invm24", "invm34", "dR12", "dR13", "dR14", "dR23", "dR24", "dR34"]
# jet variables
var_jet = ["jet1_pt", "jet1_eta", "jet1_e", "jet1_m", "jet1_btag", "jet1_CvsB", "jet1_CvsL", "dRlep1", "dRnu1", "dRnulep1", "invmlep1", "invmnu1", "jet2_pt", "jet2_eta", "jet2_e", "jet2_m", "jet2_btag", "jet2_CvsB", "jet2_CvsL", "dRlep2", "dRnu2", "dRnulep2", "invmlep2", "invmnu2", "jet3_pt", "jet3_eta", "jet3_e", "jet3_m", "jet3_btag", "jet3_CvsB", "jet3_CvsL", "dRlep3", "dRnu3", "dRnulep3", "invmlep3", "invmnu3", "jet4_pt", "jet4_eta", "jet4_e", "jet4_m", "jet4_btag", "jet4_CvsB", "jet4_CvsL", "dRlep4", "dRnu4", "dRnulep4", "invmlep4", "invmnu4"]

# inputs and outputs
train_data_out = for_train.filter( items = ['category'] )
train_event_input = for_train.filter( items = var_event )
train_jet_input = for_train.filter( items = var_jet )

valid_data_out = for_valid.filter( items = ['category'] )
valid_event_input = for_valid.filter( items = var_event )
valid_jet_input = for_valid.filter( items = var_jet )

catlist = valid_data_out.apply(set)
n_cat = catlist.str.len()
n_cat = int( n_cat )
print 'n_cat', n_cat

# set weight
cat_num_list = [(len(for_train.loc[for_train['category'] == i])) for i in range(n_cat)]
print cat_num_list
largest = max( cat_num_list )
class_weight = [ (largest/cat_num_list[i]) for i in range(n_cat) ]

# convert from pandas to array
train_data_out = np.array( train_data_out )
train_data_out = to_categorical( train_data_out )

train_event_input = np.array( train_event_input )
train_jet_input = np.array( train_jet_input )

valid_data_out = np.array( valid_data_out )
valid_data_out = to_categorical( valid_data_out )

valid_event_input = np.array( valid_event_input )
valid_jet_input = np.array( valid_jet_input )

# reshape array
train_data_out = train_data_out.reshape( train_data_out.shape[0], 1, train_data_out.shape[1] )
train_event_input = train_event_input.reshape( train_event_input.shape[0], 1, train_event_input.shape[1] )
train_jet_input = train_jet_input.reshape( train_jet_input.shape[0], 4, -1, 1 )

valid_data_out = valid_data_out.reshape( valid_data_out.shape[0], 1, valid_data_out.shape[1] )
valid_event_input = valid_event_input.reshape( valid_event_input.shape[0], 1, valid_event_input.shape[1] )
valid_jet_input = valid_jet_input.reshape( valid_jet_input.shape[0], 4, -1, 1 )

# Inputs
Input_event = Input( shape = (train_event_input.shape[1], train_event_input.shape[2]) )
Input_jet = Input( shape = (train_jet_input.shape[1], train_jet_input.shape[2], train_jet_input.shape[3]) )

# BatchNormalization
event_info = BatchNormalization( name = 'event_input_batchnorm' )(Input_event)
jets = BatchNormalization( name = 'jet_input_batchnorm' )(Input_jet)

# Dense for event
event_info = Dense(100, activation='relu')(event_info)
event_info = Dropout(0.1)(event_info)
event_info = Dense(100, activation='relu')(event_info)
event_info = Dropout(0.1)(event_info)
event_info = Dense(100, activation='relu')(event_info)
event_info = Dropout(0.1)(event_info)

from keras.layers.convolutional import Conv1D, Conv2D
# CNN for jets
#jets = Conv2D(64, (3,3), padding='same', kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv0')(jets)
jets = Conv2D(64, (3,3), padding='same', kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv0')(jets)
jets = MaxPooling2D((1,2))(jets)
jets = Dropout(0.1)(jets)
jets = Conv2D(64, (3,3), padding='same', kernel_initializer='lecun_uniform', activation='relu', name='jets_conv1')(jets)
jets = MaxPooling2D((1,2))(jets)
jets = Dropout(0.1)(jets)
jets = Conv2D(64, (3,3), padding='same', kernel_initializer='lecun_uniform', activation='relu', name='jets_conv2')(jets)
jets = MaxPooling2D((2,2))(jets)
jets = Dropout(0.1)(jets)

#jets = Conv2D(64, (3,3), padding='same', kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv0')(jets)
##jets = MaxPooling2D((1,2))(jets)
#jets = Dropout(0.1)(jets)
#jets = Conv2D(64, (3,3), padding='same', kernel_initializer='lecun_uniform', activation='relu')(jets)
#jets = MaxPooling2D((1,2))(jets)
#jets = Dropout(0.1)(jets)
#jets = Conv2D(64, (3,3), padding='same', kernel_initializer='lecun_uniform', activation='relu')(jets)
#jets = MaxPooling2D((1,2))(jets)
#jets = Dropout(0.1)(jets)
#jets = Conv2D(64, (3,3), padding='same', kernel_initializer='lecun_uniform', activation='relu')(jets)
#jets = MaxPooling2D((2,2))(jets)
#jets = Dropout(0.1)(jets)

#jets = Conv2D(64, (3,3), padding='same',  kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv1')(jets)
jets = Reshape( (1, -1) )(jets)
jets = LSTM(50, go_backwards=True, implementation=2, name='jets_lstm', return_sequences=True)(jets)
#jets = LSTM(50, go_backwards=True, implementation=2, name='jets_lstm2', return_sequences=True)(jets)

## CNN for leptons
#leptons  = Conv1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='leptons_conv0')(leptons)
#leptons = LSTM(25, go_backwards=True, implementation=2, name='leptons_lstm', return_sequences=True)(leptons)

# Concatenate
x = Concatenate()( [event_info, jets] )
x = Dense(nodes, activation='relu',kernel_initializer='lecun_uniform')(x)
x = Dropout(dropout)(x)
#x = Dense(nodes, activation='relu',kernel_initializer='lecun_uniform')(x)
#x = Dropout(dropout)(x)
#x = Dense(nodes, activation='relu',kernel_initializer='lecun_uniform')(x)
#x = Dropout(dropout)(x)
#x = Dense(nodes, activation='relu',kernel_initializer='lecun_uniform')(x)
#x = Dropout(dropout)(x)

#n_cat = 6

cat_pred = Dense( n_cat, activation='softmax',kernel_initializer='lecun_uniform',name='cat_pred')(x)
model = Model( inputs = [Input_event, Input_jet], outputs = cat_pred)

batch_size = 1024

model.compile( loss = 'categorical_crossentropy',optimizer = 'adam',metrics=['accuracy', 'categorical_accuracy'] )
hist = model.fit( x = [train_event_input, train_jet_input], y = train_data_out, batch_size = batch_size, epochs = epochs, validation_data = ( [valid_event_input, valid_jet_input], valid_data_out), class_weight=class_weight)

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#pre = model.predict(valid_data)
pred = model.predict( [valid_event_input, valid_jet_input])
pred = pred.reshape( pred.shape[0], pred.shape[2] )
pred = np.argmax(pred, axis=1)
valid_data_out_reshape = valid_data_out.reshape( valid_data_out.shape[0], valid_data_out.shape[2] )
comp = np.argmax(valid_data_out_reshape, axis=1)

val_result = pd.DataFrame({"real":comp, "pred":pred})#d event variables

result_array = []
correct = 0

for i in range(n_cat):
    result_real = val_result.loc[val_result['real']==i]
    temp = [(len(result_real.loc[result_real["pred"]==j])) for j in range(n_cat)]
    result_array.append(temp)
    correct = correct + temp[i]
    print temp, len(result_real), temp[i]

result_array_prob = []
for i in range(n_cat):
    result_real = val_result.loc[val_result['real']==i]
    temp = [(len(result_real.loc[result_real["pred"]==j])) / len(result_real) for j in range(n_cat)]
    result_array_prob.append(temp)
    print temp, len(result_real), temp[i]
from keras.layers.convolutional import Conv1D, Conv2D

print("Plotting scores")

recoeff = correct/(len(for_valid))*100
print 'recoeff', recoeff

with open(resultTxt, "a") as f_log:
    print 'writing results...'
    #f_log.write("\ntrainInput "+trainInput+'\n')
    f_log.write('Nodes: '+str(nodes)+'\nEpochs '+str(epochs)+'\nDropout '+str(dropout)+'\n')
    f_log.write('nvar: '+str(var_event+var_jet)+'\n')
    f_log.write('reco eff: '+str(correct)+' / '+str(len(for_valid))+' = '+str(recoeff)+'\n')
    f_log.write('training samples '+str(len(for_train))+'   validation samples '+str(len(for_valid))+'\n')
    f_log.write('the number of each category '+str(cat_num_list)+'\n')
    f_log.write('class_weight '+str( class_weight )+'\n')
    f_log.write('reco eff: '+str(recoeff)+'\n')

# Loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train','Test'],loc='upper right')
#plt.savefig(os.path.join(resultDir,'Loss_N'+str(nodes)+'E'+str(epochs)+'D'+str(dropout)+'.pdf'))
plt.savefig(os.path.join(resultDir,'Loss_'+plotName+'_eff'+str(recoeff)+'.pdf'))
plt.show()
plt.gcf().clear()

# Heatmap
plt.rcParams['figure.figsize'] = [7.5, 6]
cfmt = lambda x,pos: '{:.0%}'.format(x)
heatmap = sns.heatmap(result_array_prob, annot=True, cmap='YlGnBu', fmt='.1%', annot_kws={"size":12}, vmax=1, cbar_kws={'format': FuncFormatter(cfmt)} )
#plt.title('Heatmap', fontsize=15)
plt.xlabel('pred.', fontsize=12)
plt.ylabel('real', fontsize=12)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
#plt.savefig(os.path.join(resultDir,'HM_N'+str(nodes)+'E'+str(epochs)+'D'+str(dropout)+'.pdf'))
plt.savefig(os.path.join(resultDir,'HM_'+plotName+'_eff'+str(recoeff)+'.pdf'))
plt.show()




