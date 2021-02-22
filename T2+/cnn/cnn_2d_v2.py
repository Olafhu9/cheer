from __future__ import division
import sys, os

#3os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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
from sklearn.metrics import confusion_matrix
import tensorflow as tf

nodes = 50
dropout = 0.1
epochs = 50

resultTxt = "result_0126_v2.txt"
resultDir = "/home/juhee5819/T2+/result/0126_v2"
plotName = 'N'+str(nodes)+'E'+str(epochs)+'D'+str(dropout)

# input files
input_ttbb = "/home/juhee5819/T2+/array/ttbb_2018_4f_pt20_v2.h5"
input_ttcc = "/home/juhee5819/T2+/array/ttcc_2018_4f_pt20_v2.h5"
input_ttLF = "/home/juhee5819/T2+/array/ttLF_2018_4f_pt20_v2.h5"

#input_ttbb = "/home/juhee5819/T2+/array/ttbb_2018_4f_pt20_v1_2.h5"
#input_ttcc = "/home/juhee5819/T2+/array/ttcc_2018_4f_pt20_v1_4.h5"
#input_ttLF = "/home/juhee5819/T2+/array/ttLF_2018_4f_pt20_v1_4.h5"

data_ttbb = pd.read_hdf( input_ttbb )
data_ttcc = pd.read_hdf( input_ttcc )
data_ttLF = pd.read_hdf( input_ttLF )

# drop bg jet permutation
data_ttbb = data_ttbb.drop( data_ttbb[ data_ttbb['jet_perm'] == 6 ].index )
data_ttcc = data_ttcc.drop( data_ttcc[ data_ttcc['jet_perm'] == 6 ].index )
data_ttLF = data_ttLF.drop( data_ttLF[ data_ttLF['jet_perm'] == 6 ].index )

# drop ttbb
#n_remove = len( data_ttbb ) - len( data_ttcc )
#drop_indices = np.random.choice( data_ttbb.index, n_remove, replace=False )
#data_ttbb = data_ttbb.drop( drop_indices )

# drop ttLF
#n_remove = len( data_ttLF ) - len( data_ttcc )
n_remove = len( data_ttLF ) - len( data_ttbb )
drop_indices = np.random.choice( data_ttLF.index, n_remove, replace=False )
data_ttLF = data_ttLF.drop( drop_indices )

# the number of events in each jet permutation
ttbb_jet_list = [ (len(data_ttbb.loc[data_ttbb['jet_perm']==i])) for i in range(6) ]
ttcc_jet_list = [ (len(data_ttcc.loc[data_ttcc['jet_perm']==i])) for i in range(6) ]
ttLF_jet_list = [ (len(data_ttLF.loc[data_ttLF['jet_perm']==i])) for i in range(6) ]

# merge & shuffle
data = data_ttbb.append( data_ttcc ).append( data_ttLF )
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
train_event_out = for_train.filter( items = ['event_category'] )
train_jet_out = for_train.filter( items = ['jet_perm'] )
train_event_input = for_train.filter( items = var_event )
train_jet_input = for_train.filter( items = var_jet )

valid_event_out = for_valid.filter( items = ['event_category'] )
valid_jet_out = for_valid.filter( items = ['jet_perm'] )
valid_event_input = for_valid.filter( items = var_event )
valid_jet_input = for_valid.filter( items = var_jet )

# the number of event categories & jet permutations
n_event_cat = valid_event_out.apply(set)
n_event_cat = n_event_cat.str.len()
n_event_cat = int( n_event_cat ) 

n_jet_perm = valid_jet_out.apply(set)
n_jet_perm = n_jet_perm.str.len()
n_jet_perm = int( n_jet_perm )

print 'n_event_cat ', n_event_cat, ' n_jet_perm ', n_jet_perm

## set weight
#cat_num_list = [(len(for_train.loc[for_train['category'] == i])) for i in range(n_event_cat)]
#print cat_num_list
#largest = max( cat_num_list )
#class_weight = [ (largest/cat_num_list[i]) for i in range(n_event_cat) ]

# convert from pandas to array
train_event_out = np.array( train_event_out )
train_event_out = to_categorical( train_event_out )
train_jet_out = np.array( train_jet_out )
train_jet_out = to_categorical( train_jet_out )
train_event_input = np.array( train_event_input )
train_jet_input = np.array( train_jet_input )

valid_event_out = np.array( valid_event_out )
valid_event_out = to_categorical( valid_event_out )
valid_jet_out = np.array( valid_jet_out )
valid_jet_out = to_categorical( valid_jet_out )
valid_event_input = np.array( valid_event_input )
valid_jet_input = np.array( valid_jet_input )

# reshape array
train_event_out = train_event_out.reshape( train_event_out.shape[0], 1, train_event_out.shape[1] )
train_jet_out = train_jet_out.reshape( train_jet_out.shape[0], 1, train_jet_out.shape[1] )
train_event_input = train_event_input.reshape( train_event_input.shape[0], 1, train_event_input.shape[1] )
train_jet_input = train_jet_input.reshape( train_jet_input.shape[0], 4, -1, 1 )

valid_event_out = valid_event_out.reshape( valid_event_out.shape[0], 1, valid_event_out.shape[1] )
valid_jet_out = valid_jet_out.reshape( valid_jet_out.shape[0], 1, valid_jet_out.shape[1] )
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
jets = Conv2D(64, (3,4), padding='same', kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv0')(jets)
jets = MaxPooling2D((1,2))(jets)
jets = Dropout(0.1)(jets)
jets = Conv2D(64, (3,4), padding='same', kernel_initializer='lecun_uniform', activation='relu', name='jets_conv1')(jets)
jets = MaxPooling2D((1,2))(jets)
jets = Dropout(0.1)(jets)
jets = Conv2D(64, (3,4), padding='same', kernel_initializer='lecun_uniform', activation='relu', name='jets_conv2')(jets)
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

#n_event_cat = 6

#cat_pred = Dense( n_event_cat, activation='softmax',kernel_initializer='lecun_uniform',name='cat_pred')(x)
event_cat_pred = Dense( n_event_cat, activation='softmax', kernel_initializer='lecun_uniform', name='event_prediction' )(x)
jet_perm_pred2 = Dense( n_jet_perm, activation='softmax', kernel_initializer='lecun_uniform', name='jet_prediction' )(x)
model = Model( inputs = [Input_event, Input_jet], outputs = [event_cat_pred, jet_perm_pred2] )

batch_size = 1024

model.compile( loss = 'categorical_crossentropy',optimizer = 'adam',metrics=['accuracy', 'categorical_accuracy'] )
hist = model.fit( x = [train_event_input, train_jet_input], y = [train_event_out, train_jet_out], batch_size = batch_size, epochs = epochs, validation_data = ( [valid_event_input, valid_jet_input], [valid_event_out, valid_jet_out] ) )

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# predicted label
pred = model.predict( [valid_event_input, valid_jet_input])
pred_event = pred[0]
pred_event = pred_event.reshape( pred_event.shape[0], pred_event.shape[2] )
pred_event = np.argmax( pred_event, axis=1 )

pred_jet = pred[1]
pred_jet = pred_jet.reshape( pred_jet.shape[0], pred_jet.shape[2] )
pred_jet = np.argmax( pred_jet, axis=1 )

# real label
valid_event_out_reshape = valid_event_out.reshape( valid_event_out.shape[0], valid_event_out.shape[2] )
valid_jet_out_reshape = valid_jet_out.reshape( valid_jet_out.shape[0], valid_jet_out.shape[2] )
real_event = np.argmax(valid_event_out_reshape, axis=1)
real_jet = np.argmax(valid_jet_out_reshape, axis=1)

val_event_result = pd.DataFrame( {'real_event':real_event, 'pred_event':pred_event} )
val_jet_result = pd.DataFrame( {'real_jet':real_jet, 'pred_jet':pred_jet} )

val_result = pd.concat( [val_event_result, val_jet_result], axis=1 )
val_result_ttbb = val_result.loc[ val_result['real_event']==0 ]
val_result_ttcc = val_result.loc[ val_result['real_event']==1 ]
val_result_ttLF = val_result.loc[ val_result['real_event']==2 ]

# result array for event category
result_event_array = []
result_event_array_prob = []
correct_event = 0

conf_event = confusion_matrix( val_result['real_event'], val_result['pred_event'] )
conf_jet = confusion_matrix( val_result['real_jet'], val_result['pred_jet'] )
conf_jet_ttbb = confusion_matrix( val_result_ttbb['real_jet'], val_result_ttbb['pred_jet'] )
conf_jet_ttcc = confusion_matrix( val_result_ttcc['real_jet'], val_result_ttcc['pred_jet'] )
conf_jet_ttLF = confusion_matrix( val_result_ttLF['real_jet'], val_result_ttLF['pred_jet'] )

jet_ttbb = [ (len(val_result_ttbb.loc[val_result_ttbb['real_jet']==i])) for i in range( n_jet_perm )]
jet_ttcc = [ (len(val_result_ttcc.loc[val_result_ttcc['real_jet']==i])) for i in range( n_jet_perm )]
jet_ttLF = [ (len(val_result_ttLF.loc[val_result_ttLF['real_jet']==i])) for i in range( n_jet_perm )]
#conf_event2 = confusion_matrix( val_result['real_event'], val_result['pred_event'], normalize='true' )
#print conf_event
#print conf_event2

for i in range(n_event_cat):
    result_real = val_event_result.loc[val_event_result['real_event']==i]
    temp = [(len(result_real.loc[result_real["pred_event"]==j])) for j in range(n_event_cat)]
    result_event_array.append(temp)
    correct_event = correct_event + temp[i]
    print temp, len(result_real), temp[i]

for i in range(n_event_cat):
    result_real = val_event_result.loc[val_event_result['real_event']==i]
    temp = [(len(result_real.loc[result_real["pred_event"]==j])) / len(result_real) for j in range(n_event_cat)]
    result_event_array_prob.append(temp)
    print temp, len(result_real), temp[i]

# result array for jet permutatione
result_jet_array = []
result_jet_event = []
result_jet_array_prob = []
correct_jet = 0
correct_jet_event = 0
correct_jet_ttcc = 0
correct_jet_ttLF = 0


print "hhhhhhhhh"
for i in range(n_jet_perm):
    result_real = val_jet_result.loc[val_jet_result['real_jet']==i]
    temp = [(len(result_real.loc[result_real["pred_jet"]==j])) for j in range(n_jet_perm)]
    result_jet_array.append(temp)
    correct_jet = correct_jet + temp[i]
    print temp, len(result_real), temp[i]

for i in range(n_jet_perm):
    result_real = val_jet_result.loc[val_jet_result['real_jet']==i]
    temp = [(len(result_real.loc[result_real["pred_jet"]==j])) / len(result_real) for j in range(n_jet_perm)]
    result_jet_array_prob.append(temp)
    print temp, len(result_real), temp[i]



# calculate accuracy
acc_event = correct_event/(len(for_valid))*100
acc_jet = correct_jet/(len(for_valid))*100
print 'acc_event ',acc_event, '   acc_jet ', acc_jet 

print('Writing results')
# log
with open(resultTxt, "a") as f_log:
    #f_log.write("\ntrainInput "+trainInput+'\n')
    f_log.write('Nodes: '+str(nodes)+'\nEpochs '+str(epochs)+'\nDropout '+str(dropout)+'\n')
    f_log.write('nvar: '+str(var_event+var_jet)+'\n')
    f_log.write('event category accuracy: '+str(correct_event)+'/'+str(len(for_valid))+'='+str(acc_event)+'\n')
    f_log.write('jet permutation accuracy: '+str(correct_jet)+'/'+str(len(for_valid))+'='+str(acc_jet)+'\n')
    f_log.write('training samples '+str(len(for_train))+'   validation samples '+str(len(for_valid))+'\n')
    f_log.write('ttbb jet perm list '+str(ttbb_jet_list)+'\n')
    f_log.write('ttcc jet perm list '+str(ttcc_jet_list)+'\n')
    f_log.write('ttLF jet perm list '+str(ttLF_jet_list)+'\n')

    #f_log.write('the number of each category '+str(cat_num_list)+'\n')
    #f_log.write('class_weight '+str( class_weight )+'\n')

print("Plotting scores")
# loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train','Test'],loc='upper right')
#plt.savefig(os.path.join(resultDir,'Loss_N'+str(nodes)+'E'+str(epochs)+'D'+str(dropout)+'.pdf'))
plt.savefig(os.path.join(resultDir,'Loss_'+plotName+'_eff'+str(acc_event)+'_'+str(acc_jet)+'.pdf'))
plt.show()
plt.gcf().clear()

# loss for event category
plt.plot(hist.history['event_prediction_loss'])
plt.plot(hist.history['val_event_prediction_loss'])
plt.ylabel('Event Prediction Loss')
plt.xlabel('Epochs')
plt.legend(['Train','Test'],loc='upper right')
#plt.savefig(os.path.join(resultDir,'Loss_N'+str(nodes)+'E'+str(epochs)+'D'+str(dropout)+'.pdf'))
plt.savefig(os.path.join(resultDir,'Loss(e)_'+plotName+'_eff'+str(acc_event)+'.pdf'))
plt.show()
plt.gcf().clear()

# loss for jet permutation
plt.plot(hist.history['jet_prediction_loss'])
plt.plot(hist.history['val_jet_prediction_loss'])
plt.ylabel('Jet Prediction Loss')
plt.xlabel('Epochs')
plt.legend(['Train','Test'],loc='upper right')
#plt.savefig(os.path.join(resultDir,'Loss_N'+str(nodes)+'E'+str(epochs)+'D'+str(dropout)+'.pdf'))
plt.savefig(os.path.join(resultDir,'Loss(j)_'+plotName+'_eff'+str(acc_jet)+'.pdf'))
plt.show()
plt.gcf().clear()

# Heatmap for event category
plt.rcParams['figure.figsize'] = [7.5, 6]
cfmt = lambda x,pos: '{:.0%}'.format(x)
heatmap = sns.heatmap(result_event_array_prob, annot=True, cmap='YlGnBu', fmt='.1%', annot_kws={"size":12}, vmax=1, cbar_kws={'format': FuncFormatter(cfmt)} )
#plt.title('Heatmap', fontsize=15)
plt.xlabel('predicted event cat.', fontsize=12)
plt.ylabel('real event cat.', fontsize=12)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
plt.savefig(os.path.join(resultDir,'HM(e)_'+plotName+'_eff'+str(acc_event)+'.pdf'))
#plt.savefig(os.path.join(resultDir,'HM_N'+str(nodes)+'E'+str(epochs)+'D'+str(dropout)+'.pdf'))
#plt.savefig(os.path.join(resultDir,'HM_'+plotName+'_eff'+str(recoeff)+'.pdf'))
plt.show()

# Heatmap for jet permutation
plt.rcParams['figure.figsize'] = [7.5, 6]
cfmt = lambda x,pos: '{:.0%}'.format(x)
heatmap = sns.heatmap(result_jet_array_prob, annot=True, cmap='YlGnBu', fmt='.1%', annot_kws={"size":12}, vmax=1, cbar_kws={'format': FuncFormatter(cfmt)} )
#plt.title('Heatmap', fontsize=15)
plt.xlabel('predicted jet perm.', fontsize=12)
plt.ylabel('real jet perm.', fontsize=12)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
plt.savefig(os.path.join(resultDir,'HM(j)_'+plotName+'_eff'+str(acc_jet)+'.pdf'))
#plt.savefig(os.path.join(resultDir,'HM_N'+str(nodes)+'E'+str(epochs)+'D'+str(dropout)+'.pdf'))
#plt.savefig(os.path.join(resultDir,'HM_'+plotName+'_eff'+str(recoeff)+'.pdf'))
plt.show()

# Heatmap for jet permutation
plt.rcParams['figure.figsize'] = [7.5, 6]
cfmt = lambda x,pos: '{:.0%}'.format(x)
heatmap = sns.heatmap(conf_jet_ttbb/jet_ttbb, annot=True, cmap='YlGnBu', fmt='.1%', annot_kws={"size":12}, vmax=1, cbar_kws={'format': FuncFormatter(cfmt)} )
plt.title('ttbb Jet Permutation', fontsize=15)
plt.xlabel('predicted jet perm.', fontsize=12)
plt.ylabel('real jet perm.', fontsize=12)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
plt.savefig(os.path.join(resultDir,'HM_ttbb'+'.pdf'))
plt.show()

plt.rcParams['figure.figsize'] = [7.5, 6]
cfmt = lambda x,pos: '{:.0%}'.format(x)
heatmap = sns.heatmap(conf_jet_ttcc/jet_ttcc, annot=True, cmap='YlGnBu', fmt='.1%', annot_kws={"size":12}, vmax=1, cbar_kws={'format': FuncFormatter(cfmt)} )
plt.title('ttcc Jet Permutation', fontsize=15)
plt.xlabel('predicted jet perm.', fontsize=12)
plt.ylabel('real jet perm.', fontsize=12)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
plt.savefig(os.path.join(resultDir,'HM_ttcc'+'.pdf'))
plt.show()

plt.rcParams['figure.figsize'] = [7.5, 6]
cfmt = lambda x,pos: '{:.0%}'.format(x)
heatmap = sns.heatmap(conf_jet_ttLF/jet_ttLF, annot=True, cmap='YlGnBu', fmt='.1%', annot_kws={"size":12}, vmax=1, cbar_kws={'format': FuncFormatter(cfmt)} )
plt.title('ttLF Jet Permutation', fontsize=15)
plt.xlabel('predicted jet perm.', fontsize=12)
plt.ylabel('real jet perm.', fontsize=12)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
plt.savefig(os.path.join(resultDir,'HM_ttLF'+'.pdf'))
plt.show()

# Heatmap for jet permutation


