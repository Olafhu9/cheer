from __future__ import division
import sys, os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
from sklearn.utils import shuffle
import re
import string
import math
from ROOT import *
import ROOT
import numpy as np
from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation, Dropout, add, LSTM, Concatenate, MaxPooling2D, Flatten, TimeDistributed
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
dropout = 0.2
epochs = 130

resultTxt = "result_0330_var.txt"
resultDir = "/home/juhee5819/cheer/T2+/result/0330"
#plotName = 'N'+str(nodes)+'E'+str(epochs)+'D'+str(dropout)
plotName2 = '1d_v6_2'
ver = '1d_v6_2'

if not os.path.exists( resultDir+'/'+ver ):
	os.makedirs( resultDir+'/'+ver )
test = os.listdir( resultDir+'/'+ver )
for item in test:
    if item.endswith(".pdf") or item.endswith(".h5") or item.endswith("log"):
	    os.remove(os.path.join( resultDir+'/'+ver, item ))

# input files
input_ttbb = "/home/juhee5819/cheer/T2+/array/ttbb_2018_pt20_v6.h5"
input_ttcc = "/home/juhee5819/cheer/T2+/array/ttcc_2018_pt20_v6.h5"
input_ttLF = "/home/juhee5819/cheer/T2+/array/ttlf_2018_pt20_v6.h5"

data_ttbb = pd.read_hdf( input_ttbb )
data_ttcc = pd.read_hdf( input_ttcc )
data_ttLF = pd.read_hdf( input_ttLF )

# drop bg jet permutation
data_ttbb = data_ttbb.drop( data_ttbb[ data_ttbb['jet_perm'] == 15 ].index )
data_ttcc = data_ttcc.drop( data_ttcc[ data_ttcc['jet_perm'] == 15 ].index )
data_ttLF = data_ttLF.drop( data_ttLF[ data_ttLF['jet_perm'] == 15 ].index )

# drop ttbb
#n_remove = len( data_ttbb ) - len( data_ttcc )
#drop_indices = np.random.choice( data_ttbb.index, n_remove, replace=False )
#data_ttbb = data_ttbb.drop( drop_indices )

# drop ttcc
#n_remove = len( data_ttcc ) - len( data_ttbb )
#drop_indices = np.random.choice( data_ttcc.index, n_remove, replace=False )
#data_ttcc = data_ttcc.drop( drop_indices )

# drop ttLF
#n_remove = len( data_ttLF ) - len( data_ttcc )
#n_remove = len( data_ttLF ) - len( data_ttbb )
#drop_indices = np.random.choice( data_ttLF.index, n_remove, replace=False )
#data_ttLF = data_ttLF.drop( drop_indices )

# merge & shuffle
data = data_ttbb.append( data_ttcc ).append( data_ttLF )
data = data.sample(frac=1).reset_index(drop=True)
num_data = len( data )

# Split between training set and validation set
train_set = 0.7
for_valid = data[int( train_set * num_data ) : num_data]
for_train = data[0 : int( train_set * num_data )]

# event variables
var_event = ["ngoodjets", "nulep_pt", "St", "Ht", "ncjets_m", "nbjets_m", "lepton_pt", "lepton_eta", "lepton_e", "MET", "MET_phi", "dR12", "dR13", "dR14", "dR15", "dR16", "dR23", "dR24", "dR25", "dR26", "dR34", "dR35", "dR36", "dR45", "dR46", "dR56", "dEta12", "dEta13", "dEta14", "dEta15", "dEta16", "dEta23", "dEta24", "dEta25", "dEta26", "dEta34", "dEta35", "dEta36", "dEta45", "dEta46", "dEta56", "dPhi12", "dPhi13", "dPhi14", "dPhi15", "dPhi16", "dPhi23", "dPhi24", "dPhi25", "dPhi26", "dPhi34", "dPhi35", "dPhi36", "dPhi45", "dPhi46", "dPhi56", "invm12", "invm13", "invm14", "invm15", "invm16", "invm23", "invm24", "invm25", "invm26", "invm34", "invm35", "invm36", "invm45", "invm46", "invm56", "dRnulep12", "dRnulep13", "dRnulep14", "dRnulep15", "dRnulep16", "dRnulep23", "dRnulep24", "dRnulep25", "dRnulep26", "dRnulep34", "dRnulep35", "dRnulep36", "dRnulep45", "dRnulep46", "dRnulep56"]
# jet variables + nSV, nCpfc
var_jet = ["jet1_pt", "jet1_eta", "jet1_e", "jet1_m", "jet1_btag", "jet1_CvsB", "jet1_CvsL", "jet1_nSV", "jet1_nCpfC", "dRlep1", "dRnu1", "dRnulep1", "invmlep1", "invmnu1", "jet2_pt", "jet2_eta", "jet2_e", "jet2_m", "jet2_btag", "jet2_CvsB", "jet2_CvsL", "jet2_nSV", "jet2_nCpfC", "dRlep2", "dRnu2", "dRnulep2", "invmlep2", "invmnu2", "jet3_pt", "jet3_eta", "jet3_e", "jet3_m", "jet3_btag", "jet3_CvsB", "jet3_CvsL", "jet3_nSV", "jet3_nCpfC", "dRlep3", "dRnu3", "dRnulep3", "invmlep3", "invmnu3", "jet4_pt", "jet4_eta", "jet4_e", "jet4_m", "jet4_btag", "jet4_CvsB", "jet4_CvsL", "jet4_nSV", "jet4_nCpfC", "dRlep4", "dRnu4", "dRnulep4", "invmlep4", "invmnu4", "jet5_pt", "jet5_eta", "jet5_e", "jet5_m", "jet5_btag", "jet5_CvsB", "jet5_CvsL", "jet5_nSV", "jet5_nCpfC", "dRlep5", "dRnu5", "dRnulep5", "invmlep5", "invmnu5", "jet6_pt", "jet6_eta", "jet6_e", "jet6_m", "jet6_btag", "jet6_CvsB", "jet6_CvsL", "jet6_nSV", "jet6_nCpfC", "dRlep6", "dRnu6", "dRnulep6", "invmlep6", "invmnu6"]
#
# sv variables
var_sv = ["SVpt", "SVeta", "SVPhi", "SVdEta", "SVdPhi", "SVdR", "SVm", "SVdxy", "SVdxysig", "SVd3d", "SVd3dsig", "SVcosthetaPv", "SVntracks"] 

#var_cpfc = ["CpfCpt", "CpfCeta", "CpfCPhi", "CpfCptrel", "CpfCerel", "CpfCdEta", "CpfCdPhi", "CpfCdR", "CpfCdxy", "CpfCdxysig", "CpfCvtxass", "CpfCfrompv", "CpfCdz", "CpfCvtxdEta", "CpfCvtxdPhi", "CpfCcharge", "CpfClostinhits", "CpfCnpixhits", "CpfCquality", "CpfCmindRsv" "Trackmom", "Tracketa", "TrackdEta", "Trackptrel", "Trackppar", "TrackdR", "Tracksip2dval", "Tracksip2dsig", "Tracksip3dval", "Tracksip3dsig", "Trackjetdistval", "Trackjetdistsig"]
var_cpfc = ["CpfCpt", "CpfCeta", "CpfCPhi", "CpfCptrel", "CpfCerel", "CpfCdEta", "CpfCdPhi", "CpfCdR", "CpfCdxy", "CpfCdxysig", "CpfCvtxass", "CpfCfrompv", "CpfCdz", "CpfCvtxdEta", "CpfCvtxdPhi", "CpfCcharge", "CpfClostinhits", "CpfCnpixhits", "CpfCquality", "CpfCmindRsv", "Trackmom", "Tracketa", "TrackdEta", "Trackptrel", "Trackppar", "TrackdR", "Tracksip2dval", "Tracksip2dsig", "Tracksip3dval", "Tracksip3dsig", "Trackjetdistval"]



# inputs and outputs
train_event_out = for_train.filter( items = ['event_category'] )
train_jet_out = for_train.filter( items = ['jet_perm'] )
train_event_input = for_train.filter( items = var_event )
train_jet_input = for_train.filter( items = var_jet )
train_sv_input = for_train.filter( items = var_sv )
train_cpfc_input = for_train.filter( items = var_cpfc )

valid_event_out = for_valid.filter( items = ['event_category'] )
valid_jet_out = for_valid.filter( items = ['jet_perm'] )
valid_event_input = for_valid.filter( items = var_event )
valid_jet_input = for_valid.filter( items = var_jet )
valid_sv_input = for_valid.filter( items = var_sv )
valid_cpfc_input = for_valid.filter( items = var_cpfc )

# the number of event categories & jet permutations
n_event_cat = valid_event_out.apply(set)
n_event_cat = int( n_event_cat.str.len() )
#n_event_cat = int( n_event_cat ) 
n_jet_perm = valid_jet_out.apply(set)
n_jet_perm = int( n_jet_perm.str.len() )
#n_jet_perm = int( n_jet_perm )
print 'n_event_cat ', n_event_cat, ' n_jet_perm ', n_jet_perm

# the number of events in each jet permutation
ttbb_jet_list = [ int((len(data_ttbb.loc[data_ttbb['jet_perm']==i]))) for i in range(n_jet_perm) ]
ttcc_jet_list = [ int((len(data_ttcc.loc[data_ttcc['jet_perm']==i]))) for i in range(n_jet_perm) ]
ttLF_jet_list = [ int((len(data_ttLF.loc[data_ttLF['jet_perm']==i]))) for i in range(n_jet_perm) ]

tot_ttbb = sum( ttbb_jet_list )
tot_ttcc = sum( ttcc_jet_list )
tot_ttLF = sum( ttLF_jet_list )

## set weight
#cat_num_list = [(len(for_train.loc[for_train['category'] == i])) for i in range(n_event_cat)]
#print cat_num_list
#largest = max( cat_num_list )
#class_weight = [ (largest/cat_num_list[i]) for i in range(n_event_cat) ]

# convert from pandas to array
train_event_out = to_categorical( np.array( train_event_out ) )
train_jet_out = to_categorical( np.array( train_jet_out ) )
train_event_input = np.array( train_event_input )
train_jet_input = np.array( train_jet_input )
train_sv_input = np.array( train_sv_input )
train_cpfc_input = np.array( train_cpfc_input )

valid_event_out = to_categorical( np.array( valid_event_out ) )
valid_jet_out = to_categorical( np.array( valid_jet_out ) )
valid_event_input = np.array( valid_event_input )
valid_jet_input = np.array( valid_jet_input )
valid_sv_input = np.array( valid_sv_input )
valid_cpfc_input = np.array( valid_cpfc_input )

# reshape jet input array
train_jet_input = train_jet_input.reshape( train_jet_input.shape[0], 6, -1 )
valid_jet_input = valid_jet_input.reshape( valid_jet_input.shape[0], 6, -1 )

# reshape sv input array (n_events x n_jets x n_sv_features)
train_sv_input = np.concatenate( np.concatenate( train_sv_input ) )
train_sv_input = train_sv_input.reshape( len(for_train), len(var_sv), -1 )
train_sv_input = np.transpose( train_sv_input, (0, 2, 1) )
valid_sv_input = np.concatenate( np.concatenate( valid_sv_input ) )
valid_sv_input = valid_sv_input.reshape( len(for_valid), len(var_sv), -1 )
valid_sv_input = np.transpose( valid_sv_input, (0, 2, 1) )

# reshape cpf  input array (n_events x n_jetx x n_cpfc x n_cpfcan_features)
train_cpfc_input = np.concatenate( np.concatenate( np.concatenate( train_cpfc_input ) ) )
train_cpfc_input = train_cpfc_input.reshape( len(for_train), len(var_cpfc), 6, -1 )
train_cpfc_input = np.transpose( train_cpfc_input, (0, 2, 3, 1))
valid_cpfc_input = np.concatenate( np.concatenate( np.concatenate( valid_cpfc_input ) ) )
valid_cpfc_input = valid_cpfc_input.reshape( len(for_valid), len(var_cpfc), 6, -1 )
valid_cpfc_input = np.transpose( valid_cpfc_input, (0, 2, 3, 1))


# Inputs
Inputs = [ Input( shape=(train_event_input.shape[1],) ), Input( shape=(train_jet_input.shape[1], train_jet_input.shape[2]), ), Input( shape=(train_sv_input.shape[1], train_sv_input.shape[2],)), Input( shape=(train_cpfc_input.shape[1], train_cpfc_input.shape[2], train_cpfc_input.shape[3])) ]

# BatchNormalization
event_info = BatchNormalization( name = 'event_input_batchnorm' )(Inputs[0])
jets = BatchNormalization( name = 'jet_input_batchnorm' )(Inputs[1])
svtx = BatchNormalization( name = 'svtx_input_batchnorm' )(Inputs[2])
cpf = BatchNormalization( name = 'cpf_input_batchnorm' )(Inputs[3])

# Dense for event
event_info = Dense(100, activation='relu')(event_info)
event_info = Dropout(dropout)(event_info)
event_info = Dense(100, activation='relu')(event_info)
event_info = Dropout(dropout)(event_info)
event_info = Dense(100, activation='relu')(event_info)
event_info = Dropout(dropout)(event_info)
#event_info = Dense(100, activation='relu')(event_info)
#event_info = Dropout(dropout)(event_info)
#event_info = Dense(100, activation='relu')(event_info)
#event_info = Dropout(dropout)(event_info)

jets = Conv1D( 64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv0')(jets)
jets = Dropout(dropout)(jets)
jets = Conv1D( 64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv1')(jets)
jets = Dropout(dropout)(jets)
jets = Conv1D( 64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv2')(jets)
jets = Dropout(dropout)(jets)
jets = LSTM(50, go_backwards=True, implementation=2, name='jets_lstm')(jets)

svtx = Conv1D( 64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='svtx_conv0')(svtx)
svtx = Dropout(dropout)(svtx)
svtx = Conv1D( 64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='svtx_conv1')(svtx)
svtx = Dropout(dropout)(svtx)
svtx = Conv1D( 64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='svtx_conv2')(svtx)
svtx = Dropout(dropout)(svtx)
svtx = LSTM(50, go_backwards=True, implementation=2, name='svtx_lstm')(svtx)


cpf = Conv2D( 64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
cpf = Dropout(dropout)(cpf)
cpf = Conv2D( 64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
cpf = Dropout(dropout)(cpf)
cpf = Conv2D( 64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
cpf = Dropout(dropout)(cpf)
cpf = TimeDistributed( Conv2D( 64, 1, kernel_initializer='lecun_uniform', activation='relu', name='cpf_timedist') )(cpf)

cpf = Reshape( (-1, 1) )(cpf)
cpf = LSTM(50, go_backwards=True, implementation=2, name='cpf_lstm')(cpf)






# Concatenate
x = Concatenate()( [event_info, jets] )
x = Dense(nodes, activation='relu',kernel_initializer='lecun_uniform')(x)
x = Dropout(dropout)(x)

event_cat_pred = Dense( n_event_cat, activation='softmax', kernel_initializer='lecun_uniform', name='event_prediction' )(x)
jet_perm_pred2 = Dense( n_jet_perm, activation='softmax', kernel_initializer='lecun_uniform', name='jet_prediction' )(x)
model = Model( inputs=Inputs, outputs = [event_cat_pred, jet_perm_pred2] )

batch_size = 1024

modelfile = 'model_{epoch:02d}_{val_categorical_accuracy:.4f}.h5'
checkpoint = ModelCheckpoint(resultDir+'/'+ver+'/'+modelfile, monitor='val_categorical_accuracy', verbose=1, save_best_only=True)


model.compile( loss = 'categorical_crossentropy',optimizer = 'adam',metrics=['accuracy', 'categorical_accuracy'] )
hist = model.fit( x = [train_event_input, train_jet_input, train_sv_input, train_cpfc_input], y = [train_event_out, train_jet_out], batch_size = batch_size, epochs = epochs, validation_data = ( [valid_event_input, valid_jet_input, valid_sv_input, valid_cpfc_input], [valid_event_out, valid_jet_out] ) )
#hist = model.fit( x = [train_event_input, train_jet_input], y = [train_event_out, train_jet_out], batch_size = batch_size, epochs = epochs, validation_data = ( [valid_event_input, valid_jet_input], [valid_event_out, valid_jet_out] ) )
#hist = model.fit( x = [train_event_input, train_jet_input], y = [train_event_out, train_jet_out], batch_size = batch_size, epochs = epochs, validation_data = ( [valid_event_input, valid_jet_input], [valid_event_out, valid_jet_out] ), callbacks=checkpoint )

#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# predicted label
pred = model.predict( [valid_event_input, valid_jet_input, valid_sv_input, valid_cpfc_input])
pred_event = pred[0]
pred_jet = pred[1]

# probabilities
prob_ttbb = pred_event[:,0]
prob_ttcc = pred_event[:,1]
prob_ttLF = pred_event[:,2]

# one-hot encoding
pred_event = np.argmax( pred_event, axis=1 )
pred_jet = np.argmax( pred_jet, axis=1 )

# real label
real_event = np.argmax(valid_event_out, axis=1)
real_jet = np.argmax(valid_jet_out, axis=1)

# pred vs real array
val_result = pd.DataFrame( {'real_event':real_event, 'pred_event':pred_event, 'prob_ttbb':prob_ttbb, 'prob_ttcc':prob_ttcc, 'prob_ttLF':prob_ttLF, 'real_jet':real_jet, 'pred_jet':pred_jet} )

#val_result = pd.concat( [val_event_result, val_jet_result], axis=1 )
val_result_ttbb = val_result.loc[ val_result['real_event']==0 ]
val_result_ttcc = val_result.loc[ val_result['real_event']==1 ]
val_result_ttLF = val_result.loc[ val_result['real_event']==2 ]

conf_event = confusion_matrix( val_result['real_event'], val_result['pred_event'] )
conf_jet = confusion_matrix( val_result['real_jet'], val_result['pred_jet'] )
conf_jet_ttbb = confusion_matrix( val_result_ttbb['real_jet'], val_result_ttbb['pred_jet'] )
conf_jet_ttcc = confusion_matrix( val_result_ttcc['real_jet'], val_result_ttcc['pred_jet'] )
conf_jet_ttLF = confusion_matrix( val_result_ttLF['real_jet'], val_result_ttLF['pred_jet'] )

# sum of each rows of confusion matrix
sum_event = conf_event.sum(axis=1)[:, np.newaxis]
sum_jet = conf_jet.sum(axis=1)[:, np.newaxis]
sum_ttbb = conf_jet_ttbb.sum(axis=1)[:, np.newaxis]
sum_ttcc = conf_jet_ttcc.sum(axis=1)[:, np.newaxis]
sum_ttLF = conf_jet_ttLF.sum(axis=1)[:, np.newaxis]

# the number of validation events
n_event = conf_event.sum()
n_jet = conf_jet.sum()
n_ttbb = conf_jet_ttbb.sum()
n_ttcc = conf_jet_ttcc.sum()
n_ttLF = conf_jet_ttLF.sum()

# the number of correctly predicted events
n_corr_event = conf_event.trace()
n_corr_jet = conf_jet.trace()
n_corr_ttbb = conf_jet_ttbb.trace()
n_corr_ttcc = conf_jet_ttcc.trace()
n_corr_ttLF = conf_jet_ttLF.trace()

# calculate accuracy
acc_event = n_corr_event/n_event*100
acc_jet = n_corr_jet/n_jet*100
acc_ttbb = n_corr_ttbb/n_ttbb*100
acc_ttcc = n_corr_ttcc/n_ttcc*100
acc_ttLF = n_corr_ttLF/n_ttLF*100
print 'acc_event ',acc_event
print 'acc_jet ',acc_jet
print 'acc_ttbb ',acc_ttbb
print 'acc_ttcc ',acc_ttcc
print 'acc_ttLF ',acc_ttLF

print('Writing results')
with open(resultTxt, "a") as f_log:
    #f_log.write("\ntrainInput "+trainInput+'\n')
    f_log.write('\n'+ver+'\n')
    f_log.write('Nodes: '+str(nodes)+'\nEpochs '+str(epochs)+'\nDropout '+str(dropout)+'\n')
    #f_log.write('var: '+str(var_event+var_jet)+'\n')
    f_log.write('nvar: '+str(len(var_event+var_jet))+'\n')
    # accuracy
    f_log.write('event acc: '+str(n_corr_event)+'/'+str(n_event)+'='+str(acc_event)+'\n')
    f_log.write('jet acc: '+str(n_corr_jet)+'/'+str(n_jet)+'='+str(acc_jet)+'\n')
    f_log.write('ttbb acc: '+str(n_corr_ttbb)+'/'+str(n_ttbb)+'='+str(acc_ttbb)+'\n')
    f_log.write('ttcc acc: '+str(n_corr_ttcc)+'/'+str(n_ttcc)+'='+str(acc_ttcc)+'\n')
    f_log.write('ttLF acc: '+str(n_corr_ttLF)+'/'+str(n_ttLF)+'='+str(acc_ttLF)+'\n')
    # sample info
    f_log.write('training samples '+str(len(for_train))+'   validation samples '+str(len(for_valid))+'\n')
    f_log.write('ttbb jet perm list '+str(ttbb_jet_list)+' '+str(tot_ttbb)+'\n')
    f_log.write('ttcc jet perm list '+str(ttcc_jet_list)+' '+str(tot_ttcc)+'\n')
    f_log.write('ttLF jet perm list '+str(ttLF_jet_list)+' '+str(tot_ttLF)+'\n')
    #f_log.write('class_weight '+str( class_weight )+'\n')

print("Plotting scores")
# loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train','Test'],loc='upper right')
#plt.savefig(os.path.join(resultDir+'/'+ver,'Loss_N'+str(nodes)+'E'+str(epochs)+'D'+str(dropout)+'.pdf'))
plt.savefig(os.path.join(resultDir+'/'+ver,'Loss_'+ver+'.pdf'))
plt.show()
plt.gcf().clear()

# loss for event category
plt.plot(hist.history['event_prediction_loss'])
plt.plot(hist.history['val_event_prediction_loss'])
plt.title('Event Prediction Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train','Test'],loc='upper right')
#plt.savefig(os.path.join(resultDir+'/'+ver,'Loss_N'+str(nodes)+'E'+str(epochs)+'D'+str(dropout)+'.pdf'))
plt.savefig(os.path.join(resultDir+'/'+ver,'Loss_event_'+ver+'.pdf'))
plt.show()
plt.gcf().clear()

# loss for jet permutation
plt.plot(hist.history['jet_prediction_loss'])
plt.plot(hist.history['val_jet_prediction_loss'])
plt.title('Jet Prediction Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train','Test'],loc='upper right')
plt.savefig(os.path.join(resultDir+'/'+ver,'Loss_jet_'+ver+'.pdf'))
plt.gcf().clear()
#plt.savefig(os.path.join(resultDir+'/'+ver,'Loss_N'+str(nodes)+'E'+str(epochs)+'D'+str(dropout)+'.pdf'))
#real ttcc
#plt.rcParams['figure.figsize'] = [5, 5]
#cc00 = plt.hist( val_cc0['pbb'], density=True, linewidth=1.5, bins=20, histtype='step', color='royalblue' )
#cc01 = plt.hist( val_cc0['pcc'], density=True, linewidth=1.5, bins=20, histtype='step', color='indianred' )
#cc02 = plt.hist( val_cc0['plf'], density=True, linewidth=1.5, bins=20, histtype='step', color='mediumseagreen' )
#plt.xlabel('Prob.', fontsize=12)
#plt.ylabel('Normalized Entries', fontsize=12)
#plt.title('Real ttcc / Pred. ttbb', fontsize=17)
#leg = plt.legend( ['prob.ttbb', 'prob.ttcc', 'prob.ttLF'], loc='best', fontsize=14 )
#leg.get_frame().set_linewidth(0)
#plt.savefig(os.path.join(resultDir+'/'+ver,'ttcc_0.png'))
#plt.show()
#
#plt.rcParams['figure.figsize'] = [5, 5]
#cc10 = plt.hist( val_cc1['pbb'], density=True, linewidth=1.5, bins=20, histtype='step', color='royalblue' )
#cc11 = plt.hist( val_cc1['pcc'], density=True, linewidth=1.5, bins=20, histtype='step', color='indianred' )
#cc12 = plt.hist( val_cc1['plf'], density=True, linewidth=1.5, bins=20, histtype='step', color='mediumseagreen' )
#plt.xlabel('Prob.', fontsize=12)
#plt.ylabel('Normalized Entries', fontsize=12)
#plt.title('Real ttcc / Pred. ttbb', fontsize=17)
#leg = plt.legend( ['prob.ttbb', 'prob.ttcc', 'prob.ttLF'], loc='best', fontsize=14 )
#leg.get_frame().set_linewidth(0)
#plt.savefig(os.path.join(resultDir+'/'+ver,'ttcc_1.png'))
#plt.show()
#
#plt.rcParams['figure.figsize'] = [5, 5]
#cc20 = plt.hist( val_cc2['pbb'], density=True, linewidth=1.5, bins=20, histtype='step', color='royalblue' )
#cc21 = plt.hist( val_cc2['pcc'], density=True, linewidth=1.5, bins=20, histtype='step', color='indianred' )
#cc22 = plt.hist( val_cc2['plf'], density=True, linewidth=1.5, bins=20, histtype='step', color='mediumseagreen' )
#plt.xlabel('Prob.', fontsize=12)
#plt.ylabel('Normalized Entries', fontsize=12)
#plt.title('Real ttcc / Pred. ttLF', fontsize=17)
#leg = plt.legend( ['prob.ttbb', 'prob.ttcc', 'prob.ttLF'], loc='best', fontsize=14 )
#leg.get_frame().set_linewidth(0)
#plt.savefig(os.path.join(resultDir+'/'+ver,'ttcc_2.png'))
#plt.show()
#
#plt.savefig(os.path.join(resultDir+'/'+ver,'Loss_jet'+ver+'.pdf'))
#plt.show()
#plt.gcf().clear()

# Heatmap for event category
plt.rcParams['figure.figsize'] = [7.5, 6]
cfmt = lambda x,pos: '{:.0%}'.format(x)
heatmap = sns.heatmap(conf_event/sum_event, annot=True, cmap='YlGnBu', fmt='.1%', annot_kws={"size":12}, vmax=1, cbar_kws={'format': FuncFormatter(cfmt)} )
plt.title('Event Prediction Result', fontsize=15)
plt.xlabel('predicted event cat.', fontsize=12)
plt.ylabel('real event cat.', fontsize=12)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
#plt.savefig(os.path.join(resultDir+'/'+ver,'HM_N'+str(nodes)+'E'+str(epochs)+'D'+str(dropout)+'.pdf'))
#plt.savefig(os.path.join(resultDir+'/'+ver,'HM_'+plotName2+'_eff'+str(recoeff)+'.pdf'))
plt.savefig(os.path.join(resultDir+'/'+ver,'HM_event'+ver+'.pdf'))
plt.show()
plt.rcParams['figure.figsize'] = [12.5, 10]
plt.gcf().clear()

# Heatmap for jet permutation
plt.rcParams['figure.figsize'] = [12.5, 10]
cfmt = lambda x,pos: '{:.0%}'.format(x)
heatmap = sns.heatmap(conf_jet/sum_jet, annot=True, cmap='YlGnBu', fmt='.1%', annot_kws={"size":12}, vmax=1, cbar_kws={'format': FuncFormatter(cfmt)} )
plt.title('Jet Perm. Prediction Result', fontsize=15)
plt.xlabel('predicted jet perm.', fontsize=12)
plt.ylabel('real jet perm.', fontsize=12)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
#plt.savefig(os.path.join(resultDir+'/'+ver,'HM_N'+str(nodes)+'E'+str(epochs)+'D'+str(dropout)+'.pdf'))
#plt.savefig(os.path.join(resultDir+'/'+ver,'HM_'+plotName2+'_eff'+str(recoeff)+'.pdf'))
plt.savefig(os.path.join(resultDir+'/'+ver,'HM_jet_'+ver+'.pdf'))
plt.show()
plt.gcf().clear()

# Heatmap for jet permutation
plt.rcParams['figure.figsize'] = [12.5, 10]
cfmt = lambda x,pos: '{:.0%}'.format(x)
heatmap = sns.heatmap(conf_jet_ttbb/sum_ttbb, annot=True, cmap='YlGnBu', fmt='.1%', annot_kws={"size":12}, vmax=1, cbar_kws={'format': FuncFormatter(cfmt)} )
plt.title('ttbb Jet Perm. Predtiction Result', fontsize=15)
plt.xlabel('predicted jet perm.', fontsize=12)
plt.ylabel('real jet perm.', fontsize=12)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
plt.savefig(os.path.join(resultDir+'/'+ver,'HM_ttbb_'+ver+'.pdf'))
plt.show()
plt.gcf().clear()

plt.rcParams['figure.figsize'] = [12.5, 10]
cfmt = lambda x,pos: '{:.0%}'.format(x)
heatmap = sns.heatmap(conf_jet_ttcc/sum_ttcc, annot=True, cmap='YlGnBu', fmt='.1%', annot_kws={"size":12}, vmax=1, cbar_kws={'format': FuncFormatter(cfmt)} )
plt.title('ttcc Jet Perm. Prediction Result', fontsize=15)
plt.xlabel('predicted jet perm.', fontsize=12)
plt.ylabel('real jet perm.', fontsize=12)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
plt.savefig(os.path.join(resultDir+'/'+ver,'HM_ttcc_'+ver+'.pdf'))
plt.show()
plt.gcf().clear()

plt.rcParams['figure.figsize'] = [12.5, 10]
cfmt = lambda x,pos: '{:.0%}'.format(x)
heatmap = sns.heatmap(conf_jet_ttLF/sum_ttLF, annot=True, cmap='YlGnBu', fmt='.1%', annot_kws={"size":12}, vmax=1, cbar_kws={'format': FuncFormatter(cfmt)} )
plt.title('ttLF Jet Perm. Prediction Result', fontsize=15)
plt.xlabel('predicted jet perm.', fontsize=12)
plt.ylabel('real jet perm.', fontsize=12)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
plt.savefig(os.path.join(resultDir+'/'+ver,'HM_ttLF_'+ver+'.pdf'))
plt.show()
plt.rcParams['figure.figsize'] = [5, 5]
plt.gcf().clear()
plt.rcParams['figure.figsize'] = [5, 5]

