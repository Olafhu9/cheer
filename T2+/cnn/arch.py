

#Some ideas for architectures. One option is having event variable and jet level variables.

#1x1 conv for processing jet variables

n_events = 1000
n_event_categories = 5 # ttbb, ttcc, etc.
n_perm = 6 # possible permutations. Would have to be the highest number if different event categories have different possible number of permutations.

event_level_input = np.random.rand(n_events,5) # for instance H_t, n_jets, n_btags, MET
jet_input = np.random.rand(n_events,10,7) # here we could put the jet matrix, for example up to 10 jets each jet has 7 input variables
lepton_input = np.random.rand(n_events,4,4) # isolated leptons, perhaps soft as well, for example up to 4 leptons each lepton has 4 input variables


Inputs = [Input(shape=(event_input.shape[1],)),Input(shape=(jet_input.shape[1],jet_input.shape[2])),Input(shape=(lepton_input.shape[1],lepton_input.shape[2]))]

event_info = BatchNormalization(momentum=momentum,name='globals_input_batchnorm') (Inputs[0])
jets    =     BatchNormalization(momentum=momentum,name='jet_input_batchnorm')     (Inputs[1])
leptons    =     BatchNormalization(momentum=momentum,name='leptons_input_batchnorm')     (Inputs[2])

event_info = BatchNormalization(name='globals_input_batchnorm') (Inputs[0])
jets    =     BatchNormalization(name='jet_input_batchnorm')     (Inputs[1])
leptons    =     BatchNormalization(name='leptons_input_batchnorm')     (Inputs[2])

#jets  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv0')(jets)
#jets  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv1')(jets)
jets  = Conv1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv0')(jets)
jets  = Conv1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='jets_conv1')(jets)
jets  = LSTM(50,go_backwards=True,implementation=2, name='jets_lstm')(jets)

leptons  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='leptons_conv0')(leptons)
leptons  = LSTM(25,go_backwards=True,implementation=2, name='leptons_lstm')(leptons)

x = Concatenate()( [event_info,jets,leptons ])
x = Dense(50, activation='relu',kernel_initializer='lecun_uniform')(x)
event_pred=Dense(n_event_categories, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
perm_pred=Dense(n_perm, activation='softmax',kernel_initializer='lecun_uniform',name='ID_perm')(x)

predictions = [event_pred,perm_pred] #= [1 0 0 0 0    0 0 0 1 0 0]
model = Model(inputs=Inputs, outputs=predictions)
