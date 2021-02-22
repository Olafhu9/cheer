#use : $python n2a.py /path/[ntupleName].root
import numpy as np
from numpy.lib.recfunctions import stack_arrays
from ROOT import *
from root_numpy import tree2array
import glob
import pandas as pd
import deepdish.io as io

import sys

arg = sys.argv[1] #arg = ntuple.root

input_ntuple = TFile.Open(arg)
input_ntuple_tree = input_ntuple.Get('dnn_input')
input_ntuple_array = tree2array(input_ntuple_tree)
input_ntuple_df = pd.DataFrame(input_ntuple_array)

#a = [ntuple].root (delete path)
#for a in arg.split('/'): continue
#a = a.replace("ana","") #a = [ntupleName].root
a = arg.replace(".root", ".h5") #a = [ntupleName].h5

#saved as [ntupleName].h5
io.save(a, input_ntuple_df)
