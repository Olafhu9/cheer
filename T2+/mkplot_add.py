import numpy as np
import math
from ROOT import TFile, TTree
from ROOT import *
import ROOT
from array import array
import pandas as pd
import matplotlib.pyplot as plt

#input files
input_ttcc = "/home/juhee5819/T2+/array/ttcc_addjets.h5"
input_ttLF = "/home/juhee5819/T2+/array/ttLF_addjets.h5"

data_ttcc = pd.read_hdf( input_ttcc )
data_ttLF = pd.read_hdf( input_ttLF )

x1 = data_ttcc['addbjet1_eta']
y1 = data_ttcc['addbjet1_pt']
x2 = data_ttcc['addbjet2_eta']
y2 = data_ttcc['addbjet2_pt']

plt.rcParams['figure.figsize'] = (4, 4)
plt.scatter( x=x1, y=y1, s=15 )
plt.xlabel('Eta', fontsize=12 )
plt.ylabel('Pt', fontsize=12 )
plt.title('Additional b-jet 1', fontsize=15 )
plt.xticks( np.arange(-5, 5, 2.5), ('', '-2.5', '', '2.5', '') )
plt.yticks( np.arange(0, 400, 100), ('', '100', '200', '300', '400') )
plt.axvline( x=-2.5, color='r' )
plt.axvline( x=2.5, color='r' )
plt.axhline( y=20, color='r' )
plt.savefig('ttcc_addbjet1.png', bbox_inches='tight')
plt.gcf().clear()

plt.scatter( x=x2, y=y2, s=12 )
plt.xlabel('Eta', fontsize=12 )
plt.ylabel('Pt', fontsize=12 )
plt.title('Additional b-jet 2', fontsize=15 )
plt.xticks( np.arange(-5, 5, 2.5), ('', '-2.5', '', '2.5', '') )
plt.yticks( np.arange(0, 100, 50), ('', '50','') )
plt.axvline( x=-2.5, color='r' )
plt.axvline( x=2.5, color='r' )
plt.axhline( y=20, color='r' )
plt.savefig('ttcc_addbjet2.png', bbox_inches='tight')

#bjet1_pt = data_ttLF['addbjet1_pt'] > 20
#bjet1_eta1  = data_ttLF['addbjet1_eta'] < 2.5
#bjet1_eta2 = data_ttLF['addbjet1_eta'] > -2.5
#
#bjet2_pt = data_ttLF['addbjet2_pt'] > 20
#bjet2_eta1  = data_ttLF['addbjet2_eta'] < 2.5
#bjet2_eta2 = data_ttLF['addbjet2_eta'] > -2.5
#
#cjet1_pt = data_ttLF['addcjet1_pt'] > 20
#cjet1_eta1  = data_ttLF['addcjet1_eta'] < 2.5
#cjet1_eta2 = data_ttLF['addcjet1_eta'] > -2.5
#
#cjet2_pt = data_ttLF['addcjet2_pt'] > 20
#cjet2_eta1  = data_ttLF['addcjet2_eta'] < 2.5
#cjet2_eta2 = data_ttLF['addcjet2_eta'] > -2.5
#
#data_ttLF[ bjet1_pt & bjet1_eta1 & bjet1_eta2 ]
#data_ttLF[ bjet2_pt & bjet2_eta1 & bjet2_eta2 ]
#data_ttLF[ cjet1_pt & cjet1_eta1 & cjet1_eta2 ]
#data_ttLF[ cjet2_pt & cjet2_eta1 & cjet2_eta2 ]
#
#data_ttLF[ bjet1_pt & bjet1_eta1 & bjet1_eta2 & bjet2_pt & bjet2_eta1 & bjet2_eta2 ]
#data_ttLF[ cjet1_pt & cjet1_eta1 & cjet1_eta2 & cjet2_pt & cjet2_eta1 & cjet2_eta2 ]



