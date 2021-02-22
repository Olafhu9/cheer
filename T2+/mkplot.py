import numpy as np
import math
from ROOT import TFile, TTree
from ROOT import *
import ROOT
from array import array
import pandas as pd
import matplotlib.pyplot as plt

def find_bins2( xmin, xmax, binwidth ):
	xmin2 = -1 * (xmin * binwidth - binwidth)
	xmax2 = xmax - xmax % binwidth + binwidth
	n = int( (xmax2 - xmin2) / binwidth ) + 1
	bins = np.linspace( xmin2, xmax2, n )
	return bins

def find_bins( xmin, xmax, binwidth ):
	n = int( (xmax-xmin) / binwidth ) + 1 
	bins = np.linspace( xmin, xmax, n )
	# print 'n = ', n, ' bins = ', bins
	return bins

#input files
#input_ttbb = "/home/juhee5819/T2+/array/ttbb_2018_4f_pt20_v1.h5"
#input_ttcc = "/home/juhee5819/T2+/array/ttcc_2018_4f_pt20_v1.h5"
#input_ttLF = "/home/juhee5819/T2+/array/ttLF_2018_4f_pt20_v1.h5"

input_ttbb = "/home/juhee5819/T2+/array/ttbb_2018_4f_pt20_v1_2.h5"
input_ttcc = "/home/juhee5819/T2+/array/ttcc_2018_4f_pt20_v1_2.h5"
input_ttLF = "/home/juhee5819/T2+/array/ttLF_2018_4f_pt20_v1_2.h5"

ttbb = pd.read_hdf( input_ttbb )
ttcc = pd.read_hdf( input_ttcc )
ttLF = pd.read_hdf( input_ttLF )

ttbb = ttbb.drop( ttbb[ ttbb['jet_perm']==6 ].index )
ttcc = ttcc.drop( ttcc[ ttcc['jet_perm']==6 ].index )
ttLF = ttLF.drop( ttLF[ ttLF['jet_perm']==6 ].index )

# make histogram
def mkhist( var, title, xlabel, ylabel, xmin, xmax, binwidth ):
	bins = find_bins( xmin, xmax, binwidth )
	
	plt.rcParams['figure.figsize'] = (4, 4)
	hist_ttbb = plt.hist( ttbb[var], bins=bins, density=True, histtype='step', linewidth=1, color='royalblue' )
	hist_ttcc = plt.hist( ttcc[var], bins=bins, density=True, histtype='step', linewidth=1, color='indianred' )
	hist_ttLF = plt.hist( ttLF[var], bins=bins, density=True, histtype='step', linewidth=1, color='mediumseagreen' )

    # set y maximum
	maxi = max( max( hist_ttbb[0] ), max( hist_ttcc[0] ), max( hist_ttLF[0] ) ) * 1.2
	plt.ylim( 0, maxi )

	plt.title( title, fontsize=15 )
	plt.xlabel( xlabel, fontsize=10 )
	plt.ylabel( ylabel, fontsize=10 )

    # legend without line
	leg = plt.legend( ['ttbb', 'ttcc', 'ttLF'], loc='best', mode='expand', ncol=3,  fontsize=12 )
	leg.get_frame().set_linewidth(0)
	ax = plt.gca()
	ax.axes.yaxis.set_ticks([])
	plt.savefig(var+'.png', bbox_inches='tight')
	plt.show()
	#plt.savefig(var+'.png')
	#plt.show()

# global event variables
#mkhist( 'Ht', 'Ht', 'Ht (GeV)', 'Normarized Entries', 40, 500, 20 )
#mkhist( 'St', 'St', 'St (GeV)', 'Normarized Entries', 100, 800, 50 )
#mkhist( 'ngoodjets', 'Jet Multiplicity', 'Multiplicity', 'Normalized Entries', 4, 15, 1 )
#mkhist( 'nbjets_m', 'B-jet Multiplicity', 'Multiplicity', 'Normalized Entries', 0, 8, 1 )
#mkhist( 'ncjets_m', 'C-jet Multiplicity', 'Multiplicity', 'Normalized Entries', 0, 8, 1 )
#mkhist( 'lepton_pt', 'pT of Lepton', 'pT (GeV)', 'Normalized Entries', 0, 200, 20 )
#mkhist( 'lepton_eta', 'Eta of Lepton', 'Eta', 'Normalized Entries', 0, 2.5, 0.1 )
#mkhist( 'lepton_e', 'Energy of Lepton', 'Energy', 'Normalized Entries', 0, 200, 20 )
#mkhist( 'MET', 'MET', 'MET', 'Normalized Entries', 0, 200, 20 )
#mkhist( 'MET_phi', 'Phi of MET', 'Phi', 'Normalized Entries', 0, 3.5, 0.1 )
#mkhist( 'nulep_pt', 'pT of Lepton + MET', 'pT', 'Normalized Entries', 0, 300, 30 )
###
#mkhist( 'jet1_pt', 'pT of Jet1', 'pT (GeV)', 'Normalized Entries', 20, 140, 20 )
#mkhist( 'jet2_pt', 'pT of Jet2', 'pT (GeV)', 'Normalized Entries', 20, 140, 20 )
#mkhist( 'jet3_pt', 'pT of Jet3', 'pT (GeV)', 'Normalized Entries', 20, 140, 20 )
#mkhist( 'jet4_pt', 'pT of Jet4', 'pT (GeV)', 'Normalized Entries', 20, 140, 20 )
##
#mkhist( 'jet1_eta', 'Eta of Jet1', 'Eta', 'Normalized Entries', 0, 2.5, 0.1 )
#mkhist( 'jet2_eta', 'Eta of Jet2', 'Eta', 'Normalized Entries', 0, 2.5, 0.1 )
#mkhist( 'jet3_eta', 'Eta of Jet3', 'Eta', 'Normalized Entries', 0, 2.5, 0.1 )
#mkhist( 'jet4_eta', 'Eta of Jet4', 'Eta', 'Normalized Entries', 0, 2.5, 0.1 )
#
#mkhist( 'jet1_btag', 'B-discriminant of Jet1', 'B-score', 'Normalized Entries', 0, 1, 0.05 )
#mkhist( 'jet2_btag', 'B-discriminant of Jet2', 'B-score', 'Normalized Entries', 0, 1, 0.05 )
#mkhist( 'jet3_btag', 'B-discriminant of Jet3', 'B-score', 'Normalized Entries', 0, 1, 0.05 )
#mkhist( 'jet4_btag', 'B-discriminant of Jet4', 'B-score', 'Normalized Entries', 0, 1, 0.05 )
##
#mkhist( 'jet1_CvsB', 'CvsB of Jet1', 'CvsB', 'Normalized Entries', 0, 1, 0.05 )
#mkhist( 'jet2_CvsB', 'CvsB of Jet2', 'CvsB', 'Normalized Entries', 0, 1, 0.05 )
#mkhist( 'jet3_CvsB', 'CvsB of Jet3', 'CvsB', 'Normalized Entries', 0, 1, 0.05 )
#mkhist( 'jet4_CvsB', 'CvsB of Jet4', 'CvsB', 'Normalized Entries', 0, 1, 0.05 )
##
#mkhist( 'jet1_CvsL', 'CvsL of Jet1', 'CvsL', 'Normalized Entries', 0, 1, 0.05 )
#mkhist( 'jet2_CvsL', 'CvsL of Jet2', 'CvsL', 'Normalized Entries', 0, 1, 0.05 )
#mkhist( 'jet3_CvsL', 'CvsL of Jet3', 'CvsL', 'Normalized Entries', 0, 1, 0.05 )
#mkhist( 'jet4_CvsL', 'CvsL of Jet4', 'CvsL', 'Normalized Entries', 0, 1, 0.05 )
#
#
#mkhist( 'dRlep1', 'dR of Lepton & Jet1', 'dR', 'Normalized Entries', 0, 5, 0.1 )
#mkhist( 'dRlep2', 'dR of Lepton & Jet2', 'dR', 'Normalized Entries', 0, 5, 0.1 )
#mkhist( 'dRlep3', 'dR of Lepton & Jet3', 'dR', 'Normalized Entries', 0, 5, 0.1 )
#mkhist( 'dRlep4', 'dR of Lepton & Jet4', 'dR', 'Normalized Entries', 0, 5, 0.1 )
#
#mkhist( 'dRnulep1', 'dR of Neutrino+Lepton & Jet1', 'dR', 'Normalized Entries', 0, 5, 0.1 )
#mkhist( 'dRnulep2', 'dR of Neutrino+Lepton & Jet2', 'dR', 'Normalized Entries', 0, 5, 0.1 )
#mkhist( 'dRnulep3', 'dR of Neutrino+Lepton & Jet3', 'dR', 'Normalized Entries', 0, 5, 0.1 )
#mkhist( 'dRnulep4', 'dR of Neutrino+Lepton & Jet4', 'dR', 'Normalized Entries', 0, 5, 0.1 )
#
#mkhist( 'invmlep1', 'Invariantmass of Lepton & Jet1', 'Mass', 'Normalized Entries', 0, 300, 30 )
#mkhist( 'invmlep2', 'Invariantmass of Lepton & Jet2', 'Mass', 'Normalized Entries', 0, 300, 30 )
#mkhist( 'invmlep3', 'Invariantmass of Lepton & Jet3', 'Mass', 'Normalized Entries', 0, 300, 30 )
#mkhist( 'invmlep4', 'Invariantmass of Lepton & Jet4', 'Mass', 'Normalized Entries', 0, 300, 30 )
#
#mkhist( 'invmnu1', 'Invariantmass of Neutrino & Jet1', 'Mass', 'Normalized Entries', 0, 300, 30 )
mkhist( 'invmnu2', 'Invariantmass of Neutrino & Jet2', 'Mass', 'Normalized Entries', 0, 300, 30 )
#mkhist( 'invmnu3', 'Invariantmass of Neutrino & Jet3', 'Mass', 'Normalized Entries', 0, 300, 30 )
#mkhist( 'invmnu4', 'Invariantmass of Neutrino & Jet4', 'Mass', 'Normalized Entries', 0, 300, 30 )
#
#mkhist( 'dPhi12', 'dPhi of Jet1 & Jet2', 'dPhi', 'Normalized Entries', 0, 3.5, 0.1 ) 
mkhist( 'dPhi13', 'dPhi of Jet1 & Jet3', 'dPhi', 'Normalized Entries', 0, 3.5, 0.1 ) 
#mkhist( 'dPhi14', 'dPhi of Jet1 & Jet4', 'dPhi', 'Normalized Entries', 0, 3.5, 0.1 ) 
#mkhist( 'dPhi23', 'dPhi of Jet2 & Jet3', 'dPhi', 'Normalized Entries', 0, 3.5, 0.1 ) 
mkhist( 'dPhi24', 'dPhi of Jet2 & Jet4', 'dPhi', 'Normalized Entries', 0, 3.5, 0.1 ) 
#mkhist( 'dPhi34', 'dPhi of Jet3 & Jet4', 'dPhi', 'Normalized Entries', 0, 3.5, 0.1 ) 
##
#mkhist( 'invm12', 'Invariantmass of Jet1 & Jet2', 'Mass', 'Normalized Entries', 0, 200, 20 ) 
#mkhist( 'invm13', 'Invariantmass of Jet1 & Jet3', 'Mass', 'Normalized Entries', 0, 200, 20 ) 
#mkhist( 'invm14', 'Invariantmass of Jet1 & Jet4', 'Mass', 'Normalized Entries', 0, 200, 20 ) 
#mkhist( 'invm23', 'Invariantmass of Jet2 & Jet3', 'Mass', 'Normalized Entries', 0, 200, 20 ) 
#mkhist( 'invm24', 'Invariantmass of Jet2 & Jet4', 'Mass', 'Normalized Entries', 0, 200, 20 ) 
#mkhist( 'invm34', 'Invariantmass of Jet3 & Jet4', 'Mass', 'Normalized Entries', 0, 200, 20 ) 
##
#mkhist( 'dR12', 'dR of Jet1 & Jet2', 'dR', 'Normalized Entries', 0, 5, 0.1 ) 
#mkhist( 'dR13', 'dR of Jet1 & Jet3', 'dR', 'Normalized Entries', 0, 5, 0.1 ) 
#mkhist( 'dR14', 'dR of Jet1 & Jet4', 'dR', 'Normalized Entries', 0, 5, 0.1 ) 
#mkhist( 'dR23', 'dR of Jet2 & Jet3', 'dR', 'Normalized Entries', 0, 5, 0.1 ) 
#mkhist( 'dR24', 'dR of Jet2 & Jet4', 'dR', 'Normalized Entries', 0, 5, 0.1 ) 
#mkhist( 'dR34', 'dR of Jet3 & Jet4', 'dR', 'Normalized Entries', 0, 5, 0.1 ) 
##
