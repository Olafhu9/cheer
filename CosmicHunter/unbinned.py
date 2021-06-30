import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ROOT import *
from root_numpy import array2tree

# read csv file
df = pd.read_csv('/home/juhee5819/cheer/CosmicHunter/data/altitude/826m_0506/CSMHUNT_12300_2021-5-6_3-58-22.csv', sep='\s*[;]\s*', skipinitialspace=True)
drop_idx = df.loc[ df['coinc']!='ABC'].index
df = df.drop(drop_idx)
counts = df['COINC']

counts_arr = np.array( counts , dtype=[('counts', np.float32)])
tree = array2tree(counts_arr)

count = RooRealVar("counts", "Detected counts per 600 sec", 590, 750 )
count.setBins(16)
#count = RooRealVar("counts", "counts", counts.min(), counts.max() )
mean = RooRealVar("mean", "mean", counts.mean())
sig = RooRealVar("sigma", "sigma", counts.std(), 5, 50)

name = 'data'
data = RooDataSet( name, name, RooArgSet(count), RooFit.Import(tree) )

gauss = RooGaussian("Gaussian", "Gaussian", count, mean, sig)
gauss.fitTo(data)
xframe = count.frame(RooFit.Title("Muon Counts in Bukhansan"))
data.plotOn(xframe)
gauss.plotOn(xframe)
data.statOn(xframe, RooFit.Layout(0.62, 0.88, 0.89))
st1 = xframe.getAttText()
st1.SetBorderSize(0)
st1.SetTextSize(0.032)
gauss.paramOn(xframe, RooFit.Layout(0.62, 0.88, 0.71))
st2 = xframe.getAttText()
st2.SetBorderSize(0)
st2.SetTextSize(0.032)

#xframe.SetStats()
c = TCanvas('c', 'c', 3)
c.SetLeftMargin(0.15)
#c.SetTopMargin(0.15)

xframe.GetXaxis().SetTitleSize(0.04)
xframe.GetYaxis().SetTitleSize(0.04)
xframe.Draw()

#gauss.fitTo(data)
c.Print(Form("unbinned_bukhan.pdf"))
