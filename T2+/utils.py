import sys, os
import math
from tqdm import tqdm

def getVarlist():
    column = ['pt1', 'eta1', 'phi1', 'e1', 'pt2', 'eta2', 'phi2', 'e2',
        'bbdR','bbdEta','bbdPhi','bbPt','bbEta',
        'bbMass',
        'nubbdR','nubbdEta','nubbdPhi','nubbPt','nubbMass','nubbHt',
        'lbbdR','lbbdEta','lbbdPhi','lbbPt','lbbEta','lbbMass',
        'nub1dR','nub1dEta','nub1dPhi','nub1Mass',
        'nub2dR','nub2dEta','nub2dPhi','nub2Mass','nub2Ht',
        'lb1dR','lb1dEta','lb1dPhi','lb1Pt','lb1Eta','lb1Mass','lb1Ht',
        'lb2dR','lb2dEta','lb2dPhi','lb2Pt','lb2Eta','lb2Mass','lb2Ht'
#        'csv1','csv2',
#        'Wjb1dR','Wjb1dEta','Wjb1dPhi','Wjb1Pt','Wjb1Mass','Wjb1Ht',
#        'Wjb2dR','Wjb2dEta','Wjb2dPhi','Wjb2Pt','Wjb2Mass','Wjb2Ht'
    ]
    return column
#list backup
#    column = ['bbdR','bbdEta','bbdPhi','bbPt','bbEta','bbPhi','bbMass','bbHt','bbMt',
#        'nubbdR','nubbdEta','nubbdPhi','nubbPt','nubbEta','nubbPhi','nubbMass','nubbHt','nubbMt',
#        'lbbdR','lbbdEta','lbbdPhi','lbbPt','lbbEta','lbbPhi','lbbMass','lbbHt','lbbMt',
#        'nub1dR','nub1dEta','nub1dPhi','nub1Pt','nub1Eta','nub1Phi','nub1Mass','nub1Ht','nub1Mt',
#        'nub2dR','nub2dEta','nub2dPhi','nub2Pt','nub2Eta','nub2Phi','nub2Mass','nub2Ht','nub2Mt',
#        'lb1dR','lb1dEta','lb1dPhi','lb1Pt','lb1Eta','lb1Phi','lb1Mass','lb1Ht','lb1Mt',
#        'lb2dR','lb2dEta','lb2dPhi','lb2Pt','lb2Eta','lb2Phi','lb2Mass','lb2Ht','lb2Mt',
##        'csv1','csv2',
#        'Wjb1dR','Wjb1dEta','Wjb1dPhi','Wjb1Pt','Wjb1Eta','Wjb1Phi','Wjb1Mass','Wjb1Ht','Wjb1Mt',
#        'Wjb2dR','Wjb2dEta','Wjb2dPhi','Wjb2Pt','Wjb2Eta','Wjb2Phi','Wjb2Mass','Wjb2Ht','Wjb2Mt',
#        'pt1','pt2','eta1','eta2','phi1','phi2','e1','e2',
#    ]



def getHistXlabel():
    dictionary = {
        'bbdR':'#DeltaR_{b#bar{b}}', 'bbdEta':'#Delta#eta_{b#bar{b}}','bbdPhi':'#Delta#phi_{b#bar{b}}','bbPt':'pT_{b#bar{b}}', 'bbEta':'#eta_{b#bar{b}}', 'bbPhi':'#phi_{b#bar{b}}', 'bbMass':'m_{b#bar{b}}','bbHt':'HT_{b#bar{b}}','bbMt':'mT_{b#bar{b}}',
        'nubbdR':'#DeltaR_{b#bar{b}#nu}', 'nubbdEta':'#Delta#eta_{b#bar{b}#nu}','nubbdPhi':'#Delta#phi_{b#bar{b}#nu}','nubbPt':'pT_{b#bar{b}#nu}', 'nubbEta':'#eta_{b#bar{b}#nu}', 'nubbPhi':'#phi_{b#bar{b}#nu}', 'nubbMass':'m_{b#bar{b}#nu}','nubbHt':'HT_{b#bar{b}#nu}','nubbMt':'mT_{b#bar{b}#nu}',
        'lbbdR':'#DeltaR_{b#bar{b}l}', 'lbbdEta':'#Delta#eta_{b#bar{b}l}','lbbdPhi':'#Delta#phi_{b#bar{b}l}','lbbPt':'pT_{b#bar{b}l}', 'lbbEta':'#eta_{b#bar{b}l}', 'lbbPhi':'#phi_{b#bar{b}l}', 'lbbMass':'m_{b#bar{b}l}','lbbHt':'HT_{b#bar{b}l}','lbbMt':'mT_{b#bar{b}l}',
        'nub1dR':'#DeltaR_{b1#nu}', 'nub1dEta':'#Delta#eta_{b1#nu}','nub1dPhi':'#Delta#phi_{b1#nu}','nub1Pt':'pT_{b1#nu}', 'nub1Eta':'#eta_{b1#nu}', 'nub1Phi':'#phi_{b1#nu}', 'nub1Mass':'m_{b1#nu}','nub1Ht':'HT_{b1#nu}','nub1Mt':'mT_{b1#nu}',
        'nub2dR':'#DeltaR_{b2#nu}', 'nub2dEta':'#Delta#eta_{b2#nu}','nub2dPhi':'#Delta#phi_{b2#nu}','nub2Pt':'pT_{b2#nu}', 'nub2Eta':'#eta_{b2#nu}', 'nub2Phi':'#phi_{b2#nu}', 'nub2Mass':'m_{b2#nu}','nub2Ht':'HT_{b2#nu}','nub2Mt':'mT_{b2#nu}',
        'lb1dR':'#DeltaR_{b1l}', 'lb1dEta':'#Delta#eta_{b1l}','lb1dPhi':'#Delta#phi_{b1l}','lb1Pt':'pT_{b1l}', 'lb1Eta':'#eta_{b1l}', 'lb1Phi':'#phi_{b1l}', 'lb1Mass':'m_{b1l}','lb1Ht':'HT_{b1l}','lb1Mt':'mT_{b1l}',
        'lb2dR':'#DeltaR_{b2l}', 'lb2dEta':'#Delta#eta_{b2l}','lb2dPhi':'#Delta#phi_{b2l}','lb2Pt':'pT_{b2l}', 'lb2Eta':'#eta_{b2l}', 'lb2Phi':'#phi_{b2l}', 'lb2Mass':'m_{b2l}','lb2Ht':'HT_{b2l}','lb2Mt':'mT_{b2l}',
#        'csv1':'csv_{b1}', 'csv2':'csv_{b2}',
        'Wjb1dR':'#DeltaR_{b1W_{j}}', 'Wjb1dEta':'#Delta#eta_{b1W_{j}}','Wjb1dPhi':'#Delta#phi_{b1W_{j}}','Wjb1Pt':'pT_{b1W_{j}}', 'Wjb1Eta':'#eta_{b1W_{j}}', 'Wjb1Phi':'#phi_{b1W_{j}}', 'Wjb1Mass':'m_{b1W_{j}}','Wjb1Ht':'HT_{b1W_{j}}','Wjb1Mt':'mT_{b1W_{j}}',
        'Wjb2dR':'#DeltaR_{b2W_{j}}', 'Wjb2dEta':'#Delta#eta_{b2W_{j}}','Wjb2dPhi':'#Delta#phi_{b2W_{j}}','Wjb2Pt':'pT_{b2W_{j}}', 'Wjb2Eta':'#eta_{b2W_{j}}', 'Wjb2Phi':'#phi_{b2W_{j}}', 'Wjb2Mass':'m_{b2W_{j}}','Wjb2Ht':'HT_{b2W_{j}}','Wjb2Mt':'mT_{b2W_{j}}',
        'pt1':'pT_{b1}','pt2':'pT_{b2}', 'eta1':'#eta_{b1}', 'eta2':'#eta_{b2}','phi1':'#phi_{b1}', 'phi2':'#phi_{b2}', 'e1':'E_{b1}', 'e2':'E_{b2}'
    }

    #list backup
#    dictionary = {
#        'bbdR':'#DeltaR_{b#bar{b}}', 'bbdEta':'#Delta#eta_{b#bar{b}}','bbdPhi':'#Delta#phi_{b#bar{b}}','bbPt':'pT_{b#bar{b}}', 'bbEta':'#eta_{b#bar{b}}', 'bbPhi':'#phi_{b#bar{b}}', 'bbMass':'m_{b#bar{b}}','bbHt':'HT_{b#bar{b}}','bbMt':'mT_{b#bar{b}}',
#        'nubbdR':'#DeltaR_{b#bar{b}#nu}', 'nubbdEta':'#Delta#eta_{b#bar{b}#nu}','nubbdPhi':'#Delta#phi_{b#bar{b}#nu}','nubbPt':'pT_{b#bar{b}#nu}', 'nubbEta':'#eta_{b#bar{b}#nu}', 'nubbPhi':'#phi_{b#bar{b}#nu}', 'nubbMass':'m_{b#bar{b}#nu}','nubbHt':'HT_{b#bar{b}#nu}','nubbMt':'mT_{b#bar{b}#nu}',
#        'lbbdR':'#DeltaR_{b#bar{b}l}', 'lbbdEta':'#Delta#eta_{b#bar{b}l}','lbbdPhi':'#Delta#phi_{b#bar{b}l}','lbbPt':'pT_{b#bar{b}l}', 'lbbEta':'#eta_{b#bar{b}l}', 'lbbPhi':'#phi_{b#bar{b}l}', 'lbbMass':'m_{b#bar{b}l}','lbbHt':'HT_{b#bar{b}l}','lbbMt':'mT_{b#bar{b}l}',
#        'nub1dR':'#DeltaR_{b1#nu}', 'nub1dEta':'#Delta#eta_{b1#nu}','nub1dPhi':'#Delta#phi_{b1#nu}','nub1Pt':'pT_{b1#nu}', 'nub1Eta':'#eta_{b1#nu}', 'nub1Phi':'#phi_{b1#nu}', 'nub1Mass':'m_{b1#nu}','nub1Ht':'HT_{b1#nu}','nub1Mt':'mT_{b1#nu}',
#        'nub2dR':'#DeltaR_{b2#nu}', 'nub2dEta':'#Delta#eta_{b2#nu}','nub2dPhi':'#Delta#phi_{b2#nu}','nub2Pt':'pT_{b2#nu}', 'nub2Eta':'#eta_{b2#nu}', 'nub2Phi':'#phi_{b2#nu}', 'nub2Mass':'m_{b2#nu}','nub2Ht':'HT_{b2#nu}','nub2Mt':'mT_{b2#nu}',
#        'lb1dR':'#DeltaR_{b1l}', 'lb1dEta':'#Delta#eta_{b1l}','lb1dPhi':'#Delta#phi_{b1l}','lb1Pt':'pT_{b1l}', 'lb1Eta':'#eta_{b1l}', 'lb1Phi':'#phi_{b1l}', 'lb1Mass':'m_{b1l}','lb1Ht':'HT_{b1l}','lb1Mt':'mT_{b1l}',
#        'lb2dR':'#DeltaR_{b2l}', 'lb2dEta':'#Delta#eta_{b2l}','lb2dPhi':'#Delta#phi_{b2l}','lb2Pt':'pT_{b2l}', 'lb2Eta':'#eta_{b2l}', 'lb2Phi':'#phi_{b2l}', 'lb2Mass':'m_{b2l}','lb2Ht':'HT_{b2l}','lb2Mt':'mT_{b2l}',
##        'csv1':'csv_{b1}', 'csv2':'csv_{b2}',
#        'Wjb1dR':'#DeltaR_{b1W_{j}}', 'Wjb1dEta':'#Delta#eta_{b1W_{j}}','Wjb1dPhi':'#Delta#phi_{b1W_{j}}','Wjb1Pt':'pT_{b1W_{j}}', 'Wjb1Eta':'#eta_{b1W_{j}}', 'Wjb1Phi':'#phi_{b1W_{j}}', 'Wjb1Mass':'m_{b1W_{j}}','Wjb1Ht':'HT_{b1W_{j}}','Wjb1Mt':'mT_{b1W_{j}}',
#        'Wjb2dR':'#DeltaR_{b2W_{j}}', 'Wjb2dEta':'#Delta#eta_{b2W_{j}}','Wjb2dPhi':'#Delta#phi_{b2W_{j}}','Wjb2Pt':'pT_{b2W_{j}}', 'Wjb2Eta':'#eta_{b2W_{j}}', 'Wjb2Phi':'#phi_{b2W_{j}}', 'Wjb2Mass':'m_{b2W_{j}}','Wjb2Ht':'HT_{b2W_{j}}','Wjb2Mt':'mT_{b2W_{j}}',
#        'pt1':'pT_{b1}','pt2':'pT_{b2}', 'eta1':'#eta_{b1}', 'eta2':'#eta_{b2}','phi1':'#phi_{b1}', 'phi2':'#phi_{b2}', 'e1':'E_{b1}', 'e2':'E_{b2}'
#    }
    return dictionary

def getHistRange(var):

    histRange = []

    if 'dR' in var:
        histRange = [20, 0, 4]
    elif 'dEta' in var:
        histRange = [20, 0, 2.5]
    elif 'dPhi' in var:
        histRange = [20, 0, math.pi]
    elif 'Pt' in var:
        histRange = [20, 0, 400]
    elif ('Eta' in var) and (not 'dEta' in var):
        histRange = [20, -2.5, 2.5]
    elif ('Phi' in var) and (not 'dPhi' in var):
        histRange = [20, -math.pi, math.pi]
    elif ('mT' in var) or ('M' in var):
        histRange = [20, 0, 400]
    elif ('E' in var) and (not 'Et' in var) or ('e' in var):
        histRange = [20, 0, 400]
    elif 'csv' in var:
        histRange = [20, 0, 1]
    else:
        histRange = [20, 0, 10]
        print("Variable does not exist")

    return histRange

def printProgress(iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    nEvent = str(iteration) + '/' + str(total)
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100*(iteration/float(total)))
    filledLength = int(round(barLength * iteration/float(total)))
    bar = '#'*filledLength + '-'*(barLength-filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s %s' % (prefix, bar, percent, '%', suffix, nEvent)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

