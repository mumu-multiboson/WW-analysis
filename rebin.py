import ROOT
from ROOT import *
import array
import os




in_path = "/home/elham/WW-analysis/reco_histograms/all/"
out_path = "/home/elham/WW-analysis/reco_histograms/rebinned/"

comd = "mkdir -p "+out_path
os.system(comd)


# variable of interest
var_name_nunuWW = "jj_M"

# define the binning
mjj_bins = array.array('d',[0,200,400,600,800,1000,1200,1500,2000,3000,4000,6000])

nunuWW_histos = [  "wzmunu_6tev.root",
            "zzmumu_6tev.root",
            "ggwpwm_6tev.root",
            "wpwmz_ztonunu_6tev.root",
            "mumu_nunuww_SM_6TeV.root",
            "mumu_nunuww_INT_T1_12_6TeV.root",
            "mumu_nunuww_QUAD_T1_12_6TeV.root"
]

print("====== Event Yileds ========")
for root_file in nunuWW_histos:
    inFile = ROOT.TFile(in_path+'/%s'%(root_file), 'r')
    h_tmp = inFile.Get(var_name_nunuWW)

    # print total number of events
    print("%s : "%(root_file), h_tmp.Integral())

    # rebin the histogram
    h_tmp_rebin = h_tmp.Rebin(len(mjj_bins)-1, 'h_tmp_rebin', mjj_bins)

    outFile = TFile(out_path+'/%s'%(root_file),'RECREATE')
    outFile.cd()
    h_tmp_rebin.Write(var_name_nunuWW)
    outFile.Close()
    del h_tmp

print("==================================")