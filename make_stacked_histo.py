from ROOT import *
from sys import argv
import os

path = '/raid01/users/kawale/aQGC/WW-analysis/condor/stackedHistograms/allCuts/pTcut_symmetric/pT100/30tev'

inputs=argv[1:]


f0=TFile(inputs[0])
keys=f0.GetListOfKeys()

colors = [kBlue, kRed, kGreen, kYellow, kViolet, kBlack, kCyan]
legend = ['zzmumu', 'wpwmz_ztonunu', 'SM', 'wzmunu', 'ggwpwm', 'INT', 'QUAD']
for k in keys:
    c = TCanvas()
    hs = THStack("hs", "; M_jj(GeV) ; Events")
    hs.SetMinimum(0.01)
    signal = THStack("signal", "; M_jj(GeV) ; Events")
    signal.SetMinimum(0.01)
    l = TLegend(0.65,0.7,0.88,0.83)
    #l = TLegend()
    l.SetBorderSize(1)
    l.SetTextFont(62)
    l.SetTextSize(0.)
    m=0
    for f, color in zip(inputs, colors):
        i=inputs.index(f)
        f=TFile(f)
        SetOwnership(f,False)
        o=f.Get(k.GetName())
        if type(o)!=type(TH1F()): continue
        o.SetStats(0) #to delete the statistics box for a histogram
        o.SetLineColor(color)
        #o.GetXaxis().SetTitle("M_{jj} [GeV]")
        #o.GetYaxis().SetTitle("Events")
        if i<5:
            o.SetFillColor(color)
            #o.SetLineColor(color)
            #l.AddEntry(o,legend[i])
            hs.Add(o)
        else:
            signal.Add(o)
        l.AddEntry(o,legend[i])

    signal.Draw('HIST')
    hs.Draw('HIST SAME')
    #hs.GetXaxis().SetTitle("Dijet mass")
    #hs.GetYaxis().SetTitle("Events")
    l.Draw()
    c.Modified()
    c.SetLogy()
    #outputfilename = os.path.join(path,o.GetName()+'.png')
    outputfilename = os.path.join(path,'Mjj_scaled.png')
    c.SaveAs(outputfilename)