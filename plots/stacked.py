from typing import List
from ROOT import *
from pathlib import Path
import yaml
# gROOT.SetStyle('ATLAS')
gStyle.SetOptStat(1111)

backgrounds = {
    'ggwpwm': {
        'path': '/afs/cern.ch/user/a/aschuy/work/private/VBS_WGamma/muon_collider/WW-analysis/heavy_higgs/backgrounds/all/ggwpwm_6tev.root',
        'color': kRed,
        },
    'wpwmz_ztonunu': {
        'path': '/afs/cern.ch/user/a/aschuy/work/private/VBS_WGamma/muon_collider/WW-analysis/heavy_higgs/backgrounds/all/wpwmz_ztonunu_6tev.root',
        'color': kYellow,
    },
    'wzmunu': {
        'path': '/afs/cern.ch/user/a/aschuy/work/private/VBS_WGamma/muon_collider/WW-analysis/heavy_higgs/backgrounds/all/wzmunu_6tev.root',
        'color': kCyan,
    },
    'zzmumu': {
        'path': '/afs/cern.ch/user/a/aschuy/work/private/VBS_WGamma/muon_collider/WW-analysis/heavy_higgs/backgrounds/all/zzmumu_6tev.root',
        'color': kSpring,
    },
}

signals = {
    'heavy_higgs_500': {
        'path': '/afs/cern.ch/user/a/aschuy/work/private/VBS_WGamma/muon_collider/WW-analysis/heavy_higgs/all/heavy_higgs_500GeV_6TeV.root',
        'color': kBlack,
    },
    'heavy_higgs_1000': {
        'path': '/afs/cern.ch/user/a/aschuy/work/private/VBS_WGamma/muon_collider/WW-analysis/heavy_higgs/all/heavy_higgs_1000GeV_6TeV.root',
        'color': kBlack,
    },
    'heavy_higgs_2000': {
        'path': '/afs/cern.ch/user/a/aschuy/work/private/VBS_WGamma/muon_collider/WW-analysis/heavy_higgs/all/heavy_higgs_2000GeV_6TeV.root',
        'color': kBlack,
    },
    'heavy_higgs_3000': {
        'path': '/afs/cern.ch/user/a/aschuy/work/private/VBS_WGamma/muon_collider/WW-analysis/heavy_higgs/all/heavy_higgs_3000GeV_6TeV.root',
        'color': kBlack,
    },
}
c1 = TCanvas("c1", "", 800, 600)
c1.SetFillStyle(1001)
c1.SetFillColor(kWhite)
hs = THStack("hs", "")

# for k in backgrounds:
#     path = backgrounds[k]['path']
#     color = backgrounds[k]['color']
#     f = TFile(path)
#     h = f.Get('jj_M')
#     h.SetFillColor(color)
#     hs.Add(h)

f1 = TFile('/afs/cern.ch/user/a/aschuy/work/private/VBS_WGamma/muon_collider/WW-analysis/heavy_higgs/backgrounds/all/ggwpwm_6tev.root')
h1 = f1.Get('jj_M')
h1.SetFillColor(kRed)
hs.Add(h1)
f2 = TFile('/afs/cern.ch/user/a/aschuy/work/private/VBS_WGamma/muon_collider/WW-analysis/heavy_higgs/backgrounds/all/wzmunu_6tev.root')
h2 = f2.Get('jj_M')
h2.SetFillColor(kBlue)
hs.Add(h2)
# h1 = TH1F("h1", "test hstack", 10, -4, 4)
# h1.FillRandom("gaus", 20000)
# h1.SetFillColor(kRed)
# hs.Add(h1)
# h2 = TH1F("h2", "test hstack", 10, -4, 4)
# h2.FillRandom("gaus", 15000)
# h2.SetFillColor(kBlue)
# hs.Add(h2)
hs.Draw()
c1.Update()
c1.SaveAs('hstack.png')