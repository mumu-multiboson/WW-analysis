from utils import Cut
truth_cuts = []
truth_cuts.append(Cut('n(leptons) == 0', lambda d: d['leptons'] == 0))
truth_cuts.append(Cut('n(hadronic W/Z) == 2', lambda d: len(d['hadronic_WZs']) == 2))
truth_cuts.append(Cut('M(nunu) > 200 GeV', lambda d: d['mass_nunu'] > 200))
truth_cuts.append(Cut('|cos(theta_{W/Z})| < 0.8', lambda d: all(abs(d[f'hadronic_WZs'][i].P4().CosTheta()) < 0.8 for i in (0,1))))
truth_cuts.append(Cut('pT_{leading W/Z} > 100 GeV', lambda d: all(d[f'hadronic_WZs'][i].PT > 100 for i in (0,1))))