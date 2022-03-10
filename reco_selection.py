from utils import Cut
reco_cuts = []
reco_cuts.append(Cut('n(leptons) == 0', lambda d: d['n_leptons'] == 0))
reco_cuts.append(Cut('n(jets) == 2', lambda d: d['n_jets'] == 2))
reco_cuts.append(Cut('M_miss > 200 GeV', lambda d: d['missing_mass'] > 200))
reco_cuts.append(Cut('|cos(theta_j)| < 0.8', lambda d: all(abs(d[f'jet_{i}'].P4().CosTheta()) < 0.8 for i in (1,2))))
reco_cuts.append(Cut('pT_{leading jet} > 100 GeV', lambda d: d['jet_2'].PT > 100))