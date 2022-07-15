from utils import Cut

def get_cuts(max_leptons: str = '0', min_jets: str = '2', min_M_miss: str = '200', max_cos_theta: str = '0.8', pt1_min: str = '100.', pt2_min: str = '100.', mass_min: str = '40.'):
    max_leptons = int(max_leptons)
    min_jets = int(min_jets)
    min_M_miss = float(min_M_miss)
    max_cos_theta = float(max_cos_theta)
    pt1_min = float(pt1_min)
    pt2_min = float(pt2_min)
    mass_min = float(mass_min)
    reco_cuts = []
    reco_cuts.append(Cut(f'n(leptons) <= {max_leptons}', lambda d: d['n_leptons'] <= max_leptons))
    reco_cuts.append(Cut(f'n(jets) >= {min_jets}', lambda d: d['n_jets'] >= min_jets))
    reco_cuts.append(Cut(f'M_miss > {min_M_miss} GeV', lambda d: d['missing_mass'] > min_M_miss))
    reco_cuts.append(Cut(f'|cos(theta_j)| < {max_cos_theta}', lambda d: all(abs(d[f'jet_{i}'].P4().CosTheta()) < max_cos_theta for i in (1,2))))
    reco_cuts.append(Cut(f'M_{{jets}} > {mass_min} GeV', lambda d: all(d[f'jet_{i}'].Mass > mass_min for i in (1,2))))
    reco_cuts.append(Cut(f'pT_{{leading jet}} > {pt1_min} GeV', lambda d: d['jet_1'].PT > pt1_min))
    reco_cuts.append(Cut(f'pT_{{2nd leading jet}} > {pt2_min} GeV', lambda d: d['jet_2'].PT > pt2_min))
    return reco_cuts
