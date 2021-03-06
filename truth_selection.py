from utils import Cut

def get_cuts(max_leptons: str = '0', min_jets: str = '2', min_M_miss: str = '200', max_cos_theta: str = '0.8', pt_min: str = '100.'):
    max_leptons = int(max_leptons)
    min_jets = int(min_jets)
    min_M_miss = float(min_M_miss)
    max_cos_theta = float(max_cos_theta)
    pt_min = float(pt_min)

    truth_cuts = []
    truth_cuts.append(Cut(f'n(leptons) <= {max_leptons}', lambda d: d['leptons'] <= max_leptons))
    truth_cuts.append(Cut(f'n(hadronic W/Z) >= {min_jets}', lambda d: len(d['hadronic_WZs']) == min_jets))
    truth_cuts.append(Cut(f'M(nunu) > {min_M_miss} GeV', lambda d: d['mass_nunu'] > min_M_miss))
    truth_cuts.append(Cut(f'|cos(theta_{{W/Z}})| < {max_cos_theta}', lambda d: all(abs(d[f'hadronic_WZs'][i].P4().CosTheta()) < max_cos_theta for i in (0,1))))
    truth_cuts.append(Cut(f'pT_{{leading jet}} > {pt_min} GeV', lambda d: all(d[f'hadronic_WZs'][i].PT > pt_min for i in (0,1))))
    return truth_cuts