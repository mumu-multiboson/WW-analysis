from utils import Cut
truth_cuts = []
truth_cuts.append(Cut('n(leptons) == 0', lambda d: d['leptons'] == 0))
truth_cuts.append(Cut('n(hadronic W/Z) == 2', lambda d: len(d['hadronic_WZs']) == 2))
truth_cuts.append(Cut('M(nunu) > 200 GeV', lambda d: d['mass_nunu'] > 200))