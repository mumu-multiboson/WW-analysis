from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import yaml
from ROOT import *

gSystem.Load("libDelphes")
gStyle.SetOptStat(0)

try:
    from rich.progress import track as progress
    has_rich=True
except ImportError:
    print('To include progress bar, run `pip install rich`')
    has_rich=False


def deltaR(p1, p2):
    dR = p1.P4().DeltaR(p2.P4())
    return dR

def cosTheta(jet):
    return jet.P4().CosTheta()

class Cut:
    def __init__(self, description: str, func):
        self.description = description
        self.func = func

    def apply(self, input_dict: Dict[str, Any]):
        return self.func(input_dict)
    
    def __str__(self):
        return self.description
    
class EventSelection:
    def __init__(self, cuts: List[Cut], active_indices: Union[None, List[int]]):
        self.cuts = cuts
        if active_indices is None:
            active_indices = [n for n in range(len(cuts))]
        self.active_indices = active_indices
        self.n_passed = np.zeros(len(cuts))
        self.n_failed = np.zeros(len(cuts))
        print('EVENT SELECTION:')
        for i, c in enumerate(self.cuts):
            if i in self.active_indices:
                status = 'ON'
            else:
                status = 'OFF'
            print(f'\t{i}: {c} -- {status}')
        print()
    
    def apply(self, input_dict: Dict[str, Any]):
        selected = True
        for i, c in enumerate(self.cuts):
            if i in self.active_indices:
                if c.apply(input_dict):
                    self.n_passed[i] += 1
                else:
                    selected = False
                    self.n_failed[i] += 1
                    break
        return selected

    def efficiency(self):
        n_total = self.n_passed + self.n_failed
        active_mask = n_total > 0
        efficiency = np.full(len(self.cuts), fill_value=-1.0)
        efficiency[active_mask] = self.n_passed[active_mask] / n_total[active_mask]
        return efficiency
    
    def print_efficiency(self):
        efficiency = self.efficiency()
        print('EVENT SELECTION EFFICIENCY:')
        for i, c in enumerate(self.cuts):
            if i in self.active_indices:
                eff_s = 'NA'
                if efficiency[i] != -1.0:
                    eff_s = f'{efficiency[i]:.2%}'
                print(f'\t{i} -- {c}: {eff_s}')
        print()

def scale(f: TFile, luminosity: float, cross_section: float):
    keys = [k.GetName() for k in f.GetListOfKeys()]
    for key in keys:
        h = f.Get(key)
        if h.GetEntries() > 0:
            h.Scale(luminosity * cross_section / h.GetEntries())
            h.Write("", TObject.kOverwrite)

def write_histogram(input, output, cut_indices: Union[None, List[int]], n_events: int, energy: float, luminosity: float, cross_section: float):
    f=TFile(input)
    output=TFile(output,"RECREATE")	

    tree=f.Get("Delphes")

    h_missingE = TH1F('missingE', 'missingE;E_miss(GeV);Events', 200, 0, 5000)
    h_missingM = TH1F('missingM', 'missingM;M_miss(GeV);Events', 200, 0, 5000)

    h_j1_cosTheta = TH1F('j1_cosTheta', 'j1_cosTheta;cos(\\theta);Events', 20, -1, 1)
    h_j2_cosTheta = TH1F('j2_cosTheta', 'j2_cosTheta;cos(\\theta);Events', 20, -1, 1)
    h_j1_pT = TH1F('j1_pT', 'j1_pT;pT(GeV);Events', 20, 0, 6000)
    h_j2_pT = TH1F('j2_pT', 'j2_pT;pT(GeV);Events', 20, 0, 6000)
    h_jj_deltaR = TH1F('jj_deltaR', 'jj_deltaR;\\DeltaR{j1, j2};Events', 20, 0, 3.2)
    h_jj_M = TH1F('jj_M', 'jj_M;M(GeV);Events', 20, 0, 6000)
    h_jj_pT = TH1F('jj_pT', 'jj_pT;pT(GeV);Events', 20, 0, 6000)

    h_n_jets = TH1F('n_jets', 'n_jets;multiplicity;Events', 5, -0.5, 4.5)


    if n_events == -1:
        n_events = tree.GetEntries()
    else:
        n_events = min(tree.GetEntries(), n_events)
    events = range(n_events)
    if has_rich:
        events = progress(events, description=f"Writing to {output.GetName()}...")

    # Define event selection.
    cuts = []
    cuts.append(Cut('n(leptons) == 0', lambda d: d['n_leptons'] == 0))
    cuts.append(Cut('n(jets) == 2', lambda d: d['n_jets'] == 2))
    cuts.append(Cut('M_miss > 200 GeV', lambda d: d['missing_mass'] > 200))
    cuts.append(Cut('|cos(theta_j)| < 0.8', lambda d: all(abs(d[f'jet_{i}'].P4().CosTheta()) < 0.8 for i in (1,2))))
    selection = EventSelection(cuts, cut_indices)
    for event in events:
        tree.GetEntry(event)

        # First, check if the event passes the selection.
        n_leptons = len(tree.Electron) + len(tree.Muon)

        # Collect jets.
        jets = tree.VLCjetR12N2
        n_jets = len(jets)
        if n_jets == 2:
            jet_1 = min(jets[0], jets[1], key=lambda j: j.PT)
            jet_2 = max(jets[0], jets[1], key=lambda j: j.PT)
            jj = jet_1.P4() + jet_2.P4()
            mass_jj = jj.M()

            # Find M(nunu)
            CM = TLorentzVector()
            CM.SetE(1e3 * energy) # TeV -> GeV
            nunu = CM - jj
            missing_mass = nunu.M()
        else:
            jet_1, jet_2, missing_mass = None, None, None

        input_dict = {'n_leptons': n_leptons,
                        'n_jets': n_jets,
                        'jet_1': jet_1,
                        'jet_2': jet_2,
                        'missing_mass': missing_mass,
                        }
        if not selection.apply(input_dict):
            continue
        
        # Second, fill histograms.
        h_n_jets.Fill(n_jets)
        if jet_1:
            h_j1_cosTheta.Fill(cosTheta(jet_1))
            h_j1_pT.Fill(jet_1.PT)
        if jet_2:
            h_j2_pT.Fill(jet_2.PT)
            h_j2_cosTheta.Fill(cosTheta(jet_2))

        if jet_1 and jet_2:
            h_jj_deltaR.Fill(deltaR(jet_1, jet_2))
            h_jj_M.Fill(mass_jj)
            h_jj_pT.Fill(jj.Pt())

            h_missingE.Fill(nunu.E())
            h_missingM.Fill(missing_mass)
    
    selection.print_efficiency()
    output.Write()
    scale(output, luminosity, cross_section)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='*', help='List of madgraph output directories or root files.')
    parser.add_argument('--output', '-o', help='Output directory', default='histograms')
    parser.add_argument('--force_overwite', '-f', action='store_true')
    parser.add_argument('--cuts', '-c', type=lambda s: [int(item) for item in s.split(',')], help='"," delimited list of cut indices to use (starting from 0).', default=None)
    parser.add_argument('--n_events', '-n', default=-1, type=int)
    parser.add_argument('--energy', '-e', default=6, help='cm energy in TeV')
    args = parser.parse_args()

    cross_section_path = Path.cwd() / 'cross_section.yaml'
    luminosity_path = Path('lumi.yaml')
    energy = args.energy
    
    with cross_section_path.open('r') as f:
        cross_sections = yaml.load(f, Loader=yaml.SafeLoader)
    with luminosity_path.open('r') as f:
        luminosities = yaml.load(f, Loader=yaml.SafeLoader)
    try:
        luminosity = luminosities[energy]
    except:
        print(f'ERROR: "{energy}" [TeV] not found in {luminosity_path}!')

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_paths = [str(output_dir / f'{Path(i).stem}.root') for i in args.input]
    for i, o in zip(args.input, output_paths):
        if not args.force_overwite:
            if Path(o).exists():
                print(f'{o} already exists, skipping...')
                continue
        if Path(i).is_dir():
            i = str(Path(i) / 'Events' / 'run_01' / 'unweighted_events.root')
        process = Path(o).stem
        try:
            cross_section = cross_sections[process]
        except:
            print(f'ERROR: "{process}" not found in {cross_section_path}!')
        write_histogram(i, o, args.cuts, args.n_events, args.energy, luminosity, cross_section)

    
