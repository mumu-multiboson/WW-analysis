from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from functools import reduce
from operator import mul
from multiprocessing import Pool, Pipe
from multiprocessing.connection import Connection
from contextlib import ExitStack
import os

import numpy as np
import yaml
from ROOT import *
import shutil

gSystem.Load("libDelphes")
gStyle.SetOptStat(0)
try:
    from rich.progress import Progress
    has_rich=True
except ImportError:
    print('To include progress bar, run `python -m pip install --user rich`')
    has_rich=False

import logging
from utils import *
from reco_selection import get_cuts

def deltaR(p1, p2):
    dR = p1.P4().DeltaR(p2.P4())
    return dR

def cosTheta(jet):
    return jet.P4().CosTheta()

# csv_eff_output: Path = None, csv_abs_eff_output: Path = None, lock: Lock = None, 
def write_histogram(input: str, output: str, cut_indices: Union[None, List[int]], n_events: int, energy: float, luminosity: float, cross_section: float, pipe: Connection, std_pipe: Connection, cut_values: dict = {}, csv_eff_output: Path = None, csv_abs_eff_output: Path = None, lock = None, debug: bool = False):
    process_name = Path(output).stem
    f=TFile(input)
    print(f'Writing to {output}...')
    output=TFile(output,"RECREATE")	
    

    tree=f.Get("Delphes")

    h_missingE = TH1F('missingE', 'missingE;E_miss(GeV);Events', 30, 0, 6000)
    h_missingM = TH1F('missingM', 'missingM;M_miss(GeV);Events', 30, 0, 6000)

    h_j1_cosTheta = TH1F('j1_cosTheta', 'j1_cosTheta;cos(\\theta);Events', 20, -1, 1)
    h_j2_cosTheta = TH1F('j2_cosTheta', 'j2_cosTheta;cos(\\theta);Events', 20, -1, 1)
    h_j1_pT = TH1F('j1_pT', 'j1_pT;pT(GeV);Events', 20, 0, 1500)
    h_j2_pT = TH1F('j2_pT', 'j2_pT;pT(GeV);Events', 20, 0, 1500)
    h_jj_deltaR = TH1F('jj_deltaR', 'jj_deltaR;\\DeltaR{j1, j2};Events', 20, 0, 3.2)
    h_jj_M = TH1F('jj_M', 'jj_M;M(GeV);Events', 600, 0, 6000)
    h_jj_pT = TH1F('jj_pT', 'jj_pT;pT(GeV);Events', 30, 0, 3000)

    h_n_jets = TH1F('n_jets', 'n_jets;multiplicity;Events', 5, -0.5, 4.5)
    h_e_multiplicity = TH1F('e_multiplicity', 'Reco_electron;multiplicity;Events', 5, -.5, 4.5)
    h_mu_multiplicity = TH1F('mu_multiplicity', 'Reco_muon;multiplicity;Events', 5, -.5, 4.5)
    h_lepton_multiplicity = TH1F('lepton_multiplicity', 'Reco_lepton;multiplicity;Events', 5, -.5, 4.5)


    if n_events == -1:
        n_events = tree.GetEntries()
    else:
        n_events = min(tree.GetEntries(), n_events)
    events = range(n_events)

    # Define event selection.
    selection = EventSelection(get_cuts(**cut_values), cut_indices, std_pipe)
    for event in events:
        if not debug:
            pipe.send((n_events, 1))
        tree.GetEntry(event)

        # First, check if the event passes the selection.
        n_muons = 0
        n_electrons = 0
        for e in tree.Electron:
            # if e.P4().P() > 3:
            n_electrons += 1
        for m in tree.Muon:
            # if m.P4().P() > 3:
            n_muons += 1
        n_leptons = n_muons + n_electrons

        # Collect jets.
        jets = tree.VLCjetR10_inclusive
        n_jets = len(jets)
        if n_jets >= 2:
            jet_1 = jets[0]
            jet_2 = jets[1]
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
        h_e_multiplicity.Fill(n_electrons)
        h_mu_multiplicity.Fill(n_muons)
        h_lepton_multiplicity.Fill(n_leptons)
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
    
    if not debug:
        pipe.close()
        msg = selection.efficiency_msg()
        std_pipe.send(msg)

    def write_csv(out_path, relative):
        if out_path:
            csv = selection.efficiency_csv(relative=relative)
            csv = process_name + ',' + csv + '\n'
            print(f'writing {csv}')
            print(out_path)
            print(lock)
            if not debug:
                assert lock is not None
                lock.acquire()
            with out_path.open('a') as f:
                f.write(csv)
                f.flush()
                os.fsync(f)
            if not debug:
                lock.release()
    write_csv(out_path=csv_eff_output, relative=True)
    write_csv(out_path=csv_abs_eff_output, relative=False)

    output.Write()
    scale(output, luminosity, cross_section, n_events)


if __name__=='__main__':
    parse_args(write_histogram, 'reco_histograms')
