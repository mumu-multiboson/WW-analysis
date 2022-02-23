from multiprocessing import Event
from pathlib import Path
from select import select
from typing import Any, Dict, List, Tuple, Union
try:
    from rich.progress import track as progress
    has_rich=True
except ImportError:
    print('To include progress bar, run `pip install rich`')
    has_rich=False
import yaml
import numpy as np
#---------------------------------------------------------
from ROOT import *

gSystem.Load("libDelphes")
gStyle.SetOptStat(0)

#---------------------------------------------------------
#return TLorentzVector corresponding to sum of inputs

def parentConstructor(a,b):
    return a.P4()+b.P4()

#---------------------------------------------------------
#returns subset of input collection passing cut string ('x' is placeholder for item in input)
#usage: selector(tree.Electron,'x.PT>50')

def selector(input,cutString='x.PT>20'):
    return [x for x in input if eval(cutString)]

def getParents(p, tree):
    result=[p]

    motherIndices=[]
    if p.M1!=-1 and tree.Particle[p.M1].PID==p.PID:
        motherIndices.append(p.M1)
    if p.M2!=-1 and tree.Particle[p.M2].PID==p.PID:
        motherIndices.append(p.M2)
    result+=[getParents(tree.Particle[i], tree) for i in motherIndices]

    return result

def isBeamRemnant(p, tree):
    parents=getParents(p, tree)
    while type(parents)==type([]): parents=parents[-1]
    return parents.Status==4

def getWZQuarks(tree, p) -> Tuple:
    if abs(p.PID) not in (23, 24):
        return tuple()
    D1 = tree.Particle[p.D1]
    D2 = tree.Particle[p.D2]
    if (abs(D1.PID) < 8) and (abs(D2.PID) < 8):
        return (D1, D2)
    quarks = getWZQuarks(tree, D1)
    if len(quarks) == 0:
        quarks = getWZQuarks(tree, D2)
    return quarks

def deltaR(p1, p2):
    dR = p1.P4().DeltaR(p2.P4())
    return dR


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

def write_histogram(input, output, cut_indices: Union[None, List[int]], n_events: int):
    f=TFile(input)
    output=TFile(output,"RECREATE")	

    tree=f.Get("Delphes")

    
    h=TH1F('h','',200,0,10000)

    T_e_pT = TH1F('T_e_Pt', 'Truth_electron;pT(GeV);Events', 200, 0, 2500)
    T_e_p = TH1F('T_e_p', 'Truth_electron;p(GeV);Events', 200, 0, 3000)
    T_e_eta = TH1F('T_e_eta', 'Truth_electron;Eta;Events', 20, -4, 4)
    T_e_multiplicity = TH1F('T_e_multiplicity', 'Truth_electron;multiplicity;Events', 5, -.5, 4.5)
    T_e_p_eta = TH2F('T_e_p_eta','Truth_e;eta;P(GeV)',5,-5,5,200,0,4500)
    T_e_pT_eta = TH2F('T_e_pT_eta','Truth_e;eta;pT(GeV)',5,-5,5,200,0,4500)

    T_mu_pT = TH1F('T_mu_Pt', 'Truth_muon;pT(GeV);Events', 200, 0, 3000)
    T_mu_p = TH1F('T_mu_p', 'Truth_muon;p(GeV);Events', 200, 0, 3500)
    T_mu_eta = TH1F('T_mu_eta', 'Truth_muon;Eta;Events', 20, -4, 4)
    T_mu_multiplicity = TH1F('T_mu_multiplicity', 'Truth_muon;multiplicity;Events', 5, -.5, 4.5)
    T_mu_p_eta = TH2F('T_mu_p_eta','Truth_mu;eta;P(GeV)',5,-5,5,200,0,4500)
    T_mu_pT_eta = TH2F('T_mu_pT_eta','Truth_mu;eta;pT(GeV)',5,-5,5,200,0,4500)

    T_W_pT = TH1F('T_W_Pt', 'Truth_W;pT(GeV);Events', 200, 0, 6000)
    T_W_p = TH1F('T_W_p', 'Truth_W;p(GeV);Events', 200, 0, 7000)
    T_W_eta = TH1F('T_W_eta', 'Truth_W;Eta;Events', 20, -4, 4)
    T_W_multiplicity = TH1F('T_W_multiplicity', 'Truth_W;multiplicity;Events', 5, -.5, 4.5)
    T_W_p_eta = TH2F('T_W_p_eta','Truth_W;eta;P(GeV)',5,-5,5,200,0,4500)    
    T_W_pT_eta = TH2F('T_W_pT_eta','Truth_W;eta;pT(GeV)',5,-5,5,200,0,4500)

    T_z_pT = TH1F('T_z_Pt', 'Truth_Z;pT(GeV);Events', 200, 0, 5000)
    T_z_p = TH1F('T_z_p', 'Truth_Z;p(GeV);Events', 200, 0, 5000)
    T_z_eta = TH1F('T_z_eta', 'Truth_Z;Eta;Events', 20, -4, 4)
    T_z_multiplicity = TH1F('T_z_multiplicity', 'Truth_Z;multiplicity;Events', 5, -.5, 4.5)
    T_z_p_eta = TH2F('T_z_p_eta','Truth_Z;eta;P(GeV)',5,-5,5,200,0,4500)
    T_z_pT_eta = TH2F('T_z_pT_eta','Truth_Z;eta;pT(GeV)',5,-5,5,200,0,4500)

    T_gamma_pT = TH1F('T_gamma_Pt', 'Truth_photon;pT(GeV);Events', 200, 0, 400)
    T_gamma_p = TH1F('T_gamma_p', 'Truth_photon;p(GeV);Events', 200, 0, 400)
    T_gamma_eta = TH1F('T_gamma_eta', 'Truth_photon;Eta;Events', 20, -4, 4)
    T_gamma_multiplicity = TH1F('T_gamma_multiplicity', 'Truth_photon;multiplicity;Events', 6, -.5, 5.5)
    T_gamma_p_eta = TH2F('T_gamma_p_eta','Truth_gamma;eta;P(GeV)',5,-5,5,200,0,4500)
    T_gamma_pT_eta = TH2F('T_gamma_pT_eta','Truth_gamma;eta;pT(GeV)',5,-5,5,200,0,4500)

    T_missingEt = TH1F('T_missingEt', 'Truth_MissingEt;Et_miss(GeV);Events', 200, 0, 5000)
    T_missingE = TH1F('T_missingE', 'Truth_MissingE;E_miss(GeV);Events', 200, 0, 5000)
    T_missingM = TH1F('T_missingM', 'Truth_MissingM;M_miss(GeV);Events', 200, 0, 6000)

    T_V_qDeltaR = TH1F('T_V_qDeltaR', 'Truth_V_qDeltaR;\\DeltaR_{q1, q2};Events', 20, 0, 3.2)
    T_V_q1_p = TH1F('T_V_q1_p', 'Truth_V_q1_p;P(GeV);Events', 30, 0, 4000)
    T_V_q2_p = TH1F('T_V_q2_p', 'Truth_V_q2_p;P(GeV);Events', 30, 0, 4000)
    T_V_cosTheta = TH1F('T_V_cosTheta', 'Truth_V_cosTheta;cos(\\theta);Events', 20, -1, 1)
    T_VV_deltaR = TH1F('T_VV_deltaR', 'Truth_VV_deltaR;\\DeltaR{W/Z, W/Z};Events', 20, 0, 3.2)
    T_VV_M = TH1F('T_VV_M', 'Truth_VV_M;M(GeV);Events', 20, 0, 6000)
    T_VV_pT = TH1F('T_VV_pT', 'Truth_VV_pT;pT(GeV);Events', 20, 0, 6000)

    T_nunu_M = TH1F('T_nunu_M', 'Truth_nunu_M; M_{\\nu\\nu}(GeV);Events', 20, 0, 6000)

    if n_events == -1:
        n_events = tree.GetEntries()
    else:
        n_events = min(tree.GetEntries(), n_events)
    events = range(n_events)
    if has_rich:
        events = progress(events, description=f"Writing to {output.GetName()}...")

    # Define event selection.
    cuts = []
    cuts.append(Cut('n(leptons) == 0', lambda d: d['leptons'] == 0))
    cuts.append(Cut('n(hadronic W/Z) == 2', lambda d: len(d['hadronic_WZs']) == 2))
    cuts.append(Cut('M(nunu) > 200 GeV', lambda d: d['mass_nunu'] > 200))
    # cuts.append(Cut('|cos(theta_{W/Z})| < 0.8', lambda d: all(abs(d['hadronic_WZs'][i].P4().CosTheta()) < 0.8 for i in (0,1))))
    selection = EventSelection(cuts, cut_indices)
    for event in events:
        tree.GetEntry(event)

        # First, check if the event passes the selection.
        hadronic_WZs = []
        WZ_quarks = []
        leptons = 0
        neutrinos = []
        for p in tree.Particle:
            if abs(p.Status)==22 and abs(p.PID) in (23, 24):
                quarks = getWZQuarks(tree, p)
                if len(quarks) == 2:
                    WZ_quarks.append(quarks)
                    hadronic_WZs.append(p)
            elif p.Status == 1 and abs(p.PID) in (11, 13):
                leptons += 1
            elif p.Status == 1 and abs(p.PID) in (12, 14, 16):
                neutrinos.append(p)
        mass_nunu = 0.0
        if len(neutrinos) == 2:
            mass_nunu = (neutrinos[0].P4() + neutrinos[1].P4()).M()
    
        input_dict = {'leptons': leptons,
                        'hadronic_WZs': hadronic_WZs,
                        'mass_nunu': mass_nunu,
                        }
        if not selection.apply(input_dict):
            continue


        # Second, fill histograms.
        for WZ, quarks in zip(hadronic_WZs, WZ_quarks):
            T_V_qDeltaR.Fill(deltaR(quarks[0], quarks[1]))
            
            p1 = quarks[0].P4().P()
            p2 = quarks[1].P4().P()
            T_V_q1_p.Fill(max(p1, p2))
            T_V_q2_p.Fill(min(p1, p2))
            T_V_cosTheta.Fill(WZ.P4().CosTheta())

        T_VV_deltaR.Fill(deltaR(hadronic_WZs[0], hadronic_WZs[1]))
        VV = hadronic_WZs[0].P4() + hadronic_WZs[1].P4()
        T_VV_M.Fill(VV.M())
        T_VV_pT.Fill(VV.Pt())

        if len(neutrinos) == 2:
            T_nunu_M.Fill(mass_nunu)

        electrons = 0
        muons = 0
        W = 0
        z = 0
        photons = 0
        neutrino_Px = 0
        neutrino_Py = 0
        neutrino_Pz = 0
        #count = 0
        final_P4 = []
        P4_i = TLorentzVector(0,0,0,6000)   #TLorentzVector(px,py,pz,E)
        missingMass = TLorentzVector(0,0,0,0)
        for p in tree.Particle:
            final_P4.append(p.P4())
            if (p.Status==1 and abs(p.PID)==11):
                electrons = electrons + 1
                T_e_pT.Fill(p.PT)
                T_e_p.Fill(p.P4().P())
                T_e_eta.Fill(p.Eta)
                T_e_p_eta.Fill(p.Eta,p.P4().P())
                T_e_pT_eta.Fill(p.Eta,p.PT)
            elif (p.Status==1 and abs(p.PID)==13):
                muons = muons + 1
                T_mu_pT.Fill(p.PT)
                T_mu_p.Fill(p.P4().P())
                T_mu_eta.Fill(p.Eta)
                T_mu_p_eta.Fill(p.Eta,p.P4().P())
                T_mu_pT_eta.Fill(p.Eta,p.PT)
            elif (abs(p.Status)==22 and (abs(p.PID)==24)):
                W = W + 1
                T_W_pT.Fill(p.PT)
                T_W_p.Fill(p.P4().P())
                T_W_eta.Fill(p.Eta)
                T_W_p_eta.Fill(p.Eta,p.P4().P())
                T_W_pT_eta.Fill(p.Eta,p.PT)
            elif (abs(p.Status)==22 and abs(p.PID)==23):
                z = z + 1
                T_z_pT.Fill(p.PT)
                T_z_p.Fill(p.P4().P())
                T_z_eta.Fill(p.Eta)
                T_z_p_eta.Fill(p.Eta,p.P4().P())
                T_z_pT_eta.Fill(p.Eta,p.PT)
            elif (p.Status==1 and abs(p.PID==22)):
                photons = photons + 1
                T_gamma_pT.Fill(p.PT)
                T_gamma_p.Fill(p.P4().P())
                T_gamma_eta.Fill(p.Eta)
                T_gamma_p_eta.Fill(p.Eta,p.P4().P())
                T_gamma_pT_eta.Fill(p.Eta,p.PT)
            elif (p.Status ==1 and (abs(p.PID)==12 or abs(p.PID)==14 or abs(p.PID)==16)):
                neutrino_Px += p.Px
                neutrino_Py += p.Py
                neutrino_Pz += p.Pz

        
        
        T_e_multiplicity.Fill(electrons)
        T_mu_multiplicity.Fill(muons)
        T_W_multiplicity.Fill(W)
        T_z_multiplicity.Fill(z)
        T_gamma_multiplicity.Fill(photons)
                  
        T_missingEt.Fill((neutrino_Px**2 + neutrino_Py**2)**.5)
        T_missingE.Fill((neutrino_Px**2 + neutrino_Py**2 + neutrino_Pz**2)**.5)
    
    selection.print_efficiency()
    output.Write()


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='*', help='List of madgraph output directories or root files.')
    parser.add_argument('--output', '-o', help='Output directory', default='histograms')
    parser.add_argument('--force_overwite', '-f', action='store_true')
    parser.add_argument('--cuts', '-c', type=lambda s: [int(item) for item in s.split(',')], help='"," delimited list of cut indices to use (starting from 0).', default=None)
    parser.add_argument('--n_events', '-n', default=-1, type=int)
    args = parser.parse_args()

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
        write_histogram(i, o, args.cuts, args.n_events)

    
