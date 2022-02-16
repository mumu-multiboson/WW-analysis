maxEvents=9E9
DEBUG=True
from pathlib import Path
try:
    from rich.progress import track as progress
    has_rich=True
except ImportError:
    has_rich=False

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

#---------------------------------------------------------

def getParents(p):
    result=[p]

    motherIndices=[]
    if p.M1!=-1 and tree.Particle[p.M1].PID==p.PID:
        motherIndices.append(p.M1)
    if p.M2!=-1 and tree.Particle[p.M2].PID==p.PID:
        motherIndices.append(p.M2)
    result+=[getParents(tree.Particle[i]) for i in motherIndices]

    return result

def isBeamRemnant(p):
    parents=getParents(p)
    while type(parents)==type([]): parents=parents[-1]
    return parents.Status==4

#---------------------------------------------------------

#---------------------------------------------------------

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    f=TFile(args.input)
    output=TFile(args.output,"RECREATE")	

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
    
    events = range(min(tree.GetEntries(),maxEvents))
    if has_rich:
        events = progress(events, description="Processing events...")
    for event in events:
        tree.GetEntry(event)

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

        P4_f = TLorentzVector(0,0,0,0)
        for i in final_P4:
            P4_f = P4_f + i
        missingMass = P4_i - P4_f
        T_missingM.Fill(missingMass.M())
	
        beamRemnants=[]
        for p in tree.Particle:
            if p.Status==1 and abs(p.PID)==13:
                if isBeamRemnant(p): beamRemnants.append(p)

        electrons=selector(tree.Electron,'x.PT>5 and abs(x.Eta)<2')
        muons=selector(tree.Muon,'x.PT>5 and abs(x.Eta)<2')

        #fMuons=selector(tree.Muon,'abs(x.Eta)>2')
        #fMuons+=selector(tree.Electron,'abs(x.Eta)>2 and x.Particle.GetObject().PID==11 and x.Particle.GetObject().Status==1 and x.Particle.GetObject().M1<5')  #this is a hack - not a typo

        leptons=electrons+muons

        Zs=[]
        consumed=[]
        for i1 in range(len(leptons)-1):
            if i1 in consumed: continue
            l1=leptons[i1]
            for i2 in range(i1+1,len(leptons)):
                if i2 in consumed: continue
                l2=leptons[i2]

                #pdb.set_trace()
                #print l1.Charge,l2.Charge #,l1.Charge!=-l2.Charge, (type(l1)==type(Electron())) != (type(l2)==type(Electron()))
                if l1.Charge!=-l2.Charge: continue
                if (type(l1)==type(Electron())) != (type(l2)==type(Electron())): continue

                Z=parentConstructor(l1,l2)
                #if 81<Z.M() and Z.M()<101:
                Zs.append(Z)
                consumed.append(i1)
                consumed.append(i2)
        for Z in Zs: h.Fill(Z.M())
    
    output.Write()
