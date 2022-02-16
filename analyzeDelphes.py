maxEvents=9E9
DEBUG=True

import pdb
from sys import argv
#---------------------------------------------------------
from ROOT import *
from ROOT import TLorentzVector

gSystem.Load("libDelphes")
gStyle.SetOptStat(0)

#---------------------------------------------------------
def printHist(h):
    for i in range(h.GetNbinsX()+2):
        print h.GetBinContent(i),

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

if __name__=='__main__':

    #f=TFile('/Users/jstupak/ou/Snowmass2021/workArea/aQGC/MG5_aMC_v2_9_2/PROC_sm_10/Events/run_01/tag_1_delphes_events.root')
    #f=TFile('/raid01/users/kawale/MG5_aMC_v2_7_2/delphes/delphes.root')
    #f=TFile('/raid01/users/azartash/delphes/mumumumuww_vbseft10.root ')
    print str(argv[0])
    print str(argv[1])
    print str(argv[2])
    print argv
    f=TFile(str(argv[1]))
    #f=TFile('~/tmp/mumumumuww_schaneft10.root')
    #path = str(argv[1])
    #prefix = "/raid01/users/cwaits/MG5_aMC_v2_7_2/"+str(argv[2])
    #suffix = "/Events/run_01/delphes.root"
    #name = path[len(prefix):]
    #name = name[:-len(suffix)]


    tree=f.Get("Delphes")
    name=str(argv[2])
    output=TFile(name+".delphes.root","RECREATE")

    #sets upper limit for bin range
    bin_range = 15000

    #declares truth-level pT, p, eta, and multiplicity histograms for e,mu,W,Z,gamma
    T_e_pT = TH1F('T_e_Pt','Truth Electron pT;pT (GeV);Events', 200, 0, bin_range)
    T_e_p = TH1F('T_e_p', 'Truth Electron Momentum;p (GeV);Events', 200, 0, bin_range)
    T_e_eta = TH1F('T_e_eta', 'Truth Electron Eta;Eta;Events', 20, -4, 4)
    T_e_multiplicity = TH1F('T_e_multiplicity', 'Truth Electron Multiplicity;Multiplicity;Events', 7, -.5, 6.5)

    T_mu_pT = TH1F('T_mu_Pt', 'Truth Muon pT;pT (GeV);Events', 200, 0, bin_range)
    T_mu_p = TH1F('T_mu_p', 'Truth Muon Momentum;p (GeV);Events', 200, 0, bin_range)
    T_mu_eta = TH1F('T_mu_eta', 'Truth Muon Eta;Eta;Events', 20, -4, 4)
    T_mu_multiplicity = TH1F('T_mu_multiplicity', 'Truth Muon Multiplicity;Multiplicity;Events', 7, -.5, 6.5)

    T_W_pT = TH1F('T_W_Pt', 'Truth W pT;pT (GeV);Events', 200, 0, bin_range)
    T_W_p = TH1F('T_W_p', 'Truth W Momentum;p (GeV);Events', 200, 0, bin_range)
    T_W_eta = TH1F('T_W_eta', 'Truth W Eta;Eta;Events', 20, -4, 4)
    T_W_multiplicity = TH1F('T_W_multiplicity', 'Truth W Multiplicity;Multiplicity;Events', 7, -.5, 6.5)

    T_z_pT = TH1F('T_z_Pt', 'Truth Z pT;pT (GeV);Events', 200, 0, bin_range)
    T_z_p = TH1F('T_z_p', 'Truth Z Momentum;p (GeV);Events', 200, 0, bin_range)
    T_z_eta = TH1F('T_z_eta', 'Truth Z Eta;Eta;Events', 20, -4, 4)
    T_z_multiplicity = TH1F('T_z_multiplicity', 'Truth Z Multiplicity;Multiplicity;Events', 7, -.5, 6.5)

    T_gamma_pT = TH1F('T_gamma_Pt', 'Truth Photon pT;pT (GeV);Events', 200, 0, bin_range)
    T_gamma_p = TH1F('T_gamma_p', 'Truth Photon Momentum;p (GeV);Events', 200, 0, bin_range)
    T_gamma_eta = TH1F('T_gamma_eta', 'Truth Photon Eta;Eta;Events', 20, -4, 4)
    T_gamma_multiplicity = TH1F('T_gamma_multiplicity', 'Truth Photon Mulitiplcity;Mulitplicity;Events', 7, -.5, 6.5)

    #missing energy histograms
    T_missingEt = TH1F('T_missingEt', ';Missing Transverse Energy (GeV);Events', 200, 0, bin_range)
    T_missingE = TH1F('T_missingE', ';Missing Energy (GeV);Events', 200, 0, bin_range)

    #declares Reco-level inclusive, leading, 2nd leading, and 3rd leanding histograms for e,mu,gamma
    R_e_pT_I = TH1F('R_e_Pt_I', ';pT (GeV);Events', 200, 0, bin_range)
    R_e_p_I = TH1F('R_e_p_I', ';p (GeV);Events', 200, 0, bin_range)
    R_e_eta_I = TH1F('R_e_eta_I', ';Eta;Events', 20, -4, 4)
    R_e_pT_L = TH1F('R_e_Pt_L', ';pT (GeV);Events', 200, 0, bin_range)
    R_e_eta_L = TH1F('R_e_eta_L', ';Eta;Events', 20, -4, 4)
    R_e_pT_2 = TH1F('R_e_Pt_2', ';pT (GeV);Events', 200, 0, bin_range)
    R_e_eta_2 = TH1F('R_e_eta_2', ';Eta;Events', 20, -4, 4)
    R_e_pT_3 = TH1F('R_e_Pt_3', ';pT (GeV);Events', 200, 0, bin_range)
    R_e_eta_3 = TH1F('R_e_eta_3', ';Eta;Events', 20, -4, 4)
    R_e_multiplicity = TH1F('R_e_multiplicity', ';Multiplicity;Events', 7, -.5, 6.5)

    R_mu_pT_I = TH1F('R_mu_Pt_I', ';pT (GeV);Events', 200, 0, bin_range)
    R_mu_p_I = TH1F('R_mu_p_I', ';p (GeV);Events', 200, 0, bin_range)
    R_mu_eta_I = TH1F('R_mu_eta_I', ';Eta;Events', 20, -10, 10)
    R_mu_pT_L = TH1F('R_mu_Pt_L', ';pT (GeV);Events', 200, 0, bin_range)
    R_mu_eta_L = TH1F('R_mu_eta_L', ';Eta;Events', 20, -4, 4)
    R_mu_pT_2 = TH1F('R_mu_Pt_2', ';pT (GeV);Events', 200, 0, bin_range)
    R_mu_eta_2 = TH1F('R_mu_eta_2', ';Eta;Events', 20, -4, 4)
    R_mu_pT_3 = TH1F('R_mu_Pt_3', ';pT (GeV);Events', 200, 0, bin_range)
    R_mu_eta_3 = TH1F('R_mu_eta_3', ';Eta;Events', 20, -4, 4)
    R_mu_multiplicity = TH1F('R_mu_multiplicity', ';Multiplicity;Events', 7, -.5, 6.5)

    R_gamma_pT_I = TH1F('R_gamma_Pt_I', ';pT (GeV);Events', 200, 0, bin_range)
    R_gamma_p_I = TH1F('R_gamma_p_I', ';p (GeV);Events', 200, 0, bin_range)
    R_gamma_E_I = TH1F('R_gamma_E_I', ';Energy (GeV);Events', 200, 0, bin_range)
    R_gamma_eta_I = TH1F('R_gamma_eta_I', ';Eta;Events', 20, -4, 4)
    R_gamma_pT_L = TH1F('R_gamma_Pt_L', ';pT (GeV);Events', 200, 0, bin_range)
    R_gamma_E_L = TH1F('R_gamma__E_L', ';Energy (GeV);Events', 200, 0, bin_range)
    R_gamma_eta_L = TH1F('R_gamma_eta_L', ';Eta;Events', 20, -4, 4)
    R_gamma_pT_2 = TH1F('R_gamma_Pt_2', ';pT (GeV);Events', 200, 0, bin_range)
    R_gamma_eta_2 = TH1F('R_gamma_eta_2', ';Eta;Events', 20, -4, 4)
    R_gamma_pT_3 = TH1F('R_gamma_Pt_3', ';pT (GeV);Events', 200, 0, bin_range)
    R_gamma_eta_3 = TH1F('R_gamma_eta_3', ';Eta;Events', 20, -4, 4)
    R_gamma_multiplicity = TH1F('R_gamma_multiplicity', ';Multiplicity;Events', 5, -.5, 4.5)

    #histograms for OS pairs
    R_ee_pT = TH1F('ee_pT', ';pT (GeV);Events', 200, 0, bin_range)
    R_ee_eta = TH1F('ee_Eta', ';Eta;Events', 20, -4, 4)
    R_ee_mass = TH1F('ee_mass', ';Mass (GeV);Events', 200, 0, 200)
    R_ee_deltaEta = TH1F('ee_Delta Eta', ';Delta Eta;Events', 20, -4, 4)
    R_ee_multiplicity = TH1F('multiplicity', ';Multiplicity;Events', 5, -.5, 4.5)

    R_mumu_pT = TH1F('mumu_pT', ';pT (GeV);Events', 200, 0, bin_range)
    R_mumu_eta = TH1F('mumu_Eta', ';Eta;Events', 20, -4, 4)
    R_mumu_mass = TH1F('mumu_mass', ';Mass (GeV);Events', 200, 0, 200)
    R_mumu_deltaEta = TH1F('mumu_Delta Eta', ';Delta Eta;Events', 20, -4, 4)
    R_mumu_multiplicity = TH1F('mumu_multiplicity', ';Multiplicity;Events', 5, -.5, 4.5)

    R_emu_pT = TH1F('emu_pT', ';pT (GeV);Events', 200, 0, bin_range)
    R_emu_eta = TH1F('emu_Eta', ';Eta;Events', 20, -4, 4)
    R_emu_mass = TH1F('emu_mass', ';Mass (GeV);Events', 200, 0, 200)
    R_emu_deltaEta = TH1F('emu_Delta Eta', ';Delta Eta;Events', 20, -4, 4)
    R_emu_multiplicity = TH1F('emu_multplicity', ';Multiplicity;Events', 5, -.5, 4.5)

    #histograms for missing energy
    R_missingET = TH1F('R_MissingET', ';Missing Transverse Energy (GeV);Events', 200, 0, bin_range)
    R_missingE = TH1F('R_MissingE', ';Missing Energy (GeV);Events', 200, 0, bin_range)
    R_missingMass = TH1F('R_MissingMass', ';Missing Mass (GeV);Events' , 200, 0, bin_range)

    #histograms for beam remnants
    T_beamRemnants_pT = TH1F('T_beamRemnants_Pt', ';pT (GeV);Events', 200, 0, bin_range)
    T_beamRemnants_p = TH1F('T_beamRemnants_p', ';p (GeV);Events', 200, 0, bin_range)
    T_beamRemnants_eta = TH1F('T_beamRemnants_eta', ';Eta;Events', 20, -10, 10)
    T_beamRemnants_multiplicity = TH1F('T_beamRemnants_multiplicity', ';Multiplicity;Events', 7, -.5, 6.5)
    Test_beamRemnants_pT = TH1F('Test_beamRemnants_Pt', ';pT (GeV);Events', 30, 0, 3)

    for event in range(min(tree.GetEntries(),maxEvents)):
        tree.GetEntry(event)

        #truth-level
        T_electrons = 0
        T_muons = 0
        W = 0
        z = 0
        T_photons = 0
        neutrino_Px = 0
        neutrino_Py = 0
        neutrino_Pz = 0
        beamRemnants = 0
        
        for p in tree.Particle:
            if (p.Status==1 and abs(p.PID)==11):
                T_electrons = T_electrons + 1
                T_e_pT.Fill(p.PT)
                T_e_p.Fill(p.P4().P())
                T_e_eta.Fill(p.Eta)
            elif (p.Status==1 and abs(p.PID)==13):
                T_muons = T_muons + 1
                T_mu_pT.Fill(p.PT)
                T_mu_p.Fill(p.P4().P())
                T_mu_eta.Fill(p.Eta)
                if isBeamRemnant(p):
                    beamRemnants = beamRemnants + 1
                    T_beamRemnants_pT.Fill(p.PT)
                    T_beamRemnants_p.Fill(p.P4().P())
                    T_beamRemnants_eta.Fill(p.Eta)
                    if (p.PT <= 2.0):
                        Test_beamRemnants_pT.Fill(p.PT)
            elif (abs(p.Status)==22 and (abs(p.PID)==24)):
                W = W + 1
                T_W_pT.Fill(p.PT)
                T_W_p.Fill(p.P4().P())
                T_W_eta.Fill(p.Eta)
            elif (abs(p.Status)==22 and abs(p.PID)==23):
                z = z + 1
                T_z_pT.Fill(p.PT)
                T_z_p.Fill(p.P4().P())
                T_z_eta.Fill(p.Eta)
            elif (p.Status==1 and abs(p.PID==22)):
                T_photons = T_photons + 1
                T_gamma_pT.Fill(p.PT)
                T_gamma_p.Fill(p.P4().P())
                T_gamma_eta.Fill(p.Eta)
            elif (p.Status ==1 and (abs(p.PID)==12 or abs(p.PID)==14 or abs(p.PID)==16)):
                neutrino_Px += p.Px
                neutrino_Py += p.Py
                neutrino_Pz += p.Pz

        T_e_multiplicity.Fill(T_electrons)
        T_mu_multiplicity.Fill(T_muons)
        T_W_multiplicity.Fill(W)
        T_z_multiplicity.Fill(z)
        T_gamma_multiplicity.Fill(T_photons)
        T_beamRemnants_multiplicity.Fill(beamRemnants)
        T_missingEt.Fill((neutrino_Px**2 + neutrino_Py**2)**.5)
        T_missingE.Fill((neutrino_Px**2 + neutrino_Py**2 + neutrino_Pz**2)**.5)

        #reco-level
        R_missingET.Fill(tree.MissingET[0].MET)
        R_missingE.Fill(tree.MissingET[0].P4().P())
        #final and initial-state 4-vectors
        P4_i = TLorentzVector(0,0,0,10000)
        P4_f = []
        #multiplicity counters
        electrons = 0
        muons = 0
        photons = 0
        e_list1 = []
        e_list2 = []
        mu_list1 = []
        mu_list2 = []
        gamma_list1 = []
        gamma_list2 = []
        #for OS pairs
        leptons = []
        #fills electron histograms
        for p in tree.Electron:
            electrons = electrons + 1
            R_e_pT_I.Fill(p.PT)
            R_e_p_I.Fill(p.P4().P())
            R_e_eta_I.Fill(p.Eta)
            #assigns each particle to a list to be sorted by pT after going through the event to get leading, 2nd leading, and 3rd leading
            e_list1.append(p.PT)
            e_list2.append(p.Eta)
            leptons.append(p)
            P4_f.append(p.P4())
                
        for p in tree.Muon:
            muons = muons + 1
            R_mu_pT_I.Fill(p.PT)
            R_mu_p_I.Fill(p.P4().P())
            R_mu_eta_I.Fill(p.Eta)
            mu_list1.append(p.PT)
            mu_list2.append(p.Eta)
            leptons.append(p)
            P4_f.append(p.P4())

        for p in tree.Photon:
            photons = photons + 1
            R_gamma_pT_I.Fill(p.PT)
            R_gamma_p_I.Fill(p.P4().P())
            R_gamma_E_I.Fill(p.E)
            R_gamma_eta_I.Fill(p.Eta)
            gamma_list1.append(p.PT)
            gamma_list2.append(p.Eta)
            P4_f.append(p.P4())

        R_e_multiplicity.Fill(electrons)
        R_mu_multiplicity.Fill(muons)
        R_gamma_multiplicity.Fill(photons)
        final_P4 = TLorentzVector(0,0,0,0)
        for i in P4_f:
            final_P4 = final_P4 + i
        missingMass = P4_i - final_P4
        R_missingMass.Fill(missingMass.M())

        #sorts particles by pT and then fills the leading, 2nd order, and 3rd order histograms for pT and eta
        if (len(e_list1) != 0):
            z = zip(e_list1, e_list2)
            zed = sorted(z, key=lambda x:x[0])
            zed.reverse()
            if (len(e_list1) == 1):
                R_e_pT_L.Fill(zed[0][0])
                R_e_eta_L.Fill(zed[0][1])
            if (len(e_list1) == 2):
                R_e_pT_2.Fill(zed[1][0])
                R_e_eta_2.Fill(zed[1][1])
            if (len(e_list1) == 3):
                R_e_pT_3.Fill(zed[2][0])
                R_e_eta_3.Fill(zed[2][1])

        if (len(mu_list1) != 0):
            z = zip(mu_list1, mu_list2)
            zed = sorted(z, key=lambda x:x[0])
            zed.reverse()
            if (len(mu_list1) == 1):
                R_mu_pT_L.Fill(zed[0][0])
                R_mu_eta_L.Fill(zed[0][1])
            if (len(mu_list1) == 2):
                R_mu_pT_2.Fill(zed[1][0])
                R_mu_eta_2.Fill(zed[1][1])
            if (len(mu_list1) == 3):
                R_mu_pT_3.Fill(zed[2][0])
                R_mu_eta_3.Fill(zed[2][1])

        if (len(gamma_list1) != 0):
            z = zip(gamma_list1, gamma_list2)
            zed = sorted(z, key=lambda x:x[0])
            zed.reverse()
            if (len(gamma_list1) == 1):
                R_gamma_pT_L.Fill(zed[0][0])
                R_gamma_eta_L.Fill(zed[0][1])
            if (len(gamma_list1) == 2):
                R_gamma_pT_2.Fill(zed[1][0])
                R_gamma_eta_2.Fill(zed[1][1])
            if (len(gamma_list1) == 3):
                R_gamma_pT_3.Fill(zed[2][0])
                R_gamma_eta_3.Fill(zed[2][1])

        #electrons=selector(tree.Electron,'x.PT>5 and abs(x.Eta)<2')
        #muons=selector(tree.Muon,'x.PT>5 and abs(x.Eta)<2')

        #fMuons=selector(tree.Muon,'abs(x.Eta)>2')
        #fMuons+=selector(tree.Electron,'abs(x.Eta)>2 and x.Particle.GetObject().PID==11 and x.Particle.GetObject().Status==1 and x.Particle.GetObject().M1<5')  #this is a hack - not a typo
        ee = 0
        mumu = 0
        emu = 0
        if (len(leptons) > 1):
            #creates every possible unique OS lepton pair from event
            i = 0
            pairs1 = []
            pairs2 = []
            for i in range(len(leptons)):
                j = i + 1
                for j in range(i+1, len(leptons)):
                    p1 = leptons[i]
                    p2 = leptons[j]
                    #only selects OS pairs
                    if (p1.Charge != p2.Charge):
                        pairs1.append(leptons[i])
                        pairs2.append(leptons[j])
                    j = j + 1
                i = i + 1
            #creates two lists to hold the invariant mass of the pair, the difference between mass of the pair and the Z mass, and each particle of the pair
            a = []
            b = []
            for i in range(len(pairs1)):
                p1 = pairs1[i]
                p2 = pairs2 [i]
                a.append(p1.P4().M() + p2.P4().M() - 91)
                b.append(p1.P4().M() + p2.P4().M())

            #zips the 4 lists together in a tuple and sorts the list of pairs from closest to Z mass to furthest.   
            pairs = zip(a,b,pairs1,pairs2)
            pairs = sorted(pairs, key=lambda x:x[0])
            #loops through the list of sorted pairs and picks the pair closest to the Z mass that any 1 lepton participates in
            #prevents any 1 lepton from being in more than one pair. 
            consumed = []
            master_pairs = []
            for i in range(len(pairs)):
                if ((pairs[i][2] not in consumed) and (pairs[i][3] not in consumed)):
                    master_pairs.append(pairs[i])
                    consumed.append(pairs[i][2])
                    consumed.append(pairs[i][3])
            #sorts the pairs by type: ee, mumu, or emu and fills appropriate histograms
            for i in range(len(master_pairs)):
                l1 = master_pairs[i][2]
                l2 = master_pairs[i][3]
                if ((type(l1)==type(Electron())) and type(l2)==type(Electron())):
                    R_ee_pT.Fill(l1.PT + l2.PT)
                    R_ee_eta.Fill((l1.Eta + l2.Eta)/2.0)
                    R_ee_mass.Fill(l1.P4().M() + l2.P4().M())
                    if (l1.Charge < l2.Charge):
                        R_ee_deltaEta.Fill(l1.Eta - l2.Eta)
                    else:
                        R_ee_deltaEta.Fill(l2.Eta - l1.Eta)
                    ee = ee +1
                elif ((type(l1)==type(Muon())) and type(l2)==type(Muon())):
                    R_mumu_pT.Fill(l1.PT + l2.PT)
                    R_mumu_eta.Fill((l1.Eta + l2.Eta)/2.0)
                    R_mumu_mass.Fill(l1.P4().M() + l2.P4().M())
                    if (l1.Charge < l2.Charge):
                        R_mumu_deltaEta.Fill(l1.Eta - l2.Eta)
                    else:
                        R_mumu_deltaEta.Fill(l2.Eta - l1.Eta)
                    mumu = mumu +1
                elif ((type(l1)==type(Electron())) and type(l2)==type(Muon())):
                    R_emu_pT.Fill(l1.PT + l2.PT)
                    R_emu_eta.Fill((l1.Eta + l2.Eta)/2.0)
                    R_emu_mass.Fill(l1.P4().M() + l2.P4().M())
                    if (l1.Charge < l2.Charge):
                        R_emu_deltaEta.Fill(l1.Eta - l2.Eta)
                    else:
                        R_emu_deltaEta.Fill(l2.Eta - l1.Eta)
                    emu = emu +1
                elif ((type(l1)==type(Muon())) and type(l2)==type(Electron())):
                    R_emu_pT.Fill(l1.PT + l2.PT)
                    R_emu_eta.Fill((l1.Eta + l2.Eta)/2.0)
                    R_emu_mass.Fill(l1.P4().M() + l2.P4().M())
                    if (l1.Charge < l2.Charge):
                        R_emu_deltaEta.Fill(l1.Eta - l2.Eta)
                    else:
                        R_emu_deltaEta.Fill(l2.Eta - l1.Eta)
                    emu = emu +1
                                           
        R_ee_multiplicity.Fill(ee)
        R_mumu_multiplicity.Fill(mumu)
        R_emu_multiplicity.Fill(emu)

    output.Write()
