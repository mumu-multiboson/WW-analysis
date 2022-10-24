import ROOT
from ROOT import *
import array

ROOT.gROOT.LoadMacro('./share/AtlasStyle.C')
ROOT.gROOT.LoadMacro('./share/AtlasUtils.C')
ROOT.SetAtlasStyle()
gStyle.SetOptStat(0)




def loglables(h):
    h.GetXaxis().SetMoreLogLabels();
    h.GetXaxis().SetNoExponent(); 


def phi_mpi_pi(phi):
    if (phi >= PI) :
        phi -= 2.0*PI
    elif(phi < -PI):
        phi += 2.0*PI
    return phi

def stampLumiText(lumi, x, y, text, size):
    t = TLatex()
    t.SetNDC()
    #t.SetTextAlign(13)
    t.SetTextColor(kBlack)
    t.SetTextSize(size)
    #t.DrawLatex(x,y,"#int L dt = "+str(lumi)+" fb^{-1}, "+text)
    t.DrawLatex(x,y, text+", "+str(lumi)+" fb^{-1}")

def stampText(x, y, text, size):
    t = TLatex()
    t.SetNDC()
    #t.SetTextAlign(13)
    t.SetTextFont(42)
    t.SetTextColor(kBlack)
    t.SetTextSize(size)
    t.DrawLatex(x,y, text)

def stampText_colored(x, y, text, size, color):
    t1 = TLatex()
    t1.SetNDC()
    #t.SetTextAlign(13)
    t1.SetTextColor(color)
    t1.SetTextSize(size)
    t1.SetTextFont(62)
    t1.DrawLatex(x,y, text)


def normalize(h):
    h.Scale(1.0/h.Integral())


def setStyle_ratio(h, col, sty, x_axis):
    # h.Rebin(2)
    #h.Scale(lumi)
    h.SetMaximum(h.GetMaximum()*1000);
    h.SetMinimum(0.01)
    #h.GetYaxis().SetRangeUser(0.0,1.05);
    h.GetXaxis().SetRangeUser(0,6500);
    h.GetYaxis().SetTitle("Events");
    h.GetXaxis().SetTitle(x_axis);
    h.GetXaxis().SetTitleOffset(1.1);
    h.GetYaxis().SetTitleOffset(1.5);
    h.GetXaxis().SetLabelSize(0.05);
    h.GetXaxis().SetTitleSize(0.05);
    h.GetYaxis().SetLabelSize(0.05);
    h.GetYaxis().SetTitleSize(0.05);
    h.SetLineWidth(2);
    h.SetLineStyle(1);
    h.SetLineColor(1);
    h.SetMarkerStyle(sty);
    h.SetMarkerColor(col);
    h.SetFillColor(col);
    # h.SetFillColor(col-7);
    #h.SetFillStyle(3005);

def setStyle(h, col, sty, x_axis):
    # h.Rebin(2)
    #h.Scale(lumi)
    h.SetMaximum(h.GetMaximum()*1000);
    h.SetMinimum(0.01)
    #h.GetYaxis().SetRangeUser(0.0,1.05);
    h.GetXaxis().SetRangeUser(0,6500);
    h.GetYaxis().SetTitle("Events");
    h.GetXaxis().SetTitle(x_axis);
    h.GetXaxis().SetTitleOffset(0.1);
    h.GetYaxis().SetTitleOffset(0.1);
    h.GetXaxis().SetLabelSize(0.00);
    h.GetXaxis().SetTitleSize(0.04);
    h.GetYaxis().SetLabelSize(0.00);
    h.GetYaxis().SetTitleSize(0.04);
    h.SetLineWidth(2);
    h.SetLineStyle(1);
    h.SetLineColor(1);
    h.SetMarkerStyle(sty);
    h.SetMarkerColor(col);
    h.SetFillColor(col);
    # h.SetFillColor(col-7);
    #h.SetFillStyle(3005);

def setStyleDATA(h, col, sty, x_axis):
    h.SetMaximum(h.GetMaximum()*500);
    h.SetMinimum(0.02)
    #h.GetYaxis().SetRangeUser(0.0,1.05);
    #h.GetXaxis().SetRangeUser(1300,8000);
    h.GetYaxis().SetTitle("Events");
    h.GetXaxis().SetTitle(x_axis);
    h.GetXaxis().SetTitleOffset(1.3);
    h.GetYaxis().SetTitleOffset(1.5);
    h.GetXaxis().SetLabelSize(0.05);
    h.GetXaxis().SetTitleSize(0.05);
    h.GetYaxis().SetLabelSize(0.05);
    h.GetYaxis().SetTitleSize(0.05);
    h.SetLineWidth(2);
    h.SetLineStyle(1);
    h.SetLineColor(col);
    h.SetMarkerStyle(8);
    h.SetMarkerColor(col);

def ratio_style(hratio):

    #h1.GetYaxis().SetTitleSize(20);
    #h1.GetYaxis().SetTitleFont(43);
    #h1.GetYaxis().SetTitleOffset(1.55)

    hratio.GetYaxis().SetTitle("#frac{Data (full)}{Bkgkground}");

    hratio.GetYaxis().SetNdivisions(505);
    hratio.GetYaxis().SetTitleSize(25);
    hratio.GetYaxis().SetTitleFont(43);
    hratio.GetYaxis().SetTitleOffset(1.5);
    hratio.GetYaxis().SetLabelFont(43); # Absolute font size in pixel (precision 3)
    hratio.GetYaxis().SetLabelSize(25);


    # X axis ratio plot settings
    hratio.GetXaxis().SetTitleSize(30);
    hratio.GetXaxis().SetTitleFont(43);
    hratio.GetXaxis().SetTitleOffset(4);
    hratio.GetXaxis().SetLabelFont(43); # Absolute font size in pixel (precision 3)
    hratio.GetXaxis().SetLabelSize(25);


def drawIt_ratio(hbkg, heft, label_map, var_name): # wp, label, SR, suf, legend_suf):
    c = TCanvas("c", "", 1000, 1000);
    c.SetTopMargin(0.05)
    c.SetRightMargin(0.05)
    c.SetBottomMargin(0.16)
    c.SetLeftMargin(0.16)
    #c.SetLogx()
    #c.SetLogy()
    c.SetTickx(1)
    c.SetTicky(1)

    # loglables(hbkg)
    # loglables(hdata)


    pad1 = TPad("pad1", "pad1", 0, 0.35, 1, 1.0)
    pad1.SetBottomMargin(0); # Upper and lower plot are joined
    #pad1.SetGridx();        # Vertical grid
    pad1.SetLogy()
    pad1.Draw();           # Draw the upper pad: pad1
    pad1.cd();             # pad1 becomes the current pad


    l = TLegend(0.55,0.6,0.9,0.9)
    l.SetBorderSize(0)
    # l.SetTextFont(62)
    l.SetTextSize(0.045)

    bkg_stack =  THStack("bkg","Background Stack")
    for hname in hbkg:
        bkg_stack.Add(hbkg[hname])

    for hname in reversed(hbkg.keys()):
        print (hname)
        l.AddEntry(hbkg[hname], label_map[hname], "F")

    bkg_stack.SetMaximum(bkg_stack.GetMaximum()*10000)
    bkg_stack.SetMinimum(10)
    bkg_stack.Draw('hist')

    bkg_eft_stack =  THStack("sig_bkg","Signal+Bkg Stack")
    for hname in hbkg:
        bkg_eft_stack.Add(hbkg[hname])
    bkg_eft_stack.Add(heft)
    bkg_eft_stack.Draw('samehist')
    l.AddEntry(heft, "T1 (INT + QUAD)", "L")

    l.Draw()

    #stampText(0.7, 0.75, '#splitline{mean = %0.2f}{std = %0.2f}'%(h.GetMean(), h.GetStdDev()) ,0.04)

    gPad.RedrawAxis()
    ROOT.ATLAS_LABEL(0.2,0.88)
    stampText(0.4, 0.88, "", 0.045)
    stampText(0.2, 0.82, "#sqrt{s} = 6 TeV, 4 ab^{-1}", 0.04)
    # stampText_colored(0.2, 0.76, label, 0.035, kBlack)

    c.cd()
    pad2 = TPad("pad2", "pad2", 0, 0.02, 1, 0.35);
    pad2.SetTopMargin(0);
    pad2.SetBottomMargin(0.35);
    #pad2.SetGridx(); #vertical grid
    pad2.Draw();

    pad2.cd();       #pad2 becomes the current pad

    #line = TF1('line',"1",hbkg.GetBinLowEdge(1),hbkg.GetBinLowEdge(hbkg.GetNbinsX())+hbkg.GetBinWidth(hbkg.GetNbinsX()));
    line = TF1('line',"1",bkg_stack.GetHistogram().GetBinLowEdge(1),bkg_stack.GetHistogram().GetBinLowEdge(bkg_stack.GetHistogram().GetNbinsX())+bkg_stack.GetHistogram().GetBinWidth(bkg_stack.GetHistogram().GetNbinsX()));
    line.SetLineColor(kRed);
    #line.Draw();

    
    hratio = heft.Clone("hratio");
    hratio.SetLineColor(kAzure+1);
    #Define the ratio plot
    hratio.SetMinimum(0.5);  # Define Y..
    hratio.SetMaximum(1.50); # .. range
    hratio.Sumw2();
    hratio.SetStats(0);      # No statistics on lower plot
    #hratio.Add(hdata, -1)
    hratio.Divide(bkg_stack.GetStack().Last())
    hratio.SetMarkerStyle(34)
    hratio.SetMarkerColor(4)
    hratio.Draw("esame")       # Draw the ratio plot

    ratio_style(hratio)
    line.Draw("same")
    hratio.Draw("esame")
    gPad.RedrawAxis()

    pad1.cd()
    gPad.RedrawAxis()
    c.Draw()

    for i in [".png"]:
        c.SaveAs(var_name+i)
        # c.SaveAs('/home/elham/ttbarRes_updatedTools/run/dataMC_validation/'+SR+'_'+hname+'_'+wp+suf+i)



def drawIt(hbkg, heft, label_map, hname, ch_label): # wp, label, SR, suf, legend_suf):
    c = TCanvas("c", "", 1000, 1000);
    c.SetTopMargin(0.05)
    c.SetRightMargin(0.05)
    c.SetBottomMargin(0.16)
    c.SetLeftMargin(0.16)
    #c.SetLogx()
    c.SetLogy()
    c.SetTickx(1)
    c.SetTicky(1)

    # loglables(hbkg)
    # loglables(hdata)

    leg_y1 = 0.92 - (len(hbkg) + 1)*.04 
    l = TLegend(0.55,leg_y1,0.9,0.92)
    l.SetBorderSize(0)
    l.SetTextFont(42)
    l.SetTextSize(0.03)

    bkg_stack =  THStack("bkg","Background Stack")
    for hist in hbkg:
        bkg_stack.Add(hbkg[hist])

    for hist in reversed(hbkg.keys()):
        print (hist)
        l.AddEntry(hbkg[hist], label_map[hist], "F")

    bkg_stack.SetMaximum(bkg_stack.GetMaximum()*10000)
    bkg_stack.SetMinimum(10)
    bkg_stack.Draw('hist')
    bkg_stack.GetYaxis().SetTitle("Events")
    bkg_stack.GetXaxis().SetTitle("m_{WW} [GeV]")
    bkg_stack.GetXaxis().SetLabelSize(0.04)
    bkg_stack.GetYaxis().SetLabelSize(0.04)

    bkg_eft_stack =  THStack("sig_bkg","Signal+Bkg Stack")
    for hist in hbkg:
        bkg_eft_stack.Add(hbkg[hist])
    bkg_eft_stack.Add(heft)
    bkg_eft_stack.Draw('samehist')
    l.AddEntry(heft, "EFT #it{f}_{T1} = 1 TeV^{-4}", "L")

    l.Draw()


    #stampText(0.7, 0.75, '#splitline{mean = %0.2f}{std = %0.2f}'%(h.GetMean(), h.GetStdDev()) ,0.04)

    gPad.RedrawAxis()
    ROOT.ATLAS_LABEL(0.2,0.88)
    stampText(0.4, 0.88, "", 0.045)
    stampText(0.2, 0.83, "#sqrt{s} = 6 TeV, 4 ab^{-1}", 0.035)
    stampText_colored(0.2, 0.78, ch_label, 0.030, kBlack)

    for i in [".pdf"]:
        c.SaveAs(hname+i)
        # c.SaveAs('/home/elham/ttbarRes_updatedTools/run/dataMC_validation/'+SR+'_'+hname+'_'+wp+suf+i)



channel = "nunuww"
# channel = "mumuww"

if channel == "nunuww":

    # path = "/work/schuya/reco_histograms/all/"
    path = "/home/elham/WW-analysis/reco_histograms/all/"

    inFile_wzmunu = ROOT.TFile(path+'wzmunu_6tev.root', 'r')
    inFile_zzmumu = ROOT.TFile(path+'zzmumu_6tev.root', 'r')
    inFile_ggwpwm = ROOT.TFile(path+'ggwpwm_6tev.root', 'r')
    inFile_wpwmz_ztonunu = ROOT.TFile(path+'wpwmz_ztonunu_6tev.root', 'r')
    inFile_mumu_nunuww_SM = ROOT.TFile(path+'mumu_nunuww_SM_6TeV.root', 'r')
    inFile_INT_T1_12 = ROOT.TFile(path+'mumu_nunuww_INT_T1_12_6TeV.root', 'r')
    inFile_QUAD_T1_12 = ROOT.TFile(path+'mumu_nunuww_QUAD_T1_12_6TeV.root', 'r')

    mjj_bins = array.array('d',[0,200,400,600,800,1000,1200,1500,2000,3000,4000,6000])
    # mjj_bins = array.array('d',[0,500,1000,1500,2000,3000,4000,6000])

    var_name = "jj_M" #"missingE"  #"jj_pT" #"jj_M"
    x_title = "m_{WW} [GeV]" #, "Missing Energy" #"Dijet p_{T} [GeV]" #"Dijet Invariant Mass [GeV]"

    h_wzmunu = inFile_wzmunu.Get(var_name)
    h_zzmumu = inFile_zzmumu.Get(var_name)
    h_ggwpwm  = inFile_ggwpwm.Get(var_name)
    h_wpwmz_ztonunu = inFile_wpwmz_ztonunu.Get(var_name)
    h_mumu_nunuww_SM = inFile_mumu_nunuww_SM.Get(var_name)
    h_INT_T1 = inFile_INT_T1_12.Get(var_name)
    h_QUAD_T1 = inFile_QUAD_T1_12.Get(var_name)

    h_EFT = h_INT_T1.Clone()
    h_EFT.Add(h_QUAD_T1)


    print("h_wzmunu ", h_wzmunu.Integral())
    print("h_zzmumu ", h_zzmumu.Integral())
    print("h_ggwpwm ", h_ggwpwm.Integral())
    print("h_wpwmz_ztonunu ", h_wpwmz_ztonunu.Integral())
    print("h_mumu_nunuww_SM ", h_mumu_nunuww_SM.Integral())


    h_wzmunu_rebin = h_wzmunu.Rebin(len(mjj_bins)-1, 'h_wzmunu_rebin', mjj_bins)
    h_zzmumu_rebin = h_zzmumu.Rebin(len(mjj_bins)-1, 'h_zzmumu_rebin', mjj_bins)
    h_ggwpwm_rebin = h_ggwpwm.Rebin(len(mjj_bins)-1, 'h_ggwpwm_rebin', mjj_bins)
    h_wpwmz_ztonunu_rebin = h_wpwmz_ztonunu.Rebin(len(mjj_bins)-1, 'h_wpwmz_ztonunu_rebin', mjj_bins)
    h_mumu_nunuww_SM_rebin = h_mumu_nunuww_SM.Rebin(len(mjj_bins)-1, 'h_mumu_nunuww_SM_rebin', mjj_bins)
    h_EFT_rebin = h_EFT.Rebin(len(mjj_bins)-1, 'h_EFT_rebin', mjj_bins)

    setStyle(h_ggwpwm_rebin, kOrange+8, 8, x_title)
    setStyle(h_wzmunu_rebin, kOrange, 8, x_title)
    setStyle(h_mumu_nunuww_SM_rebin, kCyan+2, 8, x_title)
    setStyle(h_wpwmz_ztonunu_rebin, kAzure+2, 8, x_title)
    setStyle(h_zzmumu_rebin, kAzure+3, 8, x_title)

    h_EFT_rebin.SetFillColor(0)
    h_EFT_rebin.SetLineColor(kRed)
    h_EFT_rebin.SetLineStyle(2)
    h_EFT_rebin.SetLineWidth(2)
    h_EFT_rebin.GetXaxis().SetTitle(x_title)
    h_EFT_rebin.GetXaxis().SetRangeUser(0,6500)


    nunuWW_hbks_map = {"zzmumu": h_zzmumu_rebin,
                "wpwmz_ztonunu": h_wpwmz_ztonunu_rebin,
                "mumu_nunuww_SM": h_mumu_nunuww_SM_rebin,
                "wzmunu": h_wzmunu_rebin,
                "ggwpwm": h_ggwpwm_rebin,
                }

    nunuWW_label_map = {"zzmumu": "ZZ#mu#mu",  #"ZZ#mu^{+}#mu^{-}",
                "wpwmz_ztonunu": "WWZ (Z#rightarrow #nu#nu)",
                "mumu_nunuww_SM": "WW#nu#nu", #"W^{+}W^{-}#nu#nu",
                "wzmunu": "WZ#mu#nu", #"W^{#pm}Z#mu^{#mp}#nu",
                "ggwpwm": "WW#mu#mu", #"W^{+}W^{-}#mu^{+}#mu^{-}",
                }

    print(len(nunuWW_hbks_map))
    drawIt(nunuWW_hbks_map, h_EFT_rebin, nunuWW_label_map, 'mjj_nunu', "WW#nu#nu Channel")


elif channel == "mumuww":
    mumu_path = "/work/cwaits/dijet_hists/"

    var_name = "R_dijet_mass" #"R_dijet_mass_10GeVbinning"
    x_title = "m_{WW} [GeV]"

    inFile_mumuww_SM = ROOT.TFile(mumu_path+'SM6_100k_normalizedHists.root', 'r')
    inFile_INT_T1_12 = ROOT.TFile(mumu_path+'INT6_100k_1E-12_normalizedHists.root', 'r')
    inFile_QUAD_T1_12 = ROOT.TFile(mumu_path+'QUAD6_100k_1E-12_normalizedHists.root', 'r')

    mjj_bins = array.array('d',[0,200,400,600,800,1000,1200,1500,2000,3000,4000,6000])
    # mjj_bins = array.array('d',[0,500,1000,1500,2000,3000,4000,6000])

    h_mumuww_SM  = inFile_mumuww_SM.Get(var_name)
    h_INT_T1 = inFile_INT_T1_12.Get(var_name)
    h_QUAD_T1 = inFile_QUAD_T1_12.Get(var_name)

    print("h_mumuww_SM ", h_mumuww_SM.Integral())

    h_EFT = h_INT_T1.Clone()
    h_EFT.Add(h_QUAD_T1)

    h_mumuww_SM_rebin = h_mumuww_SM.Rebin(len(mjj_bins)-1, 'h_ggwpwm_rebin', mjj_bins)
    # h_EFT_rebin = h_EFT.Rebin(len(mjj_bins)-1, 'h_EFT_rebin', mjj_bins)
    h_EFT_rebin = h_EFT.Clone()

    setStyle(h_mumuww_SM, kOrange+8, 8, x_title)

    h_EFT_rebin.SetFillColor(0)
    h_EFT_rebin.SetLineColor(kRed)
    h_EFT_rebin.SetLineStyle(2)
    h_EFT_rebin.SetLineWidth(2)
    h_EFT_rebin.GetXaxis().SetTitle(x_title)
    h_EFT_rebin.GetXaxis().SetRangeUser(0,6500)

    mumuWW_hbks_map = {"mumuww_SM": h_mumuww_SM,
    
    }

    nunuWW_label_map = {
    "mumuww_SM": "WW#mu#mu",
    }
    print(len(mumuWW_hbks_map))

    drawIt(mumuWW_hbks_map, h_EFT_rebin, nunuWW_label_map, 'mjj_mumu', "WW#mu#mu Channel")

