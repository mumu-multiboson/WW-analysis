import ROOT
from ROOT import *
import array
import os
import argparse
from pathlib import Path


def rebin(root_files: Path, out_path: Path, var_name: str):
    out_path.mkdir(parents=True, exist_ok=True)
    # variable of interest
    var_name = "jj_M"

    # define the binning
    mjj_bins = array.array('d',[0,200,400,600,800,1000,1200,1500,2000,3000,4000,6000])

    print("====== Event Yields ========")
    for root_file in root_files:
        inFile = ROOT.TFile(str(root_file), 'r')
        h_tmp = inFile.Get(var_name)

        # print total number of events
        print(f"{root_file} : {h_tmp.Integral()}")

        # rebin the histogram
        h_tmp_rebin = h_tmp.Rebin(len(mjj_bins)-1, 'h_tmp_rebin', mjj_bins)

        outFile = TFile(str(out_path / root_file.name),'RECREATE')
        outFile.cd()
        h_tmp_rebin.Write(var_name)
        outFile.Close()
        del h_tmp

print("==================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+', help='List of root files.')
    parser.add_argument('--output', '-o', help='Output directory', default='reco_histograms/rebinned')
    parser.add_argument('--var_name', '-v', default="jj_M")
    args = parser.parse_args()

    root_files = [Path(s) for s in args.input]
    out_path = Path(args.output)
    rebin(root_files, out_path, args.var_name)
