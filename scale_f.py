from pathlib import Path
import yaml
import subprocess
import argparse
from ROOT import *

def scale(input_path: str, output_path: str, r: float):
    """cross_section: pb
       luminosity: pb^-1"""
    input_file = TFile(input_path)
    output_file = TFile(output_path, "RECREATE")
    output_file.cd()
    keys = [k.GetName() for k in input_file.GetListOfKeys()]
    for key in keys:
        h = input_file.Get(key)
        h2 = h.Clone()
        h2.Scale(r)
        h2.Write()
    output_file.Close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input file')
    parser.add_argument('output', help='output file')
    parser.add_argument('ratio', type=float, help='ratio of new / old couplings.')
    args = parser.parse_args()
    scale(args.input, args.output, r=args.ratio)

if __name__ == '__main__':
    main()