from pathlib import Path
import yaml
import subprocess
import argparse
from ROOT import *

def scale(input_path: Path, output_path: Path, cross_section: float, luminosity: float):
    """cross_section: pb
       luminosity: pb^-1"""
    input_file = TFile(str(input_path))
    output_file = TFile(str(output_path), "RECREATE")
    output_file.cd()
    keys = [k.GetName() for k in input_file.GetListOfKeys()]
    for key in keys:
        h = input_file.Get(key)
        h2 = h.Clone()
        if h.GetEntries() > 0:
            h2.Scale(luminosity * cross_section / h.GetEntries())
            h2.Write()
    output_file.Close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', metavar="DIR", help='input directory', default='histograms')
    parser.add_argument('--output', '-o', metavar="DIR", help='output directory', default='scaled_histograms')
    parser.add_argument('--energy', '-e', default=6, help='cm energy in TeV')
    args = parser.parse_args()
    root_input_path = Path(args.input)
    root_output_path = Path(args.output)
    xsec_path = Path.cwd() / 'cross_section.yaml'
    lumi_path = Path('lumi.yaml')
    energy = args.energy
    
    with xsec_path.open('r') as f:
        xsecs = yaml.load(f, Loader=yaml.SafeLoader)
    with lumi_path.open('r') as f:
        lumis = yaml.load(f, Loader=yaml.SafeLoader)
    lumi = lumis[energy]

    for input_path in root_input_path.rglob('*.root'):
        try:
            relative_path = input_path.relative_to(root_input_path)
            process = input_path.stem
            xsec = xsecs[process]
            output_path = root_output_path / relative_path
            output_path.parent.mkdir(exist_ok=True, parents=True)
            scale(input_path, output_path, cross_section=xsec, luminosity=lumi)
        except BaseException as e:
            print(e)

if __name__ == '__main__':
    main()