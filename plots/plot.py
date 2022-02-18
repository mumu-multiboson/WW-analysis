from ROOT import *
from pathlib import Path
from typing import List
gStyle.SetOptStat(1111)

def plot(paths: List[Path], out_name: str, titles: List[str] = None):
    tfiles = [TFile(str(p)) for p in paths]
    if titles is None:
        titles = [f.GetName() for f in tfiles]
    keys = [k.GetName() for k in tfiles[0].GetListOfKeys()]
    for i, key in enumerate(keys):
        c = TCanvas("c", "hist")
        leg = TLegend(0.68,0.68,0.98,0.92)
        title = f"Title:{key}"
        for j, f in enumerate(tfiles):
            h = f.Get(key)
            h.Draw("sames,E")
            gPad.Update()
            h.SetLineColor(j+2)
            leg.AddEntry(h, titles[j])
        leg.Draw()
        if i == 0:
            c.Print(f"{out_name}(",title)
        elif i < (len(keys) - 1):
            c.Print(out_name,title)
        else:
            c.Print(f"{out_name})",title)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='*', help='List of root files containing histograms.')
    parser.add_argument('--output', '-o', help='output path', default='plots.pdf')
    args = parser.parse_args()

    args.input = [Path(p) for p in args.input]

    plot(args.input, args.output)

if __name__ == '__main__':
    main()