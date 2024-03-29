from typing import List
from ROOT import *
from pathlib import Path
import yaml
# gROOT.SetStyle('ATLAS')
gStyle.SetOptStat(1111)

palette = [40, 41, 42, 30, 46, 38, 28, 36]

def normalize_h(h, normalize):
    if normalize:
        h.Scale(1./h.Integral())

def plot(paths: List[Path], out_name: str, titles: List[str] = None, normalize: bool = False):
    tfiles = [TFile(str(p)) for p in paths]
    if titles is None:
        titles = [f.GetName() for f in tfiles]
    keys = [k.GetName() for k in tfiles[0].GetListOfKeys()]
    print(len(keys))
    for i, key in enumerate(keys):
        c = TCanvas("c", "hist")
        if len(tfiles) > 1:
            leg = TLegend(0.68,0.68,0.98,0.92)
        title = f"Title:{key}"
        if len(tfiles) == 2:
            # Draw ratio plot
            h1 = tfiles[0].Get(key)
            if isinstance(h1, TH2):
                continue
            normalize_h(h1, normalize)
            h2 = tfiles[1].Get(key)
            normalize_h(h2, normalize)
            h1.SetLineColor(2)
            h2.SetLineColor(3)
            rp = TRatioPlot(h1, h2)
            leg.AddEntry(h1, titles[0])
            leg.AddEntry(h2, titles[1])
            rp.Draw()
            rp.GetLowerRefGraph().SetMinimum(-3)
            rp.GetLowerRefGraph().SetMaximum(3)
        else:
            for j, f in enumerate(tfiles):
                h = f.Get(key)
                normalize_h(h, normalize)
                try:
                    h.Draw("sames,E")
                    if len(tfiles) == 1:
                        gPad.Update()
                    h.SetLineColor(palette[j%len(palette)])
                    if len(tfiles) > 1:
                        leg.AddEntry(h, titles[j])
                except:
                    pass
        print(i)
        if len(tfiles) > 1:
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
    parser.add_argument('config', help='Config file.')
    args = parser.parse_args()

    with Path(args.config).open('r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    out_name = config['output']
    paths = []
    titles = []
    for p in config['inputs']:
        paths.append(p)
        titles.append(config['inputs'][p]['title'])
    if 'prefix' in config:
        prefix = Path(config['prefix'])
        paths = [prefix / p for p in paths]
    Path(out_name).parent.mkdir(exist_ok=True, parents=True)
    plot(paths=paths, out_name=out_name, titles=titles, normalize=config['normalize'])

if __name__ == '__main__':
    main()