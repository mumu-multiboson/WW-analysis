maxEvents=9E9
DEBUG=True
from pathlib import Path
try:
    from rich.progress import track as progress
    has_rich=True
except ImportError:
    print('To include progress bar, run `pip install rich`')
    has_rich=False
import yaml

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


def printEvent(input, out, n):
    f=TFile(input)
    tree=f.Get("Delphes")
    tree.GetEntry(n)
    with out.open('w+') as o:
        o.write('status pid')
        for p in tree.Particle:
            o.write(f'{p.Status} {p.PID}\n')


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='madgraph output directory or root file.')
    parser.add_argument('-n', default=0, type=int)
    parser.add_argument('output', default='event.txt')
    args = parser.parse_args()

    i = args.input
    if Path(i).is_dir():
        i = str(Path(i) / 'Events' / 'run_01' / 'unweighted_events.root')
    printEvent(i, Path(args.output), args.n)

    
