from pathlib import Path 
import argparse
import subprocess
import pexpect
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('root', nargs='+')
parser.add_argument('--energy', default='-1')
parser.add_argument('--hist_dir', default='/eos/user/a/aschuy/public/muon_collider/histograms/heavy_higgs')
parser.add_argument('--executable', default='/afs/cern.ch/user/a/aschuy/work/private/VBS_WGamma/muon_collider/WW-analysis/analyzeDelphes.sh')
args = parser.parse_args()
out_dir = Path(args.hist_dir) / f'{args.energy}tev'
executable = str(Path(args.executable).resolve())
energy = args.energy
password = "888071VAnHx6"

for p in args.root:
    p = Path(p).resolve()
    process = p.stem
    out_file = str(out_dir / process)
    p = str(p)


    cmd = f'ssh aschuy@lxplus.cern.ch "{executable} {out_file} {p} {energy}"'
    print(cmd)
    