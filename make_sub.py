from pathlib import Path 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('root', nargs='+')
parser.add_argument('--energy', default='-1')
parser.add_argument('--hist_dir', default='/eos/user/a/aschuy/public/muon_collider/histograms/heavy_higgs')
parser.add_argument('--executable', default='/afs/cern.ch/user/a/aschuy/work/private/VBS_WGamma/muon_collider/WW-analysis/analyzeDelphes.sh')
parser.add_argument('--condor_dir', default='/afs/cern.ch/user/a/aschuy/work/public/muon_collider/mumu_multiboson/histograms/condor')
args = parser.parse_args()
out_dir = Path(args.hist_dir) / f'{args.energy}tev'
executable = str(Path(args.executable).resolve())
condor_dir = Path(args.condor_dir).resolve()
energy = args.energy
def make_if_needed(d):
    if not d.exists():
        d.mkdir(parents=True)
make_if_needed(condor_dir)
make_if_needed(condor_dir / 'output')
make_if_needed(condor_dir / 'condor')
make_if_needed(condor_dir / 'error')
make_if_needed(condor_dir / 'submit')

for p in args.root:
    p = Path(p).resolve()
    process = p.stem
    out_file = str(out_dir / process)
    p = str(p)
    with Path('job.tmp').open('r') as f:
        template_text = f.read()
        template_text = template_text.format(executable=executable, in_file=p, out_file=out_file, energy=energy, condor_dir=str(condor_dir))
    with (condor_dir / 'submit' / Path(f'{Path(p).stem}.sub')).open('w') as f:
        f.write(template_text)
    