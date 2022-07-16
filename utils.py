from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from functools import reduce
from operator import mul
from multiprocessing import Pool, Pipe, Manager
from multiprocessing.connection import Connection
from contextlib import ExitStack


import numpy as np
import yaml
from ROOT import *

gSystem.Load("libDelphes")
gStyle.SetOptStat(0)
try:
    from rich.progress import Progress
    has_rich=True
except ImportError:
    print('To include progress bar, run `pip install rich`')
    has_rich=False

import logging


class Cut:
    def __init__(self, description: str, func):
        self.description = description
        self.func = func

    def apply(self, input_dict: Dict[str, Any]):
        return self.func(input_dict)
    
    def __str__(self):
        return self.description
    
class EventSelection:
    def __init__(self, cuts: List[Cut], active_indices: Union[None, List[int]], pipe: Connection):
        self.cuts = cuts
        if active_indices == 'all':
            active_indices = [n for n in range(len(cuts))]
        elif active_indices == 'none':
            active_indices = []
        self.active_indices = active_indices
        self.n_passed = np.zeros(len(cuts))
        self.n_failed = np.zeros(len(cuts))
        self.pipe = pipe

        msg = 'EVENT SELECTION:\n'
        for i, c in enumerate(self.cuts):
            if i in self.active_indices:
                status = 'ON'
            else:
                status = 'OFF'
            msg = msg + f'\t{i}: {c} -- {status}\n'
        pipe.send(msg)
    
    def apply(self, input_dict: Dict[str, Any]):
        selected = True
        for i, c in enumerate(self.cuts):
            if i in self.active_indices:
                if c.apply(input_dict):
                    self.n_passed[i] += 1
                else:
                    selected = False
                    self.n_failed[i] += 1
                    break
        return selected

    def efficiency(self, relative=True):
        n_total = self.n_passed + self.n_failed
        active_mask = n_total > 0
        efficiency = np.full(len(self.cuts), fill_value=-1.0)
        if relative:
            efficiency[active_mask] = self.n_passed[active_mask] / n_total[active_mask]
        else:
            # absolute efficiency
            efficiency[active_mask] = self.n_passed[active_mask] / n_total[self.active_indices[0]]
        return efficiency
    
    def efficiency_msg(self):
        efficiency = self.efficiency()
        absolute_efficiency = self.efficiency(relative=False)
        msg = 'EVENT SELECTION EFFICIENCY:\n'
        for i, c in enumerate(self.cuts):
            if i in self.active_indices:
                eff_s = 'NA'
                if efficiency[i] != -1.0:
                    eff_s = f'{efficiency[i]:.2%}'
                abs_eff_s = 'NA'
                if absolute_efficiency[i] != -1.0:
                    abs_eff_s = f'{absolute_efficiency[i]:.2%}'
                msg = msg + f'\t{i} -- {c}: {eff_s} ({self.n_passed[i]} / {self.n_passed[i] + self.n_failed[i]}, relative) {abs_eff_s} ({self.n_passed[i]} / {self.n_passed[self.active_indices[0]] + self.n_failed[self.active_indices[0]]}, absolute) \n'
        total_eff = reduce(mul, efficiency[efficiency != -1.0], 1.0)
        msg = msg + f'\ttotal: {total_eff:.2%}\n\n'
        return msg

    def efficiency_csv(self, relative=True, expected_yield=None):
        efficiency = self.efficiency(relative)
        if expected_yield is not None:
            efficiency = [f'{e*expected_yield:.2f}' if e != -1.0 else 'NA' for e in efficiency]
        else:
            efficiency = [f'{e:.2%}' if e != -1.0 else 'NA' for e in efficiency]
        return ','.join(efficiency)
        

def scale(f: TFile, luminosity: float, cross_section: float, n_entries: int):
    keys = [k.GetName() for k in f.GetListOfKeys()]
    for key in keys:
        h = f.Get(key)
        if h.GetEntries() > 0:
            h.Scale(luminosity * cross_section / n_entries)
            h.Write(key, TObject.kOverwrite)


def parse_args(func, default_out):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+', help='List of madgraph output directories or root files.')
    parser.add_argument('--output', '-o', help='Output directory', default=default_out)
    parser.add_argument('--force_overwrite', '-f', action='store_true')
    parser.add_argument('--cuts', '-c', type=lambda s: [int(item) for item in s.split(',')], help='"," delimited list of cut indices to use (starting from 0).', default=None)
    parser.add_argument('--n_events', '-n', default=-1, type=int)
    parser.add_argument('--energy', '-e', default=6, help='cm energy in TeV', type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--all', action='store_true', help='use all cuts')
    parser.add_argument('--ncpus', type=int, default=10, help='number of cpus to use')

    args, unknown = parser.parse_known_args()
    cut_values = {arg.split('=')[0].lstrip('-'): arg.split('=')[1] for arg in unknown}

    global has_rich
    if args.debug:
        has_rich = False

    if args.cuts is None:
        if args.all:
            args.cuts = 'all'
        else:
            args.cuts = 'none'
        s = args.cuts
    else:
        s = 's' + ''.join(str(i) for i in args.cuts)
    output_dir = Path(args.output) / s
    output_dir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=output_dir/'hist.log',
                    filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    csv_eff_output = output_dir / 'relative_efficiency.csv'
    csv_eff_output.unlink(missing_ok=True)
    csv_abs_eff_output = output_dir / 'absolute_efficiency.csv'
    csv_abs_eff_output.unlink(missing_ok=True)
    csv_yield_output = output_dir / 'yield.csv'
    csv_yield_output.unlink(missing_ok=True)

    cross_section_path = Path(__file__).parent.absolute() / 'cross_section.yaml'
    luminosity_path = Path(__file__).parent.absolute() / 'lumi.yaml'
    energy = args.energy
    
    with cross_section_path.open('r') as f:
        cross_sections = yaml.load(f, Loader=yaml.SafeLoader)
    with luminosity_path.open('r') as f:
        luminosities = yaml.load(f, Loader=yaml.SafeLoader)
    try:
        luminosity = luminosities[energy]
    except:
        logging.error(f'"{energy}" [TeV] not found in {luminosity_path}!')
        raise

    output_paths = [str(output_dir / f'{Path(i).stem}.root') for i in args.input]

    with ExitStack() as stack:
        if has_rich:
            progress = stack.enter_context(Progress(transient=True))

        m = Manager()
        lock = m.Lock()
        with Pool(args.ncpus) as pool:
            proc_args = []
            pipes = []
            tasks = []
            std_pipes = []
            for input, output in zip(args.input, output_paths):
                if not args.force_overwrite:
                    if Path(output).exists():
                        logging.info(f'{output} already exists, skipping...')
                        continue
                if Path(input).is_dir():
                    input = str(Path(input) / 'Events' / 'run_01' / 'unweighted_events.root')
                process = Path(output).stem
                try:
                    cross_section = cross_sections[process]
                except:
                    logging.error(f'"{process}" not found in {cross_section_path}!')
                    raise
                if has_rich:
                    tasks.append(progress.add_task(f"Processing {Path(output).stem}..."))
                p_output, p_input = Pipe()
                pipes.append(p_output)
                std_pipe_out, std_pipe_in = Pipe()
                std_pipes.append(std_pipe_out)
                proc_args.append((input, output, args.cuts, args.n_events, args.energy, luminosity, cross_section, p_input, std_pipe_in, cut_values, csv_eff_output, csv_abs_eff_output, csv_yield_output, lock))
            if args.debug:
                for p_args in proc_args:
                    func(*p_args, debug=True)
            else:
                result = pool.starmap_async(func, proc_args)
                while not result.ready():
                    # Update progressbar.
                    if has_rich:
                        for i, (pipe, task) in enumerate(zip(pipes, tasks)):
                            if pipe.poll():
                                total, advance = pipe.recv()
                                progress.update(task, total=total, advance=advance)
                    # Print info messages from processes.
                    for std_pipe, input in zip(std_pipes, args.input):
                        if std_pipe.poll():
                            msg = std_pipe.recv()
                            logging.info(Path(input).stem)
                            logging.info(msg)