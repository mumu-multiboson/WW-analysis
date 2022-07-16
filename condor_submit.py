import argparse
from pathlib import Path
import subprocess
import uuid
import os
import stat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_log', default='condor/output', help='dir which will contain standard output logs for the job.')
    parser.add_argument('--error_log', default='condor/error', help='dir which will contain standard error logs for the job.')
    parser.add_argument('--condor_log', default='condor/condor', help='dir which will contain condor logs for the job.')
    parser.add_argument('--python_script', default='analyzeDelphes_RecoHistograms.py')
    parser.add_argument('--python_args', '-pargs', help='args for python script')

    args = parser.parse_args()

    Path(args.output_log).mkdir(exist_ok=True, parents=True)
    Path(args.error_log).mkdir(exist_ok=True, parents=True)
    Path(args.condor_log).mkdir(exist_ok=True, parents=True)
    condor_dir = Path('condor')
    condor_dir.mkdir(exist_ok=True, parents=True)

    setup_text = """#!/bin/bash\nshopt -s expand_aliases\nalias setupATLAS='source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh'\nsetupATLAS\nlsetup "views LCG_101 x86_64-centos7-gcc8-opt" """
    python_script = Path(args.python_script).absolute()
    python_text = f"python {python_script} {args.python_args}"
    run_text = f"""{setup_text}\n{python_text}"""
    unique_id = uuid.uuid4().hex
    run_file = (condor_dir / f'{unique_id}.sh').absolute()
    with run_file.open('w') as fd:
        fd.write(run_text)
    st = os.stat(run_file)
    os.chmod(run_file, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    arguments = '$(ClusterId) $(ProcId)'
    output = f'{args.output_log}/$(ClusterId).$(ProcId).out'
    error = f'{args.error_log}/$(ClusterId).$(ProcId).err'
    log = f'{args.condor_log}/$(ClusterId).log'
    submit_text = f"""executable = {run_file}\ngetenv = True\narguments = {arguments}\noutput = {output}\nerror = {error}\nlog = {log}\nqueue"""

    submit_file = condor_dir / f'{unique_id}.sub'
    with submit_file.open('w') as fd:
        fd.write(submit_text)
    
    subprocess.run(['condor_submit', str(submit_file.absolute())])



if __name__ == '__main__':
    main()
