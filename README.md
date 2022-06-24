# analysis
This repository contains the analysis scripts for event selection, plotting e.t.c


## histogram making
Suppose you want to write to an output directory `out` using all of the cuts from `reco_selection.py`, for some root ntuple that is located at `../ntuples/input.root`, but only the first 10k events. Then, you can run:

`python3 analyzeDelphes_RecoHistograms.py -o out --all -n 10000 ../ntuples/input.root`

The output will then be written to `out/input.root` with an accompanying log file `hist.log` that lists the selections that were used, along with their efficiencies.
If you want to run on multiple input files, you can specify them separately:
`python3 analyzeDelphes_RecoHistograms.py -o out --all -n 10000 ../ntuples/input.root ../ntuples/input2.root`
Or you can specify `../ntuples/*.root`.
Or, if you have madgraph output directories that have a single run (`Events/run_01/unweighted_events.root`), you can specify the madgraph directory instead of `unweighted_events.root`, so that the correct name of the process is used for the output root file.
If you want to force the script to overwrite an existing output file, you can add -f
If you want to specify a subset of the cuts, you can specify them with -c, e.g., `-c 0,1,3` will use the 0, 1, and 3 cuts, given the order in reco_selection.py
The output histograms will be automatically scaled according to the luminosity in lumi.yaml and cross_section.yaml.
lumi.yaml maps from energy in TeV (3, 6, 10, 30) to integrated luminosity. You can specify the energy with -e (default, 6)
cross_section.yaml maps from the input process name to a cross section. The input process name is inferred from the stem of the input root file, or from the name of the directory, if a madgraph directory is specified.
Cross sections should be in fb, luminosities should be in fb^-1
If you specify multiple files, they will be run in parallel. Progress will be shown with a rich progress bar, if you have rich installed in your python environment.


### Cut values
You can specify cut values to `analyzeDelphes_RecoHistograms.py` using the variable names specified in `reco_selection.py`. For example, add the following to set the pt min to 200 GeV:
```
--pt_min=200
```

Note the `=`!

## Condor submission
To submit a histogramming job to condor, you can use `condor_submit.py` (assuming you have your environment set up for condor submission, and that you have set up your environment such that the usual histogram scripts are runnable). 

Example:
```
python condor_submit.py --python_args "indir -f -o outdir -n 1000 -c 1,4 --ncpus 1"
```

To submit many jobs, you can write, e.g., a simple bash script:

for PT_MIN in 100 200 300 400
do
    python condor_submit.py --python_args "indir -f -o outdir -n 1000 -c 1,4 --ncpus 1 --pt_min=$PT_MIN"
done