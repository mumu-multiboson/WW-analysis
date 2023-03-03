out=$1
in=$2
energy=$3

originalDir=$PWD
cd /afs/cern.ch/user/a/aschuy/work/private/VBS_WGamma/muon_collider/event-generation
source setup.sh
cd $originalDir

filename=$(basename -- "$out")
echo "filename="$filename

python /afs/cern.ch/user/a/aschuy/work/private/VBS_WGamma/muon_collider/WW-analysis/analyzeDelphes_RecoHistograms.py --all -f -o $out -e $energy $in
mv $out/all/$filename.root $out/../$filename.root