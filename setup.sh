#!/bin/sh

function version_gt() { test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"; }

# Setup ROOT and gcc
# added back by Michele
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh --quiet
if [ "${ROOTSYS}" = "" ]; then
    lsetup "views LCG_101_ATLAS_7 x86_64-centos7-gcc11-opt" --quiet
else
    root_version=`root-config --version`
    if version_gt 6.20 $root_version; then
	echo "ERROR root already loaded, but root version too old: $root_version"
    fi
fi

if [ "${ROOTSYS}" = "" ]; then
   echo -e "\033[41;1;37m Error initializing ROOT. ROOT is not set up. Please check. \033[0m"
else
   echo -e "\033[42;1;37m ROOT has been set to: *${ROOTSYS}* \033[0m"
fi

