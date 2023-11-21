
export PATH=$PATH:/afs/cern.ch/work/g/gkrzmanc/pypack
export PYTHONPATH=$PYTHONPATH:/afs/cern.ch/work/g/gkrzmanc/pypack/
export LC_ALL=C.UTF-8
export LANG=C.UTF-8


export HGCALML=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)
export DEEPJETCORE_SUBPACKAGE=$HGCALML

export PATH=$HGCALML/scripts:$PATH
export PYTHONPATH=$HGCALML/modules:$PYTHONPATH
export LD_LIBRARY_PATH=$HGCALML/modules:$LD_LIBRARY_PATH

export PATH=$PATH:/mnt/proj3/dd-23-91/cern/pypack
export PYTHONPATH=$PYTHONPATH:/mnt/proj3/dd-23-91/cern/pypack

# wandb api key
# if ~/private/wandb_api.sh file exists then source from it
if [[ -f ~/private/wandb_api.sh ]]; then
   source ~/private/wandb_api.sh
fi


export PATH=$PATH:/afs/cern.ch/work/m/mgarciam/private/python_pkgs_hgcal
export PYTHONPATH=$PYTHONPATH:/afs/cern.ch/work/m/mgarciam/private/python_pkgs_hgcal/
#export PATH=$PATH:/afs/cern.ch/work/g/gkrzmanc/pypack
#export PYTHONPATH=$PYTHONPATH:/afs/cern.ch/work/g/gkrzmanc/pypack/
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

export PATH=$PATH:/afs/cern.ch/work/m/mgarciam/private/python_pkgs_hgcal
export PYTHONPATH=$PYTHONPATH:/afs/cern.ch/work/m/mgarciam/private/python_pkgs_hgcal/
#export PATH=$PATH:/afs/cern.ch/work/g/gkrzmanc/pypack
#export PYTHONPATH=$PYTHONPATH:/afs/cern.ch/work/g/gkrzmanc/pypack/
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
