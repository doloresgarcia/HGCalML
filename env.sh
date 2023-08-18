export HGCALML=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)
export DEEPJETCORE_SUBPACKAGE=$HGCALML
export PATH=$HGCALML/scripts:$PATH
export PYTHONPATH=$HGCALML/modules:$PYTHONPATH
export LD_LIBRARY_PATH=$HGCALML/modules:$LD_LIBRARY_PATH
export LC_ALL=C.UTF-8 	# necessary for wandb
export LANG=C.UTF-8	# necessary for wandb
#?export PYTHONPATH=$HGCALML/modules/datastructures:$PYTHONPATH
#for ffmpeg


export PATH=$PATH:/afs/cern.ch/work/g/gkrzmanc/pypack
export PYTHONPATH=$PYTHONPATH:/afs/cern.ch/work/g/gkrzmanc/pypack/
export LC_ALL=C.UTF-8
export LANG=C.UTF-8


export PATH=$PATH:/mnt/proj3/dd-23-91/cern/pypack
export PYTHONPATH=$PYTHONPATH:/mnt/proj3/dd-23-91/cern/pypack


