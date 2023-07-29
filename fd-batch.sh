#!/bin/bash

if [ $# -eq 0 ]
  then # set the defaults
    PARTITION='gpu-debug'
elif [ $1 = 'gpu' ] || [ $1 = 'gpu-debug' ]
  then
    PARTITION=$1
else
  echo "Usage: $0 [partition]"
  exit 1
fi

if [ $PARTITION = 'gpu' ]
  then
    TIME="1-23:59:59"
elif [ $PARTITION = 'gpu-debug' ]
  then
    TIME="3:59:59"
fi


FD_VERSION='4'
PROGRAM_RUN_CMD="python3 main.py --epochs 5000 --fdversion $FD_VERSION"

LOGDIR=./tmplog
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

function program_commands() {
  dataset_name=$1
  ndim=$2
  
}

function slurm_flags() {
  dataset_name=$1
  ndim=$2
  if [ $PARTITION = 'gpu' ]
    then
      s="--time=1-23:59:00"
  elif [ $PARTITION = 'gpu-debug' ]
    then
    s="--time=3:59:00"
  fi
  s="$s -o $LOGDIR/fd-$ndim-$dataset_name-%j.txt -e $LOGDIR/fd-$ndim-$dataset_name-%j.err"
  
  echo $s
}

function run_sbatch_script() {
  local dataset_name=$1
  local ndim=$2
  local method=$3

  program_cmd="$PROGRAM_RUN_CMD -d $dataset_name --ndim $ndim"

  sbatch_flags="-p $PARTITION -A general --time=$TIME -J $ndim-$dataset_name --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --mem=200G"
  sbatch_flags="$sbatch_flags -o $LOGDIR/fd-$ndim-$dataset_name-%j.txt -e $LOGDIR/fd-$ndim-$dataset_name-%j.err"

  sbatch $sbatch_flags --wrap "module load deeplearning/2.12.0; $program_cmd"
}

function run_srun_script() {
  local dataset_name=$1
  local ndim=$2 
  local method=$3

  program_cmd=$(program_commands $dataset_name $ndim $method)
  slurm_flags=$(slurm_flags)
  
  srun $slurm_flags $program_cmd
}

# # for method in forcedirected node2vec deepwalk line sdne struc2vec; do
# for method in forcedirected; do
#   for ndim in 6 12 24 32 64 128; do
#     for dataset_name in cora citeseer pubmed ego-facebook corafull wiki ; do
#       run_sbatch_script "$dataset_name" "$ndim" "$method" 
#     done
#   done
# done

for method in forcedirected; do
  for ndim in 64 128; do
    for dataset_name in cora ego-facebook ; do
      run_sbatch_script "$dataset_name" "$ndim" "$method" 
    done
  done
done