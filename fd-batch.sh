#!/bin/bash

FD_VERSION='4'
FD_VERSION='4nodrop'
FD_VERSION='5'
FD_VERSION='5z2'
# FD_VERSION='6'
# FD_VERSION='7'
FD_VERSION='104'
FD_VERSION='106'
# FD_VERSION='107'
FD_VERSION='108'
FD_VERSION='109'
FD_VERSION='110'
FD_VERSION='110k100'

# PROGRAM_RUN_CMD="python3 main.py --epochs 5000 --fdversion $FD_VERSION"
PROGRAM_RUN_CMD="python3 -m forcedirected --epochs 5000 --fdversion $FD_VERSION"

if [ $# -eq 0 ]
  then # set the defaults
    PARTITION='gpu-debug'
elif [ $1 = 'gpu' ] || [ $1 = 'gpu-debug' ] || [ $1 = 'general' ]
  then
    PARTITION=$1
else
  echo "Usage: $0 [partition]"
  exit 1
fi

if [ $PARTITION = 'gpu' ] || [ $PARTITION = 'general' ]
  then
    TIME="1-23:59:59"
elif [ $PARTITION = 'gpu-debug' ] || [ $PARTITION = 'debug' ]
  then
    TIME="3:59:59"
fi

# if 'Carbonate' is in $HOME, then set MODULE_CMD='module load deeplearning/2.12.0; module unload cudatoolkit/11.7;'
# else if 'BigReg' is in $HOME, then set MODULE_CMD='module load deeplearning/2.12.0;'
if [[ $HOME == *"Carbonate"* ]]; then
    LOGIN_NODE='carbonate'
    MODULE_CMD='module load python/gpu/3.10.10; module unload cudatoolkit/11.7 '
elif [[ $HOME == *"BigRed200"* ]]; then
    LOGIN_NODE='bigred200'
    MODULE_CMD='module load python/gpu/3.10.10 '
else
    MODULE_CMD='echo No module command provided '
fi

# make the srun systems flag




LOGDIR=./tmplog
# rm -rf "$LOGDIR"
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
  # program_cmd="$program_cmd --outputdir_root ./embeddings-$PARTITION-$LOGIN_NODE"
  program_cmd="$program_cmd --outputdir_root ./embeddings-tmp"
  echo $program_cmd

  if [ $PARTITION = 'gpu' ] || [ $PARTITION = 'gpu-debug' ]; then
    sbatch_flags="-p $PARTITION -A general --time=$TIME -J $ndim-$dataset_name --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --mem=200G"
  elif [ $PARTITION = 'general' ] || [ $PARTITION = 'debug' ]; then
    sbatch_flags="-p $PARTITION -A general --time=$TIME -J $ndim-$dataset_name --nodes=1 --ntasks-per-node=1 --mem=200G"
  fi
  sbatch_flags="$sbatch_flags -o $LOGDIR/fd-%j-$ndim-$dataset_name.txt -e $LOGDIR/fd-%j-$ndim-$dataset_name.err"
  sbatch $sbatch_flags --wrap "$MODULE_CMD; $program_cmd"
}

function run_srun_script() {
  local dataset_name=$1
  local ndim=$2 
  local method=$3

  program_cmd=$(program_commands $dataset_name $ndim $method)
  slurm_flags=$(slurm_flags)
  
  srun $slurm_flags $program_cmd
}

echo $(date '+%y-%m-%d,%H:%M:%S') >> slurm-jobids.txt
echo Partition:  $PARTITION >> slurm-jobids.txt
echo Login Node: $LOGIN_NODE >> slurm-jobids.txt

# run_sbatch_script corafull 24 forcedirected

# for method in forcedirected node2vec deepwalk line sdne struc2vec; do
for ndim in 24 32 64 128 12; do
# for ndim in 24; do
  for method in forcedirected; do
    for dataset_name in cora citeseer ego-facebook wiki pubmed corafull ; do
      run_sbatch_script "$dataset_name" "$ndim" "$method" >> slurm-jobids.txt 2>&1
    done
  done
done


# python3 main.py --epochs 5000 --fdversion 6 -d pubmed --ndim 128 --outputdir_root ./embeddings-gpu-debug-bigred200
# Submitted batch job 1870617
# python3 main.py --epochs 5000 --fdversion 6 -d corafull --ndim 128 --outputdir_root ./embeddings-gpu-debug-bigred200
# Submitted batch job 1870618

# # for method in forcedirected node2vec deepwalk line sdne struc2vec; do
# for ndim in 128; do
# # for ndim in 64 128; do
#   for method in forcedirected; do
#     for dataset_name in pubmed corafull ; do
#       run_sbatch_script "$dataset_name" "$ndim" "$method" >> slurm-jobids.txt 2>&1
#     done
#   done
# done

echo "" >> slurm-jobids.txt
# for method in forcedirected; do
#   for ndim in 64 128; do
#     for dataset_name in cora ego-facebook ; do
#       run_sbatch_script "$dataset_name" "$ndim" "$method" 
#     done
#   done
# done