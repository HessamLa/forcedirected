#!/bin/bash

LOGDIR=./tmplog
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"
FD_VERSION='4'

function run_sbatch_script {
    local dataset_name="$1"
    local ndim="$2"
    local cmd="python3 main.py -d $dataset_name --epochs 5000 --ndim $ndim --fdversion $FD_VERSION"
    echo "$cmd"

    # Make the sbatch script
    sbatch <<EOT
#!/bin/bash
#SBATCH -J fd$ndim-$dataset_name
#SBATCH -p gpu
#SBATCH --time=1-23:00:00
#SBATCH -A general 
#SBATCH -o $LOGDIR/fd-$ndim-$dataset_name-%j.txt
#SBATCH -e $LOGDIR/fd-$ndim-$dataset_name-%j.err
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G

module load deeplearning/2.12.0

srun $cmd

EOT
}

# Uncomment this block if you need the command list for vgnn
# echo '' > commands.txt
# for METHOD in node2vec deepwalk line sdne struc2vec
# do
#     for DATASET_NAME in cora citeseer pubmed ego-facebook corafull wiki
#     do
#         cmd="python3 make-embedding.py -m $METHOD -d $DATASET_NAME"
#         echo "$cmd" >> commands.txt
#     done
# done

# for NDIM in 64 128
for NDIM in 6 12 24 32 64 128
do
    # for METHOD in node2vec deepwalk line struc2vec sdne
    # for METHOD in node2vec deepwalk line struc2vec
    for DATASET_NAME in cora citeseer wiki ego-facebook pubmed corafull blogcatalog
    # for DATASET_NAME in corafull pubmed
    do
        # for DATASET_NAME in pubmed corafull wiki
        # for DATASET_NAME in ego-facebook

        # echo "$DATASET_NAME"
        # cmd=''
        # cmd="python3 main.py -d $DATASET_NAME --epochs 5000 --ndim $NDIM --fdversion 4"
        # echo "$cmd"

        # Call the function to make batch script
        run_sbatch_script "$DATASET_NAME" "$NDIM"
    done
done


# DATASET_NAME=cora; python3 model.py -d $DATASET_NAME --epochs 5000 > ./tmplog/fd-$DATASET_NAME.txt 2> ./tmplog/fd-$DATASET_NAME.err


# sbatch <<EOT
# #!/bin/bash
# #SBATCH -J test
# #SBATCH -p gpu-debug
# #SBATCH -A general 
# #SBATCH -o test.txt
# #SBATCH -e test.err
# #SBATCH --nodes=1 
# #SBATCH --ntasks-per-node=1
# #SBATCH --gpus-per-node=1
# #SBATCH --time=1:00:00
# #SBATCH --mem=10G

# # source ~/../BigRed200/gnn/.venv/bin/activate

# module load deeplearning/2.12.0
# module unload cudatoolkit/11.7

# module list

# srun python3 -c \
# 'import torch; 
# print(torch.cuda.is_available()); 
# print(torch.cuda.get_device_name()); 
# print(torch.cuda.device_count()); 
# print(torch.cuda.current_device())
# print(torch.__version__);
# import pandas as pd;
# '

# EOT

# sbatch <<EOT
# #!/bin/bash
# #SBATCH -J linkpred
# #SBATCH -p gpu-debug
# #SBATCH -A general 
# #SBATCH -o linkpred-%j.txt
# #SBATCH -e linkpred-%j.err
# #SBATCH --nodes=1 
# #SBATCH --ntasks-per-node=1
# #SBATCH --gpus-per-node=1
# #SBATCH --time=4:00:00
# #SBATCH --mem=200G

# srun python linkprediction.py
# EOT