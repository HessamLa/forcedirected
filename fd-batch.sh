#!/bin/bash
LOGDIR=./tmplog
rm -rf $LOGDIR
mkdir -p $LOGDIR

vgnn

# echo '' > commands.txt
# for METHOD in node2vec deepwalk line sdne struc2vec
# do
# for DATASET_NAME in cora citeseer pubmed ego-facebook corafull wiki
# do
# cmd="python3 make-embedding.py -m $METHOD -d $DATASET_NAME ";
# echo $cmd >> commands.txt
# done
# done

# for METHOD in node2vec deepwalk line struc2vec sdne
# for METHOD in  node2vec deepwalk line struc2vec
# for DATASET_NAME in cora citeseer pubmed corafull wiki ego-facebook
for DATASET_NAME in pubmed corafull wiki
# for DATASET_NAME in ego-facebook
do
echo $DATASET_NAME
cmd='';
cmd="python3 model.py -d $DATASET_NAME --epochs 4000 ";
echo $cmd

# make the sbatch script
sbatch <<EOT
#!/bin/bash
#SBATCH -J fd-$DATASET_NAME
#SBATCH -p gpu-debug
#SBATCH -A general 
#SBATCH -o $LOGDIR/fd-$DATASET_NAME.txt
#SBATCH -e $LOGDIR/fd-$DATASET_NAME.err
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --time=4:00:00
#SBATCH --mem=200G

source ~/../BigRed200/gnn/.venv/bin/activate

load deeplearning/2.12.0

srun $cmd

EOT
done

# DATASET_NAME=cora; python3 model.py -d $DATASET_NAME --epochs 4000 > ./tmplog/fd-$DATASET_NAME.txt 2> ./tmplog/fd-$DATASET_NAME.err


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