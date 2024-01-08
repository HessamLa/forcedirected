import os, sys
import subprocess
import time
from datetime import datetime
import argparse 
from pprint import pprint

# Set default values
fd_version = '121'  # The last assigned value in the original script
fd_version = '201'  # The last assigned value in the original script

def parse_args():
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('--submitjobs', action='store_true', default=False,
                        help='This is required to submit jobs')
    parser.add_argument('-v', '--fdversion', type=str, required=True,
                        help='ForceDirected version')
    parser.add_argument('-p', '--partition', type=str, required=True, choices=['gpu', 'gpu-debug', 'general', 'debug'],
                        help='Partition to submit jobs to')
    args, unknown = parser.parse_known_args()

    if(len(unknown)>0):
        print("====================================")
        print("THERE ARE UNKNOWN ARGUMENTS PASSED:")
        pprint(unknown)
        print("====================================")
        input("Enter a key to continue...")
    return args

args=parse_args()

# generate the paths
DATA_ROOT = os.path.expanduser('~/gnn/data/graphs')
# Create a dictionary to hold the data paths
DATA_PATHS = {}
for dataset_XX in ['cora', 'pubmed', 'citeseer', 'tinygraph', 'ego-facebook', 'corafull', 'wiki', 'blogcatalog']:
    DATA_PATHS[dataset_XX] = {}
    if dataset_XX in ['cora', 'pubmed', 'citeseer', 'tinygraph', 'ego-facebook', 'corafull']:
        DATA_PATHS[dataset_XX]['nodelist'] = os.path.join(DATA_ROOT, dataset_XX, f'{dataset_XX}_nodes.txt')
        DATA_PATHS[dataset_XX]['features'] = os.path.join(DATA_ROOT, dataset_XX, f'{dataset_XX}_x.txt')
        DATA_PATHS[dataset_XX]['labels']   = os.path.join(DATA_ROOT, dataset_XX, f'{dataset_XX}_y.txt')
        DATA_PATHS[dataset_XX]['edgelist'] = os.path.join(DATA_ROOT, dataset_XX, f'{dataset_XX}_edgelist.txt')
    elif dataset_XX in ['wiki']:
        DATA_PATHS[dataset_XX]['nodelist'] = os.path.join(DATA_ROOT, 'wiki', 'Wiki_nodes.txt')
        DATA_PATHS[dataset_XX]['features'] = os.path.join(DATA_ROOT, 'wiki', 'Wiki_category.txt')
        DATA_PATHS[dataset_XX]['labels']   = os.path.join(DATA_ROOT, 'wiki', 'Wiki_labels.txt')
        DATA_PATHS[dataset_XX]['edgelist'] = os.path.join(DATA_ROOT, 'wiki', 'Wiki_edgelist.txt')
    elif dataset_XX in ['blogcatalog']:
        DATA_PATHS[dataset_XX]['nodelist'] = os.path.join(DATA_ROOT, 'BlogCatalog-dataset', 'data', 'nodes.csv')
        DATA_PATHS[dataset_XX]['edgelist'] = os.path.join(DATA_ROOT, 'BlogCatalog-dataset', 'data', 'edges.csv')
        DATA_PATHS[dataset_XX]['labels']   = os.path.join(DATA_ROOT, 'BlogCatalog-dataset', 'data', 'labels.csv')


thisdate = datetime.now().strftime('%y%m%d-%H%M')
# thisdate = "240102-1536"
def generate_program_command(fd_version, dataset_name, ndim, 
                             epochs=5000, save_stats_every=1, outputdir_root=f'./embeddings-tmp-{thisdate}'):
    paths=DATA_PATHS[dataset_name]
    cmd = f"python3 -m forcedirected --fdversion {fd_version} "
    cmd+= f"--dataset_name {dataset_name} "
    cmd+= f"--ndim {ndim} "
    cmd+= f"--edgelist {paths['edgelist']} "
    cmd+= f"--nodelist {paths['nodelist']} "
    cmd+= f"--epochs {epochs} --save-stats-every {save_stats_every} "
    cmd+= f"--outputdir_root {outputdir_root} "
    return cmd

def get_count(partition, user='hessamla'):
    """Get the number of jobs in the queue, on the partition"""
    cmd = f"squeue -p {partition} -u {user} | grep {user} | wc -l"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if (len(r.stdout) > 0): r = int(r.stdout)
    else: r = 0
    return int(r)

LOGDIR = "./tmplog"
os.makedirs(LOGDIR, exist_ok=True)

def generate_sbatch_script(partition, duration, jobname, bash_cmd="{bash_cmd}", mem="200G", outpath=f"{LOGDIR}/fd-%j-%x.txt", errpath=f"{LOGDIR}/fd-%j-%x.err"):
    sbatch_flags = "-A r00372 "
    sbatch_flags += f" -p {partition} --time={duration} -J {jobname}"
    sbatch_flags += f" --mem={mem} --nodes=1 --ntasks-per-node=1"
    sbatch_flags += f" -o {outpath} -e {errpath}"
    sbatch_flags += " --gpus-per-node=1" if partition in ['gpu', 'gpu-debug'] else ""
    sbatch_cmd = f"sbatch {sbatch_flags} --wrap \"{bash_cmd}\""
    return sbatch_cmd

def run_cmd (cmd):
    ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    v=''
    if(len(ret.stdout) > 0): v = ret.stdout
    if(len(ret.stderr) > 0): v += ret.stderr
    if(v[-1]=='\n'): v = v[:-1]
    return v

def run_sbatch_script(cmd):
    # program_cmd = f"{program_run_cmd}"
    sbatch_flags = f"-p {partition} --time={duration} -J {ndim}-{dataset_name}"
    sbatch_flags += f" -A r00372 --mem=200G --nodes=1 --ntasks-per-node=1"
    sbatch_flags += " --gpus-per-node=1" if partition in ['gpu', 'gpu-debug'] else ""
    sbatch_flags += f" -o {LOGDIR}/fd-%j-%x.txt -e {LOGDIR}/fd-%j-%x.err"
    sbatch_cmd = f"sbatch {sbatch_flags} --wrap \"{module_cmd}; {cmd}\""
    print(sbatch_cmd)
    ret = subprocess.run(sbatch_cmd, 
                        shell=True, capture_output=True, text=True)
    
    v=''
    if(len(ret.stdout) > 0): v = ret.stdout
    if(len(ret.stderr) > 0): v += ret.stderr
    if(v[-1]=='\n'): v = v[:-1]
    return v

class Namespace():
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
ns = Namespace

# Set time based on partition
PARTITION_SETTINGS = {
    'gpu': ns(duration="1-23:59:59", maxsubmit=500),
    'general': ns(duration="3-23:59:59", maxsubmit=500),
    'gpu-debug': ns(duration="0:59:59", maxsubmit=4),
    'debug': ns(duration="0:59:59", maxsubmit=4)
}
partition = args.partition
duration = PARTITION_SETTINGS[partition].duration
maxsubmit = PARTITION_SETTINGS[partition].maxsubmit

# Set MODULE_CMD based on $HOME
home = os.environ['HOME']
login_node = ''
module_cmd = ''

if "Carbonate" in home:
    login_node = 'carbonate'
    module_cmd = 'module load python/gpu/3.10.10; module unload cudatoolkit/11.7 '
elif "BigRed200" in home:
    login_node = 'bigred200'
    module_cmd = 'module load python/gpu/3.10.10 '
else:
    module_cmd = 'echo No module command provided '


# %%
# Write to slurm-jobids.txt
with open('slurm-jobids.txt', 'a') as file:
    file.write("----------------------------------\n")
    file.write(datetime.now().strftime('%y-%m-%d,%H:%M:%S') + "\n")
    file.write(f"Partition:  {partition}\n")
    file.write(f"Login Node: {login_node}\n")


# Loop through dimensions, methods, and datasets
methods = ['forcedirected']
datasets = ['corafull', 'pubmed', 'cora', 'citeseer', 'ego-facebook', 'wiki']
ndims = [128, 64, 32, 24, 12] # start with the larger datasets and dimensions

datasets = ['corafull', 'pubmed']
ndims = [128, 64, 32, 24, 12] # start with the larger datasets and dimensions
# ndims = [12] # start with the larger datasets and dimensions

for method in methods:
    for ndim in ndims:
        for dataset_name in datasets:
            time.sleep(0.1) # wait a bit to make sure that the job submit is settled
            cnt = get_count(partition)
            while(cnt >= maxsubmit):
                print(f"(this pid {os.getpid()})   job count:{cnt} sleeping")
                time.sleep(60)
                cnt = get_count(partition)
            # make the program command
            sbatch_script = generate_sbatch_script(partition, duration, f"{ndim}-{dataset_name}")
            cmd = generate_program_command(args.fdversion, dataset_name, ndim)
            print(cmd)

            if(args.submitjobs is False):
                print(" ** NOT SUBMITTED. The --submitjobs flag is required.\n")
                continue
            submit_script = sbatch_script.format(bash_cmd=cmd)
            ret = run_cmd(submit_script)
            # ret = run_sbatch_script(cmd)
            print(ret)
            with open('slurm-jobids.txt', 'a') as file:
                file.write(cmd+"\n")
                file.write(sbatch_script+"\n")
                file.write(ret+"\n")
