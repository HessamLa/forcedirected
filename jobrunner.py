# %%
import subprocess
import sys

# %%
# import subprocess

# filename = "testjob.py"
# argslist = ['a1', 'a2']
# sbatch_script = f"""#!/bin/bash
# #SBATCH -J {filename}
# #SBATCH -p gpu-debug 
# #SBATCH -A general 
# #SBATCH -o testjob_%j.txt 
# #SBATCH -e testjob_%j.err 
# #SBATCH --nodes=1 
# #SBATCH --ntasks-per-node=1
# #SBATCH --gpus-per-node=1
# #SBATCH --time=04:00:00
# #SBATCH --mem=32G

# python {filename} {' '.join(argslist)}
# """
# print(sbatch_script)
# try:
#     result = subprocess.run(['sbatch'], input=sbatch_script, capture_output=True, text=True, check=True)
#     print(result.stdout)
#     print(result.stderr)
# except subprocess.CalledProcessError as e:
#     print(f"Failed to submit job: {e}")


# %%
import subprocess

codes = []
commands = []
for dsname in ['cora', 'ego-facebook']:
    for add_noise in ['--add-noise ', '']:
        for do_random_selection in ['--do-random-selection ', '']:
            noisetag = "noise" if len(add_noise)>0 else ""
            rselecttag = "rselect" if len(do_random_selection)>0 else ""

            # dsname="cora"
            # add_noise = "--add-noise "
            # do_random_selection = "--do-random-selection "
            taskname="embed"
            pycode = f" embed.py --dataset={dsname} "
            pycode+= f" --outputdir=embeddings-test/{dsname}-{noisetag}-{rselecttag}"
            pycode+= f" {add_noise} {do_random_selection}"
            pycode+= f" --description \"dataset:{dsname} {noisetag} {rselecttag}\""
            
            outfile = f"{taskname}-{dsname}_%j.txt" 
            errfile = f"{taskname}-{dsname}_%j.err"

            command = f"srun -o {outfile} -e {errfile}"
            command+=f" python {pycode} " 

            codes.append(pycode)
            commands.append(command)
for c in codes:
    print(c)
print(codes)
# %%
def submit_job(sbatch_directives, job_name, command):
    script_content = f"{sbatch_directives}\n{command}"
    print(f"submitting sbatch script: \n{script_content}")
    try:
        result = subprocess.run(['sbatch'], input=script_content, capture_output=True, text=True, check=True)
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job: {e}")
        return None

    # Extract the job ID from the sbatch output
    job_id = extract_job_id(result.stdout)
    if job_id is None:
        print("Failed to extract job ID.")
        return None

    print(f"Job submitted with ID: {job_id}")

    # Write the job information to a file for later reference
    write_job_info(job_id, job_name, command)

    return job_id

def extract_job_id(output):
    """
    Extracts the job ID from the sbatch output.
    Modify this function based on the output format of your SLURM cluster.
    """
    lines = output.splitlines()
    if len(lines) < 1:
        return None

    # Assuming the job ID is present in the first line
    job_id = lines[0].strip().split()[-1]
    return job_id

def write_job_info(job_id, job_name, command):
    # Write the job information to a file
    with open("job_info.txt", "a") as f:
        f.write(f"Job ID: {job_id}\n")
        f.write(f"Job Name: {job_name}\n")
        f.write(f"command: {command}\n")
        f.write("\n")

# Example usage
if __name__ == "__main__":
    # Specify the SLURM script content
    sbatch_script = f"""#!/bin/bash
#SBATCH -J {taskname}
#SBATCH -p gpu 
#SBATCH -A general 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
"""

for i, pycode in enumerate(codes):
    command = commands[i]

    # Submit the job
    job_id = submit_job(sbatch_script, pycode, command)

    # Check if job submission was successful
    if job_id is not None:
        # Do further processing or monitoring of the job
        # For example, you can store the job ID and other information for later use
        print("Job submission successful!")
