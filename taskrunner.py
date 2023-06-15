# %%
import subprocess

def submit_slurm_batchjob(job_script):
    # Submit a Slurm job using sbatch command
    try:
        result = subprocess.run(['sbatch', job_script], capture_output=True, text=True)
        output = result.stdout.strip()
        job_id = output.split()[-1]  # Extract the job ID from the output
        print(f"Submitted Slurm job with ID: {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting Slurm job: {e}")
        return None

def submit_slurm_batchjob(job_script):
    # Submit a Slurm job using sbatch command
    try:
        result = subprocess.run(['sbatch', job_script], capture_output=True, text=True)
        output = result.stdout.strip()
        job_id = output.split()[-1]  # Extract the job ID from the output
        print(f"Submitted Slurm job with ID: {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting Slurm job: {e}")
        return None

def check_slurm_job_status(job_id):
    # Check the status of a Slurm job using squeue command
    try:
        result = subprocess.run(['squeue', '-j', job_id], capture_output=True, text=True)
        output = result.stdout.strip()
        if output:
            print("Job is still running")
        else:
            print("Job has completed")
    except subprocess.CalledProcessError as e:
        print(f"Error checking Slurm job status: {e}")

def cancel_slurm_job(job_id):
    # Cancel a Slurm job using scancel command
    try:
        subprocess.run(['scancel', job_id])
        print(f"Canceled Slurm job with ID: {job_id}")
    except subprocess.CalledProcessError as e:
        print(f"Error canceling Slurm job: {e}")

if __name__ == '__main__':
    # Example usage
    job_script = 'your_job_script.sh'  # Specify the path to your Slurm job script
    
    srun_script = """
    -p gpu -A general
    python -c "print('test')"
    """
    
    # Submit a Slurm job
    job_id = submit_slurm_job(srun_script)
    
    # Check the status of the Slurm job
    if job_id:
        check_slurm_job_status(job_id)
    
    # # Cancel the Slurm job
    # if job_id:
    #     cancel_slurm_job(job_id)

# %%
