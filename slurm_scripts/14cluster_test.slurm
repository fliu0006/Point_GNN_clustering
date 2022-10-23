#!/bin/bash
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=14cluster_test

# Request CPU resource for a serial job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1


# Request for GPU

# SBATCH --gres=gpu:V100:2
#SBATCH --account=hk34
#SBATCH --partition=m3g


# Memory usage (MB)
# SBATCH --mem-per-cpu=4000
# SBATCH --mem=30000

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=1-00:00:00


# To receive an email when job completes or fails
#SBATCH --mail-user=fliu0006@student.monash.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set the file for output (stdout)
#SBATCH --output=/scratch/hk34/liu/Point-GNN/Point-GNN/output/14cluster_test/improved_T1/logs/cluster_train-%j.out

# Set the file for error log (stderr)
#SBATCH --error=/scratch/hk34/liu/Point-GNN/Point-GNN/output/14cluster_test/improved_T1/logs/cluster_train-%j.err

# --------------------------------------------------------
# Script:
# --------------------------------------------------------

# load correct python version and activate virtual environment.
module load python/3.7.3-system				
cd /scratch/hk34/liu/Point-GNN/Point-GNN	
source ../bin/activate				
	
# load cuda and cudnn packages (dependencies for tensorflow GPU)
module load cuda/10.0
module load cudnn/7.6.5.32-cuda10

# Load MPI for multi-task capability, then train the T3 network on Riley's 200,000 cluster images
python run.py checkpoints/cluster_improved_T1_train/ --dataset_root_dir /scratch/hk34/henderson/no_noise/21by21/14cluster_ssd4_test --dataset_split_file /scratch/hk34/henderson/no_noise/21by21/14cluster_ssd4_test/14cluster_ssd4_testlabels.csv --output_dir /scratch/hk34/liu/Point-GNN/Point-GNN/output/14cluster_test/improved_T1/
