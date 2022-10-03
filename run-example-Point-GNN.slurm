#!/bin/bash
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=run-example-Point-GNN

# Request CPU resource for a serial job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1


# Request for GPU

#SBATCH --gres=gpu:V100:1
#SBATCH --account=hk34
#SBATCH --partition=m3g


# Memory usage (MB)
# SBATCH --mem-per-cpu=4000

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=1-00:00:00


# To receive an email when job completes or fails
# SBATCH --mail-user=riley.henderson1.monash.edu
# SBATCH --mail-type=END
# SBATCH --mail-type=FAIL


# Set the file for output (stdout)
#SBATCH --output=/scratch/hk34/liu/Point-GNN/Point-GNN/output/logs/run-example-Point-GNN-%j.out

# Set the file for error log (stderr)
#SBATCH --error=/scratch/hk34/liu/Point-GNN/Point-GNN/output/logs/run-example-Point-GNN-%j.err

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

# run the example Point-GNN inference script
python run.py checkpoints/car_auto_T3_train/ --dataset_root_dir /scratch/hk34/henderson/KITTI_dataset/ --output_dir /scratch/hk34/liu/Point-GNN/Point-GNN/output/

