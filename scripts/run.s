#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=3:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:k80:4
#SBATCH --reservation=cds-courses
#SBATCH --job-name=hpml-lab-4-css459
#SBATCH --mail-type=END
#SBATCH --mail-user=css@nyu.edu
#SBATCH --output=slurm_%j.out

cd $HOME/dl-final-project

module purge
module load  python3/intel/3.6.3
module load cuda/9.2.88

source $HOME/hpml/hpml-lab-4/src/py3.6.3/bin/activate
srun python3 resnet-train.py
