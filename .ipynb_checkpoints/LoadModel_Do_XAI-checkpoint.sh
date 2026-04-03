#!/bin/bash
#PBS -N Run_Noise_Script
#PBS -A NAML0001 
#PBS -l walltime=12:00:00
#PBS -o RUN_Climate_RMSE.out
#PBS -e RUN_Climate_RMSE.out
#PBS -q casper
#PBS -l select=1:ncpus=32:ngpus=1:mem=250GB
#PBS -l gpu_type=a100
#PBS -m a
#PBS -M wchapman@ucar.edu

module load conda
conda activate /glade/work/wchapman/conda-envs/credit-casper-modern
# python /glade/work/wchapman/miles_branchs/CESM_Spatial_PS/applications/Quick_Climate_Year_Slow_parrallel_opt.py --config /glade/derecho/scratch/wchapman/miles_branchs/CESM_Spatial_PS_WXmod/be21_coupled-v2025.2.0.yml --input_shape 1 136 1 192 288 --forcing_shape 1 6 1 192 288 --output_shape 1 145 1 192 288 --device cuda --init_noise 1 --model_name checkpoint.pt00275.pt

python -c "import torch; torch.cuda.empty_cache()"

CONFIG=/glade/work/wchapman/SphereOfInfluence_CUVACAR/be21_coupled-v2025.1.2.0_Prect_TS.yml
SCRIPT=/glade/work/wchapman/SphereOfInfluence_CUVACAR/IntegratedGradients_try.py
BASE_ARGS="--config $CONFIG \
  --input_shape 1 136 1 192 288 \
  --forcing_shape 1 6 1 192 288 \
  --output_shape 1 145 1 192 288 \
  --device cuda"


# Only run checkpoint 91
i=91
ckpt=$(printf "%05d" "$i")

# Make 15 ensembles
NENS=1
for e in $(seq 1 $NENS); do
  ens=$(printf "%02d" "$e")

  python $SCRIPT $BASE_ARGS \
    --model_name checkpoint.pt${ckpt}.pt \
    --save_append run_tester_${ckpt}_ens${ens} \
    --init_noise $e
done
