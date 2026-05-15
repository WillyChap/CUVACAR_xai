#!/bin/bash -l
#PBS -N IGrun
#PBS -l select=1:ngpus=1:gpu_type=a100_80gb:mem=120GB:ncpus=16
#PBS -l walltime=12:00:00
#PBS -A "P06010014"
#PBS -q casper@casper-pbs
#PBS -o logs/pregen_^array_index^.log

module load conda
conda activate /glade/work/wchapman/conda-envs/credit-coupling

# Loop through all dates in 1981
start_date="1981-10-15"
end_date="1981-12-31"

current_date="$start_date"

while [[ $(date -d "$current_date" +%s) -le $(date -d "$end_date" +%s) ]]; do
    # Format date components
    date_str=$(date -d "$current_date" +"%Y-%m-%d")
    echo "Processing date: $date_str"
    
    if [ -d "/glade/derecho/scratch/kjmayer/CUVACAR_xai/IG/${date_str}" ]; then
        echo "Directory exists."
    else
        mkdir -p /glade/derecho/scratch/kjmayer/CUVACAR_xai/IG/${date_str}/
    fi
    
    python IntegratedGradients_Climo_Baseline.py \
        --config camulator_config.yml \
        --input_shape 1 138 1 192 288 \
        --forcing_shape 1 4 1 192 288 \
        --output_shape 1 145 1 192 288 \
        --device cuda \
        --model_name checkpoint.pt00091.pt \
        --init_tensor /glade/derecho/scratch/kjmayer/CUVACAR_xai/init_b2014_${date_str}_00_00_00_be21_condition_tensor.pth \
        --baseline_tensor /glade/derecho/scratch/wchapman/CUVACAR/init_${date_str}_00_00_00_be21_condition_tensor_baseline.pth \
        --IGsave_path /glade/derecho/scratch/kjmayer/CUVACAR_xai/IG/${date_str}/

    # Check for errors
    if [[ $? -ne 0 ]]; then
        echo "ERROR: Failed on date $date_str" >&2
        # Optional: exit 1  # Uncomment to stop on first failure
    fi

    # Advance by one day
    current_date=$(date -d "$current_date + 1 day" +"%Y-%m-%d")
done

echo "All dates processed."
