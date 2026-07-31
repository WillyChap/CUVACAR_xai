#!/bin/bash -l
#PBS -N PREDrun
#PBS -l select=1:ngpus=1:gpu_type=a100:mem=60GB:ncpus=8
#PBS -l walltime=04:00:00
#PBS -A NAML0001
#PBS -q casper@casper-pbs
#PBS -o logs/predictions.log

module load conda
conda activate /glade/work/wchapman/conda-envs/credit-coupling

# PBS starts the job in $HOME; the script and config below are referenced relatively.
cd "${PBS_O_WORKDIR:-$(dirname "$0")}" || exit 1

# Saves the one-step (6 h) prediction for every XAI initial condition, plus the matching
# climatology-baseline prediction. Unlike runxai_baseline.sh this loops over dates inside a
# single python process -- the forward passes are cheap, so reloading the model per date
# would dominate the runtime.
#
# Dates default to the dated subdirectories present under the IG output root, i.e. exactly
# the cases the IG runs covered, at 00Z. To cover every 6-hourly init instead, add:
#     --dates_from range --start_date 1981-01-01 --end_date 1981-12-31 --hours 00 06 12 18
#
# Each date also appends a completeness check (sum(IG) vs F(x)-F(baseline)) to
# $SAVE_PATH/completeness.csv. 12 of the 126 targets per date keeps the run I/O-cheap;
# --completeness_targets all checks every one but reads 3.8 GB of IG per date (~90 min more).

# Output goes to wchapman scratch -- kjmayer's CUVACAR_xai tree is readable but not writable by
# other users, and is where the 1.3 TB of IG already lives. Override --save_path if running as
# kjmayer.
SAVE_PATH=/glade/derecho/scratch/wchapman/CUVACAR/predictions/
mkdir -p "$SAVE_PATH" logs

python SavePredictions_XAI.py \
    --config camulator_config.yml \
    --device cuda \
    --model_name checkpoint.pt00091.pt \
    --ig_dir /glade/derecho/scratch/kjmayer/CUVACAR_xai/IG/ \
    --init_dir /glade/derecho/scratch/kjmayer/CUVACAR_xai/ \
    --baseline_dir /glade/derecho/scratch/wchapman/CUVACAR/ \
    --save_path "$SAVE_PATH" \
    --hours 00 \
    --completeness_targets 12 \
    --completeness_csv "$SAVE_PATH"/completeness.csv

if [[ $? -ne 0 ]]; then
    echo "ERROR: prediction run failed" >&2
    exit 1
fi

echo "All predictions written to $SAVE_PATH"
