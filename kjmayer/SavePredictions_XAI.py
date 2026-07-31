"""Save CAMULATOR predictions for every initial condition used in the XAI (IG) runs.

The IG runs in IntegratedGradients_Climo_Baseline.py saved attributions but not the
predictions they attribute. This script fills that gap: for each XAI initial condition it
runs the model forward one step (lead_time_periods = 6 h) and saves

    F(x)         -- prediction from the b2014 initial condition
    F(baseline)  -- prediction from the matching climatology baseline

Both are saved as raw, un-rescaled model output, i.e. exactly the tensor that
integrated_gradients_chunked differentiates. That keeps them in the same normalized space
as the IG files and as the init/baseline tensors, so:

  * completeness can be checked directly for any IG target (lev, 0, lat, lon):
        IG.sum()  ==  F(x)[0, lev, 0, lat, lon] - F(baseline)[0, lev, 0, lat, lon]
    Every one of the 126 IG targets for a date is contained in the single saved field.
    The check runs inline and is appended to a CSV per date (see --completeness_targets),
    so a partial run still leaves usable diagnostics.
  * a 00Z prediction can be compared against the 06Z init tensor of the same day
    (they are both valid at 06Z), which is what the U-evolution comparison needs.

Output tensors are [1, 145, 1, 192, 288] float32 (~32 MB each). Channel order follows the
model output convention: 0:32 U, 32:64 V, 64:96 T, 96:128 Qtot, 128 PS, 129 TREFHT,
130:145 diagnostics (PRECT, TS, CLDHGH, ...). Note this differs from the 136-channel input
tensor, which carries 6 input-only forcings instead of the 15 diagnostics.

How to run:
    python SavePredictions_XAI.py --config camulator_config.yml \
        --device cuda --model_name checkpoint.pt00091.pt \
        --save_path /glade/derecho/scratch/kjmayer/CUVACAR_xai/predictions/
"""

import os
import re
import csv
import sys
import glob
import time
import yaml
import logging
import warnings
import argparse
from datetime import datetime, timedelta

import numpy as np
import torch

# ---------- #
# credit
from credit.models import load_model, load_model_name
from credit.parser import credit_main_parser

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Defaults match the paths hard-coded in runxai_baseline.sh / IntegratedGradients_Climo_Baseline.py
IG_DIR = "/glade/derecho/scratch/kjmayer/CUVACAR_xai/IG/"
INIT_DIR = "/glade/derecho/scratch/kjmayer/CUVACAR_xai/"
BASELINE_DIR = "/glade/derecho/scratch/wchapman/CUVACAR/"
# Writes go to wchapman scratch: the IG/init tree under kjmayer scratch is world-readable but
# not world-writable, and ToDo.txt notes kjmayer is short on space. This directory already holds
# the climatology baselines and is world-readable, so both users can get at the output.
SAVE_DIR = "/glade/derecho/scratch/wchapman/CUVACAR/predictions/"

# tensor_steps24_parrallel_lev00014_lat00016_lon00000.npy
IG_PATTERN = re.compile(r"tensor_steps\d+_parrallel_lev(\d+)_lat(\d+)_lon(\d+)\.npy$")

CSV_FIELDS = [
    "init_date", "init_hour", "valid_time",
    "lev", "var_idx", "lat", "lon",
    "ig_sum", "f_init", "f_baseline", "f_diff",
    "abs_gap", "rel_gap",
]


def init_tensor_path(init_dir, date_str, hour):
    """Init condition written by the XAI pre-processing (b2014 states, 1981 tag)."""
    return os.path.join(
        init_dir,
        f"init_b2014_{date_str}_{hour}_00_00_be21_condition_tensor.pth",
    )


def baseline_tensor_path(baseline_dir, date_str, hour):
    """Climatology baseline matching the init condition above."""
    return os.path.join(
        baseline_dir,
        f"init_{date_str}_{hour}_00_00_be21_condition_tensor_baseline.pth",
    )


def prediction_path(save_path, date_str, hour, lead_hours, kind):
    """Prediction filename carrying BOTH the init time and the valid time.

    kind is 'init' (prediction from the b2014 state) or 'baseline'.
    Sorting on this name still sorts chronologically by init time.
    """
    valid = datetime.strptime(f"{date_str} {hour}", "%Y-%m-%d %H") + timedelta(hours=lead_hours)
    tag = "prediction" if kind == "init" else "prediction_baseline"
    return os.path.join(
        save_path,
        f"{tag}_init{date_str}_{hour}Z_valid{valid:%Y-%m-%d}_{valid:%H}Z.npy",
    )


def collect_dates(args):
    """Date list: by default exactly the dates the IG runs produced output for."""
    if args.dates_from == "ig":
        dates = sorted(
            d for d in os.listdir(args.ig_dir)
            if os.path.isdir(os.path.join(args.ig_dir, d)) and len(d) == 10 and d[4] == "-"
        )
        if not dates:
            raise FileNotFoundError(f"No dated IG directories found under {args.ig_dir}")
        return dates

    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    if end < start:
        raise ValueError(f"--end_date {args.end_date} precedes --start_date {args.start_date}")

    dates = []
    current = start
    while current <= end:
        dates.append(f"{current:%Y-%m-%d}")
        current += timedelta(days=1)
    return dates


def ig_targets_for_date(ig_dir, date_str, n_targets):
    """IG targets available for a date, parsed back out of the filenames.

    n_targets=None checks every target; a smaller count takes an evenly spaced subset of the
    sorted (lev, lat, lon) list so the sample still spans levels and locations.
    """
    pattern = os.path.join(ig_dir, date_str, "tensor_steps*_parrallel_lev*_lat*_lon*.npy")

    targets = []
    for path in sorted(glob.glob(pattern)):
        match = IG_PATTERN.search(os.path.basename(path))
        if match:
            lev, lat, lon = (int(g) for g in match.groups())
            targets.append((lev, lat, lon, path))

    if n_targets is not None and 0 < n_targets < len(targets):
        keep = np.unique(np.linspace(0, len(targets) - 1, n_targets).round().astype(int))
        targets = [targets[i] for i in keep]

    return targets


def completeness_rows(fx, fb, targets, date_str, hour, valid_str, var_idx=0):
    """Check sum(IG) == F(x) - F(baseline) for each target.

    Exact only in the limit of infinite integration steps; the IG files were made with
    num_steps=24, so a small residual is expected. Summation is accumulated in float64 --
    the IG fields hold ~7.5e6 float32 values and we do not want the check measuring its own
    rounding error.
    """
    rows = []
    for lev, lat, lon, path in targets:
        ig_sum = float(np.load(path).sum(dtype=np.float64))
        f_init = float(fx[0, lev, var_idx, lat, lon])
        f_base = float(fb[0, lev, var_idx, lat, lon])
        f_diff = f_init - f_base
        abs_gap = abs(ig_sum - f_diff)

        rows.append({
            "init_date": date_str,
            "init_hour": hour,
            "valid_time": valid_str,
            "lev": lev,
            "var_idx": var_idx,
            "lat": lat,
            "lon": lon,
            "ig_sum": f"{ig_sum:.8e}",
            "f_init": f"{f_init:.8e}",
            "f_baseline": f"{f_base:.8e}",
            "f_diff": f"{f_diff:.8e}",
            "abs_gap": f"{abs_gap:.8e}",
            "rel_gap": f"{abs_gap / abs(f_diff):.8e}" if f_diff != 0.0 else "nan",
        })

    return rows


def append_csv_rows(csv_path, rows):
    """Append and flush per date, so a killed job still leaves usable diagnostics."""
    if not rows:
        return
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fo:
        writer = csv.DictWriter(fo, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
        fo.flush()


def existing_csv_keys(csv_path):
    """(date, hour) pairs already checked, so a resumed run does not duplicate rows."""
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as fo:
        return {(r["init_date"], r["init_hour"]) for r in csv.DictReader(fo)}


def load_camulator(config, model_name, device):
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
    conf = credit_main_parser(conf, parse_training=False, parse_predict=True, print_summary=False)
    conf["predict"]["mode"] = None

    print("...loading model...")
    if model_name:
        model = load_model_name(conf, model_name, load_weights=True)
    else:
        model = load_model(conf, load_weights=True)
    model = model.to(device)
    model.eval()

    # Predictions only -- no gradients anywhere.
    for p in model.parameters():
        p.requires_grad_(False)

    return model, conf


def predict_one(model, tensor_path, device):
    x = torch.load(tensor_path, map_location=torch.device(device)).to(device)
    with torch.no_grad():
        y = model(x.float())
    return y.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Save one-step (6 h) CAMULATOR predictions for the XAI initial conditions."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the model configuration YAML file.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run the model on (cuda or cpu).")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Optional model checkpoint name, e.g. checkpoint.pt00091.pt")
    parser.add_argument("--ig_dir", type=str, default=IG_DIR,
                        help="IG output root; its dated subdirectories define the default date list.")
    parser.add_argument("--init_dir", type=str, default=INIT_DIR,
                        help="Directory holding the b2014 init condition tensors.")
    parser.add_argument("--baseline_dir", type=str, default=BASELINE_DIR,
                        help="Directory holding the climatology baseline tensors.")
    parser.add_argument("--save_path", type=str, default=SAVE_DIR,
                        help="Directory to write predictions to (created if absent).")
    parser.add_argument("--dates_from", type=str, default="ig", choices=["ig", "range"],
                        help="'ig' uses the dates present in --ig_dir; 'range' uses start/end.")
    parser.add_argument("--start_date", type=str, default="1981-01-01",
                        help="First date, used only with --dates_from range.")
    parser.add_argument("--end_date", type=str, default="1981-12-31",
                        help="Last date, used only with --dates_from range.")
    parser.add_argument("--hours", type=str, nargs="+", default=["00"],
                        help="Init hours to process. IG was run on 00Z only; pass 00 06 12 18 "
                             "to cover every 6-hourly init instead.")
    parser.add_argument("--no_baseline", action="store_true",
                        help="Skip the baseline predictions (completeness checks need them).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute predictions whose output file already exists.")
    parser.add_argument("--completeness_csv", type=str, default=None,
                        help="CSV to append completeness checks to. "
                             "Default: <save_path>/completeness.csv")
    parser.add_argument("--completeness_targets", type=str, default="12",
                        help="IG targets to check per date: an integer (evenly spaced across the "
                             "126 available), 'all', or 'none' to skip the check. 'all' reads the "
                             "full ~3.8 GB of IG per date and adds roughly 90 min over a year.")

    args = parser.parse_args()
    start_time = time.time()

    os.makedirs(args.save_path, exist_ok=True)

    # Completeness needs both predictions to difference.
    n_targets = None
    do_completeness = not args.no_baseline
    if args.completeness_targets.lower() == "none":
        do_completeness = False
    elif args.completeness_targets.lower() != "all":
        n_targets = int(args.completeness_targets)
        if n_targets <= 0:
            do_completeness = False

    if args.no_baseline and args.completeness_targets.lower() != "none":
        print("NOTE: --no_baseline given, so the completeness check is disabled.", file=sys.stderr)

    csv_path = args.completeness_csv or os.path.join(args.save_path, "completeness.csv")
    checked = existing_csv_keys(csv_path) if do_completeness else set()

    dates = collect_dates(args)
    hours = [f"{int(h):02d}" for h in args.hours]

    model, conf = load_camulator(args.config, args.model_name, args.device)
    lead_hours = conf["data"]["lead_time_periods"]

    n_cases = len(dates) * len(hours)
    n_fields = n_cases * (1 if args.no_baseline else 2)
    print(f"...{len(dates)} dates x {len(hours)} hours = {n_cases} initial conditions...")
    print(f"...one model step = {lead_hours} h; writing ~{n_fields} fields to {args.save_path}...")

    if do_completeness:
        scope = "all" if n_targets is None else f"{n_targets}"
        print(f"...completeness: {scope} targets/date appended to {csv_path}"
              f" ({len(checked)} cases already present)...")

    n_written = 0
    n_skipped = 0
    n_checked = 0
    missing = []

    for date_str in dates:
        for hour in hours:
            jobs = [("init", init_tensor_path(args.init_dir, date_str, hour))]
            if not args.no_baseline:
                jobs.append(("baseline", baseline_tensor_path(args.baseline_dir, date_str, hour)))

            preds = {}
            for kind, tensor_path in jobs:
                out_path = prediction_path(args.save_path, date_str, hour, lead_hours, kind)

                if os.path.exists(out_path) and not args.overwrite:
                    n_skipped += 1
                    preds[kind] = (out_path, None)
                    continue

                if not os.path.exists(tensor_path):
                    print(f"MISSING {kind} tensor: {tensor_path}", file=sys.stderr)
                    missing.append(tensor_path)
                    continue

                arr = predict_one(model, tensor_path, args.device)
                np.save(out_path, arr)
                preds[kind] = (out_path, arr)
                n_written += 1

            have_both = "init" in preds and "baseline" in preds
            if do_completeness and have_both and (date_str, hour) not in checked:
                targets = ig_targets_for_date(args.ig_dir, date_str, n_targets)
                if targets:
                    # Reuse the arrays just predicted; only reload when the date was skipped.
                    fx = preds["init"][1]
                    fb = preds["baseline"][1]
                    fx = np.load(preds["init"][0]) if fx is None else fx
                    fb = np.load(preds["baseline"][0]) if fb is None else fb

                    valid_str = os.path.basename(preds["init"][0]).split("_valid")[1][:-4]
                    rows = completeness_rows(fx, fb, targets, date_str, hour, valid_str)
                    append_csv_rows(csv_path, rows)
                    checked.add((date_str, hour))
                    n_checked += len(rows)

        print(f"{date_str}: {n_written} written, {n_skipped} skipped, "
              f"{n_checked} checked, {len(missing)} missing")

    if args.device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start_time
    print(f"Done. {n_written} written, {n_skipped} already present, {len(missing)} inputs missing.")
    if do_completeness:
        print(f"Completeness: {n_checked} targets checked -> {csv_path}")
    print(f"Elapsed: {elapsed:.2f} s ({elapsed / 60:.2f} min)")

    if missing:
        print("Missing input tensors:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
