import os
import gc
import sys
import yaml
import time
import logging
import warnings
import copy
from glob import glob
from pathlib import Path
from argparse import ArgumentParser
import multiprocessing as mp
from collections import defaultdict
import cftime
from cftime import DatetimeNoLeap
import json
import pickle
import argparse
import time

# ---------- #
# Numerics
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

# ---------- #
import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data import get_worker_info
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import profile, record_function, ProfilerActivity

# ---------- #
# credit
from credit.models import load_model, load_model_name
from credit.seed import seed_everything
# from credit.loss import latitude_weights

from credit.data import (
    concat_and_reshape,
    reshape_only,
    drop_var_from_dataset,
    generate_datetime,
    nanoseconds_to_year,
    hour_to_nanoseconds,
    get_forward_data,
    extract_month_day_hour,
    find_common_indices,
)

from credit.transforms import load_transforms, Normalize_ERA5_and_Forcing
from credit.pbs import launch_script, launch_script_mpi
from credit.pol_lapdiff_filt import Diffusion_and_Pole_Filter
from credit.metrics import LatWeightedMetrics
from credit.forecast import load_forecasts
from credit.distributed import distributed_model_wrapper, setup
from credit.models.checkpoint import load_model_state
from credit.parser import credit_main_parser, predict_data_check
from credit.output import load_metadata, make_xarray, save_netcdf_increment
from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def get_gradients(model, inputs, top_pred_idx=None):
    """
    Computes the gradients of outputs w.r.t input image.

    Args:
        model: The neural network model.
        inputs: 2D/3D/4D tensor of samples.
        top_pred_idx: (optional) Predicted label for the inputs
                      if classification problem. If regression,
                      do not include.

    Returns:
        Gradients of the predictions w.r.t inputs.
    """
    # Ensure inputs are of type float and require gradients
    inputs = inputs.clone().detach().float()
    inputs.requires_grad = True

    # Forward pass
    preds = model(inputs)

    # For classification, grab the top class
    if top_pred_idx is not None:
        preds = preds[:,top_pred_idx[0], top_pred_idx[1], top_pred_idx[2], top_pred_idx[3]]        

    # Backward pass to compute gradients
    preds.backward(torch.ones_like(preds))

    # Retrieve the gradients of the inputs
    grads = inputs.grad
    return grads


def get_integrated_gradients(inputs, model, baseline=None, num_steps=50, top_pred_idx=None):
    """Computes Integrated Gradients for a prediction.

    Args:
        inputs (ndarray): 2D/3D/4D matrix of samples
        baseline (ndarray): The baseline image to start with for interpolation
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.
        top_pred_idx: (optional) Predicted label for the x_data
                      if classification problem. If regression,
                      do not include.            

    Returns:
        Integrated gradients w.r.t input image
    """
    # If baseline is not provided, start with zeros
    # having same size as the input image.
    device = 'cuda'
    if baseline is None:
        input_size = np.shape(inputs)[1:]
        baseline = torch.tensor(np.zeros(input_size).astype(np.float32)).to(device)
    else:
        baseline = torch.load(baseline, map_location=torch.device(device)).to(device)
        baseline = torch.tensor(baseline.astype(np.float32)).to(device)

    # 1. Do interpolation.
    inputs = inputs.float()
    interpolated_inputs = [
        baseline + (step / num_steps) * (inputs - baseline)
        for step in range(num_steps + 1)
    ]
    # interpolated_inputs = torch.tensor(interpolated_inputs)

    # 3. Get the gradients
    grads = []
    for i, x_data in enumerate(interpolated_inputs):
        grad = get_gradients(model, x_data, top_pred_idx=top_pred_idx).cpu().numpy()     
        grads.append(grad)
    grads = np.asarray(grads)

    # 4. Approximate the integral using the trapezoidal rule
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = torch.mean(torch.from_numpy(grads).to(device), dim=0)
    # 5. Calculate integrated gradients and return
    integrated_grads = (inputs - baseline) * avg_grads
    return integrated_grads



def integrated_gradients_chunked(x, model, target, num_steps=50, step_chunk=12, baseline=None):
    """
    x: input tensor (your loaded x), shape like [B, ...]
    target: tuple (lev, var, lat, lon) selecting a scalar from model output
    """
    device = x.device
    if baseline is None:
        baseline = torch.zeros_like(x)

    lev, var_idx, lat, lon = target
    dx = (x - baseline)

    # alpha grid includes endpoints: 0..1 (num_steps intervals)
    alphas = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=torch.float32)

    # trapezoid sum over intervals
    trap_sum = torch.zeros_like(x, dtype=torch.float32)
    prev_grad = None

    model.eval()

    # important: avoid parameter grads overhead
    for p in model.parameters():
        p.requires_grad_(False)

    s = 0
    B = x.shape[0]

    while s < alphas.numel():
        a = alphas[s:s + step_chunk]   # [m]
        m = a.numel()

        # build batch of interpolated inputs: [m*B, ...]
        # a_view becomes [m,1,1,...] and broadcasts over x
        a_view = a.view(m, *([1] * x.ndim))
        xb = baseline.unsqueeze(0) + a_view * dx.unsqueeze(0)   # [m, B, ...]
        xb = xb.view(m * B, *x.shape[1:]).detach().requires_grad_(True)

        out = model(xb)  # [m*B, ...]
        sel = out[:, lev, var_idx, lat, lon].sum()

        grads = torch.autograd.grad(sel, xb, retain_graph=False, create_graph=False)[0]
        grads = grads.view(m, B, *x.shape[1:]).to(dtype=torch.float32)  # [m,B,...]

        # trapezoid accumulation across alpha boundaries
        if prev_grad is not None:
            trap_sum += 0.5 * (prev_grad + grads[0])

        if m > 1:
            trap_sum += 0.5 * (grads[:-1] + grads[1:]).sum(dim=0)

        prev_grad = grads[-1]
        s += step_chunk

    avg_grads = trap_sum / float(num_steps)
    ig = dx * avg_grads
    return ig.to(dtype=x.dtype)


def save_task(data, meta_data, conf):
    """Wrapper function for saving data in parallel."""
    darray_upper_air, darray_single_level, init_datetime_str, lead_time, forecast_hour = data
    save_netcdf_increment(
        darray_upper_air,
        darray_single_level,
        init_datetime_str,
        lead_time * forecast_hour,
        meta_data,
        conf,
    )


class ForcingDataset(IterableDataset):
    """
    Streams dynamic forcing data in time-chunks, moves them to GPU asynchronously.
    """
    def __init__(self, ds, df_vars, start, num_ts, chunk_size):
        self.ds = ds
        self.df_vars = df_vars
        self.start = start
        self.num_ts = num_ts
        self.chunk = chunk_size

    def __iter__(self):
        for block_start in range(self.start, self.start + self.num_ts, self.chunk):
            block_end = min(block_start + self.chunk, self.start + self.num_ts)
            arr = self.ds.isel(time=slice(block_start, block_end)).load().values
            cpu_tensor = torch.from_numpy(arr).pin_memory()
            gpu_tensor = cpu_tensor.to(self.ds.encoding.get('device', torch.device('cuda')), non_blocking=True)
            yield gpu_tensor



def shift_input_for_next(x, y_pred, history_len, varnum_diag, static_dim):
    """
    Roll the input tensor forward by one time-step using cached dimension values.
    """
    if history_len == 1:
        if varnum_diag > 0:
            return y_pred[:, :-varnum_diag, ...].detach()
        else:
            return y_pred.detach()
    else:
        if static_dim == 0:
            x_detach = x[:, :, 1:, ...].detach()
        else:
            x_detach = x[:, :-static_dim, 1:, ...].detach()
        if varnum_diag > 0:
            return torch.cat([x_detach, y_pred[:, :-varnum_diag, ...].detach()], dim=2)
        else:
            return torch.cat([x_detach, y_pred.detach()], dim=2)



def run_year_rmse(p, config, input_shape, forcing_shape, output_shape, device, model_name=None, init_noise=None, save_append=None, init_tensor=None, baseline_tensor=None):
    # Load and parse configuration
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
    conf = credit_main_parser(conf, parse_training=False, parse_predict=True, print_summary=False)
    conf["predict"]["mode"] = None

    if save_append:
        base = conf["predict"].get("save_forecast")
        if not base:
            raise KeyError("'save_forecast' missing in config")
        conf["predict"]["save_forecast"] = str(Path(base).expanduser() / save_append)

    # Cache these once for speed
    history_len = conf["data"]["history_len"]
    varnum_diag = len(conf["data"]["diagnostic_variables"])
    static_dim = len(conf["data"]["static_variables"]) if not conf["data"]["static_first"] else 0

    print('...starting transform...')
    
    # Transform and model setup
    transform = load_transforms(conf)
    state_transformer = Normalize_ERA5_and_Forcing(conf) if conf["data"]["scaler_type"] == "std_new" else _not_supported()
    model = (load_model_name(conf, model_name, load_weights=True) if model_name else load_model(conf, load_weights=True)).to(device)
    truth_field = torch.load(conf['predict']['seasonal_mean_fast_climate'], map_location=torch.device(device))
    
    print('...loading model...')
    
    distributed = conf["predict"]["mode"] in ["ddp", "fsdp"]
    if distributed:
        model = distributed_model_wrapper(conf, model, device)
        if conf["predict"]["mode"] == "fsdp": model = load_model_state(conf, model, device)
    model.eval()

    x = torch.load(init_tensor, map_location=torch.device(device)).to(device)

    if init_noise is not None:
        print('adding forecast noise')
        # Define the standard deviation for the noise (e.g., 0.01)
        noise_std = 0.05
        
        # Generate random noise tensor with the same shape as `x`
        # Use `torch.randn` for a normal distribution with mean=0 and std=1
        noise = torch.randn_like(x) * noise_std

        # Add the noise to `x`
        x = x + noise.to(device)
    
    # Post-processing flags (cached)
    post_conf = conf["model"]["post_conf"]
    flag_mass = post_conf["activate"] and post_conf["global_mass_fixer"]["activate"]
    flag_water = post_conf["activate"] and post_conf["global_water_fixer"]["activate"]
    flag_energy = post_conf["activate"] and post_conf["global_energy_fixer"]["activate"]
    if flag_mass: opt_mass = GlobalMassFixer(post_conf)
    if flag_water: opt_water = GlobalWaterFixer(post_conf)
    if flag_energy: opt_energy = GlobalEnergyFixer(post_conf)

    print('...done with fixers...')
    print('...done model...')
    print('...done model...')
    print(model)

    print(f'preds shape: {model(x).shape}')
    print('doing IG')


    start_time = time.perf_counter()
    device = 'cuda'

    ## add for loop here: 


    

    out_vec_zeros = model(x)

    arr = out_vec_zeros.detach().cpu().numpy()
    np.save(
            f"/glade/derecho/scratch/wchapman/sphereofinf/tensor_zeros_prediction.npy",
            arr
        )

    
    return arr



def main():
    parser = argparse.ArgumentParser(description="Run year RMSE for CAMULATOR model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the model configuration YAML file.')
    parser.add_argument('--input_shape', type=int, nargs='+', required=True, help='Input shape as a list of integers.')
    parser.add_argument('--forcing_shape', type=int, nargs='+', required=True, help='Forcing shape as a list of integers.')
    parser.add_argument('--output_shape', type=int, nargs='+', required=True, help='Output shape as a list of integers.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (cuda or cpu).')
    parser.add_argument('--model_name', type=str, default=None, help='Optional model checkpoint name.')
    parser.add_argument('--init_noise', type=int, default=None, help='init model noise')
    parser.add_argument('--save_append', type=str, default=None, help='append a folder to the output to save to')
    parser.add_argument('--init_tensor', type=str, default=None, help='path to initial condition')
    parser.add_argument('--baseline_tensor', type=str, default=None, help='path to baseline')

    args = parser.parse_args()

    start_time = time.time()

    # # Call the run_year_rmse function with parsed arguments
    # test_tensor_rmse, truth_field, inds_to_rmse, metrics, conf, METS = run_year_rmse(
    #     config=args.config,
    #     input_shape=args.input_shape,
    #     forcing_shape=args.forcing_shape,
    #     output_shape=args.output_shape,
    #     device=args.device,
    #     model_name=args.model_name
    # )
    
    num_cpus = 1
    with mp.Pool(num_cpus) as p:
        run_year_rmse(p, config=args.config, input_shape=args.input_shape,
        forcing_shape=args.forcing_shape,
        output_shape=args.output_shape,
        device=args.device,
        model_name=args.model_name,
        init_noise=args.init_noise,
        save_append=args.save_append, 
        init_tensor=args.init_tensor,
        baseline_tensor=args.baseline_tensor)

    end_time = time.time()
    elapsed_time = end_time - start_time
    # How to run:
    # python IntegratedGradients_zeros.py --config camulator_config.yml --input_shape 1 138 1 192 288 --forcing_shape 1 4 1 192 288 --output_shape 1 145 1 192 288 --device cuda --model_name checkpoint.pt00091.pt
    print(f"Run completed. Results saved to configured location. Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Run completed. Results saved to configured location. Elapsed time: {elapsed_time/60:.2f} minutes")


if __name__ == "__main__":
    main()
