import sys
from collections import Counter
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV, graphical_lasso
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids
import networkx as nx
import community as community_louvain
from scipy.stats import pearsonr, rankdata, norm, zscore, spearmanr
from numpy.linalg import slogdet
import time
import pickle
import logging
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
from networkx.algorithms.community.quality import modularity
from scipy.linalg import solve_discrete_lyapunov, solve_continuous_lyapunov, eig, eigh, eigvals,solve_continuous_are
import math
import jpype as jp
from scipy.signal import hilbert
from sklearn.linear_model import Ridge


BASE_SEED = 42

PROJECT_ROOT = Path(os.environ.get("THESIS_PROJECT_ROOT", Path(__file__).resolve().parents[1],))

BIDS_ROOT = Path(os.environ.get("BIDS_ROOT",PROJECT_ROOT,))

PATIENT_ID = os.environ.get("PATIENT_ID","pt01",)

SCRIPTS_DIR = PROJECT_ROOT / "Scripts"
PLOTS_DIR = PROJECT_ROOT / "Plots"
LOGS_DIR = PROJECT_ROOT / "Logs"
SLURM_DIR = PROJECT_ROOT / "Slurm"
NOTEBOOKS_DIR = PROJECT_ROOT / "Notebooks"
RESULTS_ROOT = PROJECT_ROOT / "Results"

RESULTS_DIR = (RESULTS_ROOT/ PATIENT_ID)
PREPARED_DIR = (RESULTS_DIR/ "Prepared")
BASELINE_DIR = (RESULTS_DIR/ "Baseline")
STIM_DIR = (RESULTS_DIR/ "Stim")
CONTROLLABILITY_DIR = (RESULTS_DIR/ "Controllability")
COMBINED_DIR = (RESULTS_DIR/ "Combined")
ANALYSIS_DIR = (RESULTS_DIR/ "Analysis")

def make_output_dirs():
    """This function creates the patient-specific directories."""
    for d in [RESULTS_DIR,PREPARED_DIR,CONTROLLABILITY_DIR,BASELINE_DIR,STIM_DIR,PLOTS_DIR,]:
        d.mkdir(parents=True, exist_ok=True)

def load_raw_ieeg(sub, ses, task, acq, run, bids_root):
    """This function loads a single BIDS iEEG run, removes bad and miscellaneous channels, and finally applies average referencing.
    
    Parameters
    ----------
    sub
        BIDS subject identifier.
    ses
        BIDS session identifier.
    task
        BIDS task label.
    acq
        BIDS acquisition label.
    run
        BIDS run identifier or simulation run index.
    bids_root
        Root directory of the BIDS dataset.
    
    Returns
    -------
    object
        Tuple containing the preprocessed MNE Raw object and its sampling frequency."""
    bids_path = BIDSPath(subject=sub,session=ses,task=task,acquisition=acq,run=run,suffix="ieeg",extension=".vhdr",root=bids_root,)
    
    raw = read_raw_bids(bids_path=bids_path,extra_params=dict(preload=True),)

    raw.drop_channels(raw.info["bads"])

    raw.drop_channels([ch for ch, typ in zip(raw.ch_names, raw.get_channel_types())if typ == "misc"])

    raw.set_eeg_reference("average", projection=False)

    return raw, raw.info["sfreq"]

def get_patient_channels(sub,ses,task,acq,run,bids_root):
    """This function is used to return the usable channel names for a patient.
    
    Parameters
    ----------
    sub
        BIDS subject identifier.
    ses
        BIDS session identifier.
    task
        BIDS task label.
    acq
        BIDS acquisition label.
    run
        BIDS run identifier or simulation run index.
    bids_root
        Root directory of the BIDS dataset.
    
    Returns
    -------
    object
        List of usable channel names."""
    raw, sfreq = load_raw_ieeg(sub=sub,ses=ses,task=task,acq=acq,run=run,bids_root=bids_root)
    
    return list(raw.ch_names)

import re
import numpy as np
import pandas as pd


def get_onset_offset(events_tsv, verbose=True):
    """
    This function is used to detect seizure onset and offset from the BIDS events.tsv file.
    Detection is based on the priority of the patterns.

    Parameters
    ----------
    events_tsv : str or pathlib.Path
        Path to the BIDS *_events.tsv file.

    verbose : bool
        Print which annotations were selected.

    Returns
    -------
    onset : float
        Seizure onset time in seconds.

    offset : float
        Seizure offset time in seconds.

    Raises
    ------
    ValueError
        If no defensible onset or offset can be identified.
    """

    events = pd.read_csv(events_tsv, sep="\t")

    if "trial_type" not in events.columns:
        raise ValueError(f"No 'trial_type' column found in {events_tsv}")

    if "onset" not in events.columns:
        raise ValueError(f"No 'onset' column found in {events_tsv}")

    events = events.copy()
    events["trial_type_clean"] = (events["trial_type"].fillna("").astype(str).str.strip().str.lower())

    events["onset"] = pd.to_numeric(events["onset"],errors="coerce")

    events = events[np.isfinite(events["onset"])].copy()

    def find_prioritized_match(df, patterns):
        """
        patterns = list of (label, regex)

        Pattern priority is respected first.
        If several events match the same pattern,
        the earliest event is returned.
        """

        for label, pattern in patterns:

            mask = df["trial_type_clean"].str.contains(pattern,regex=True,na=False)

            matches = df.loc[mask]

            if len(matches) > 0:
                row = matches.sort_values("onset").iloc[0]
                return row, label

        return None, None
    onset_patterns = [
        ("explicit seizure onset", r"\b(?:sz|seizure)\s+onset\b"),
        ("explicit seizure start", r"\b(?:sz|seizure)\s+start\b"),
        ("plain onset", r"\bonset\b"),
        ("sz event", r"\bsz\s+event\s*#?"),]
    onset_row, onset_method = find_prioritized_match(events,onset_patterns)

    if onset_row is None:
        annotations = events[["onset", "trial_type"]].to_string(index=False)

        raise ValueError(
            "Could not identify a defensible seizure onset.\n\n"
            f"File: {events_tsv}\n\n"
            f"Available events:\n{annotations}"
        )

    onset = float(onset_row["onset"])
    after_onset = events[events["onset"] > onset].copy()
    offset_patterns = [
        ("explicit seizure offset", r"\b(?:sz|seizure)\s+offset\b"),
        ("explicit seizure end", r"\b(?:sz|seizure)\s+end\b"),
        ("electrographic end", r"\belectrographic\s+end\b"),
        ("event over", r"\bevent\s+over\b"),
        ("OVER", r"^\s*(?:z\s+)?over\s*$"),
        ("definite off", r"\bdefinite\s+off\b"),
        ("seizure off", r"\bseizure\s+off\b"),
        ("sz off", r"\bsz\s+off\b"),
        ("plain offset", r"^\s*offset\s*$"),
        ("stopping", r"\bstopping\b"),
    ]

    offset_row, offset_method = find_prioritized_match(after_onset,offset_patterns)

    if offset_row is None:
        annotations = after_onset[["onset", "trial_type"]].to_string(index=False)

        raise ValueError(
            "Seizure onset was found, but no defensible seizure "
            "offset could be identified.\n\n"
            f"File: {events_tsv}\n"
            f"Detected onset: {onset:.6f} s\n"
            f"Onset annotation: {onset_row['trial_type']}\n\n"
            f"Events after onset:\n{annotations}"
        )

    offset = float(offset_row["onset"])

    if offset <= onset:
        raise ValueError(
            f"Invalid seizure interval: onset={onset}, "
            f"offset={offset}"
        )

    duration = offset - onset

    if verbose:
        print(
            f"Seizure onset:  {onset:.6f} s "
            f"[{onset_method}: {onset_row['trial_type']}]"
        )

        print(
            f"Seizure offset: {offset:.6f} s "
            f"[{offset_method}: {offset_row['trial_type']}]"
        )

        print(
            f"Seizure duration: {duration:.3f} s"
        )

    return onset, offset

def detect_global_seizure_time(R, dt, threshold=0.5, global_fraction=0.5, min_duration=0.05):
    """This function is used to detect global seizure time, or when the required fraction of nodes is active. We do not use it for further analysis
    
    Parameters
    ----------
    R
        Node-by-time radius/amplitude trajectories.
    dt
        Simulation time step in seconds.
    threshold
        Node activation threshold.
    global_fraction
        Fraction of nodes required for global seizure activation.
    min_duration
        Minimum duration for sustained activation, givenn in seconds.
    
    Returns
    -------
    object
        Detected global onset time and the number of active nodes at each time point."""
    N, T = R.shape
    min_samples = int(min_duration / dt)

    active = R > threshold
    n_active = active.sum(axis=0)

    required_nodes = int(np.ceil(global_fraction * N))

    count = 0
    global_time = np.nan

    for t in range(T):
        if n_active[t] >= required_nodes:
            count += 1
        else:
            count = 0

        if count >= min_samples:
            global_time = (t - min_samples + 1) * dt
            break

    return global_time, n_active
    
def _require_torch():
    """This function tries to import pytorch
    
    Returns
    -------
    object
        Imported torch module."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "GPU simulation requires PyTorch. Install a CUDA-enabled PyTorch build "
            "on the cluster, or set SIM_DEVICE=cpu for the PyTorch CPU backend."
        ) from exc
    return torch


def _resolve_torch_device(device=None):
    """This function resolves the requested pytorch device and validates availability of cuda.
    
    Parameters
    ----------
    device
        PyTorch device specification.
    
    Returns
    -------
    object
        Tuple containing the torch module and resolved device."""
    torch = _require_torch()
    requested = device or os.environ.get("SIM_DEVICE", "cuda")
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "SIM_DEVICE requests CUDA, but torch.cuda.is_available() is False. "
            "Request a GPU node and load a CUDA-enabled PyTorch module, or use SIM_DEVICE=cpu."
        )
    return torch, resolved


def _torch_dtype(torch, dtype):
    """This function is used to convert a dtype name to a PyTorch dtype.
    
    Parameters
    ----------
    torch
        Torch.
    dtype
        Floating-point precision for PyTorch computation.
    
    Returns
    -------
    object
        Resolved PyTorch floating-point dtype."""
    name = str(dtype).lower()
    if name in {"float64", "double", "torch.float64"}:
        return torch.float64
    if name in {"float32", "single", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported SIM_DTYPE={dtype!r}; use float32 or float64.")


def _summary_from_switch_times(run, seed, tau_n, global_onset_time, stim_nodes, stim_strength):
    """This function is used to build a per-run summary of node switching and recruitment timing.
    
    Parameters
    ----------
    run
        BIDS run identifier or simulation run index.
    seed
        Seed.
    tau_n
        Tau n, or switching time of our nodes.
    global_onset_time
        Global onset time.
    stim_nodes
        Node indices to stimulate; None uses the function-specific default.
    stim_strength
        Magnitude of the stimulation-induced excitability change.
    
    Returns
    -------
    object
        The summary of the run, including the node switching and recruitment timing"""
    tau_n = np.asarray(tau_n, dtype=float)
    finite_mask = np.isfinite(tau_n)

    if np.any(finite_mask):
        first_onset_time = float(np.min(tau_n[finite_mask]))
        recruitment_times = np.full_like(tau_n, np.nan, dtype=float)
        recruitment_times[finite_mask] = tau_n[finite_mask] - first_onset_time
        switching_order = np.argsort(np.nan_to_num(tau_n, nan=np.inf))
        onset_node = int(switching_order[0])
        last_onset_time = float(np.max(tau_n[finite_mask]))
        total_recruitment_time = last_onset_time - first_onset_time
    else:
        first_onset_time = np.nan
        last_onset_time = np.nan
        total_recruitment_time = np.nan
        recruitment_times = np.full_like(tau_n, np.nan, dtype=float)
        switching_order = np.array([], dtype=int)
        onset_node = None

    return {
        "run": int(run),
        "seed": int(seed),
        "onset_node": onset_node,
        "first_onset_time": first_onset_time,
        "last_onset_time": last_onset_time,
        "total_recruitment_time": total_recruitment_time,
        "tau_n": tau_n,
        "recruitment_times": recruitment_times,
        "global_onset_time": float(global_onset_time) if np.isfinite(global_onset_time) else np.nan,
        "n_switched": int(np.sum(finite_mask)),
        "switching_order": switching_order,
        "switch_times": tau_n,
        "stim_nodes": stim_nodes,
        "stim_strength": float(stim_strength),
    }


def simulate_many_synthetic_seizures_gpu(A,sigma,n_runs=100,chunk_base_seed=42,dt=1e-3,max_search_time=60.0,k=0.1,noise_amp=0.05,stim_nodes=None,
                                         stim_strength=0.0,stim_start=0.0,stim_end=np.inf,node_threshold=0.5,global_fraction=0.3,min_duration=0.05,
                                         batch_size=128,keep_trajectories=0,device=None,dtype="float64",):

    """This is the main function for simulation, that simulates many stochastic synthetic seizures in GPU batches,
      using our bistable oscillator model.
    
    Parameters
    ----------
    A
        connectivity matrix.
    sigma
        excitability per node.
    n_runs
        Number of runs.
    chunk_base_seed
        Chunk base seed.
    dt
        Simulation time step in seconds.
    max_search_time
        Max search time.
    k
        K.
    noise_amp
        Noise amp.
    stim_nodes
        Nodes to stimulate.
    stim_strength
        Magnitude of the stimulation-induced excitability change; 1 is used for all tests.
    stim_start
        Stim start.
    stim_end
        Stim end.
    node_threshold
        Node threshold for switching.
    global_fraction
        Fraction of nodes required for global seizure activation.
    min_duration
        Minimum duration for sustained activation, in seconds.
    batch_size
        Number of simulations processed per GPU batch.
    keep_trajectories
        Keep trajectories.
    device
        PyTorch device specification; if None, use the configured default.
    dtype
        Floating-point precision for PyTorch computation.
    
    Returns
    -------
    object
        Stored example trajectories and a DataFrame with one summary row per simulation run."""
    torch, device = _resolve_torch_device(device)
    torch_dtype = _torch_dtype(torch, dtype)

    A_np = np.asarray(A, dtype=float)
    sigma_np = np.asarray(sigma, dtype=float)
    if A_np.ndim != 2 or A_np.shape[0] != A_np.shape[1]:
        raise ValueError(f"A must be square, got {A_np.shape}.")
    if A_np.shape[0] != sigma_np.size:
        raise ValueError("A and sigma have incompatible node counts.")
    if n_runs <= 0:
        raise ValueError("n_runs must be positive.")
    if dt <= 0 or max_search_time <= 0:
        raise ValueError("dt and max_search_time must be positive.")

    n_nodes = sigma_np.size
    n_steps = int(round(max_search_time / dt))
    min_samples = max(1, int(round(min_duration / dt)))
    required_nodes = int(np.ceil(global_fraction * n_nodes))

    A_t = torch.as_tensor(A_np, dtype=torch_dtype, device=device)
    sigma_t = torch.as_tensor(sigma_np, dtype=torch_dtype, device=device)
    row_sums = A_t.sum(dim=1)

    omega = torch.full((n_nodes,), 2.0 * np.pi * 60.0, dtype=torch_dtype, device=device)
    cos_w = torch.cos(omega * dt)
    sin_w = torch.sin(omega * dt)
    sqrt_dt = float(np.sqrt(dt))

    stim_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    if stim_nodes is not None:
        stim_idx = np.atleast_1d(stim_nodes).astype(int)
        if np.any(stim_idx < 0) or np.any(stim_idx >= n_nodes):
            raise IndexError("stim_nodes contains an invalid node index.")
        stim_mask[torch.as_tensor(stim_idx, device=device)] = True

    summaries = []
    kept_results = []

    for batch_start in range(0, n_runs, batch_size):
        current_batch = min(batch_size, n_runs - batch_start)
        x = torch.zeros((current_batch, n_nodes), dtype=torch_dtype, device=device)
        y = torch.zeros_like(x)

        active_counter = torch.zeros((current_batch, n_nodes), dtype=torch.int32, device=device)
        switched = torch.zeros((current_batch, n_nodes), dtype=torch.bool, device=device)
        switch_times = torch.full((current_batch, n_nodes), torch.nan, dtype=torch_dtype, device=device)
        global_counter = torch.zeros(current_batch, dtype=torch.int32, device=device)
        global_onset = torch.full((current_batch,), torch.nan, dtype=torch_dtype, device=device)

        n_keep_here = max(0, min(keep_trajectories - len(kept_results), current_batch))
        if n_keep_here:
 
            X_keep = torch.empty((n_keep_here, n_nodes, n_steps), dtype=torch_dtype, device=device)
            Y_keep = torch.empty_like(X_keep)
            R_keep = torch.empty_like(X_keep)

        generator = torch.Generator(device=device)
        generator.manual_seed(int(chunk_base_seed + batch_start))

        for step in range(n_steps):
            t = step * dt
            r2 = x.square() + y.square()
            g_vec = 2.0 * r2 - r2.square()

            coupling_x = k * (torch.matmul(x, A_t.T) - x * row_sums)
            coupling_y = k * (torch.matmul(y, A_t.T) - y * row_sums)

            if stim_mask.any() and stim_start <= t <= stim_end:
                effective_sigma = torch.where(
                    stim_mask,
                    torch.full_like(sigma_t, -float(stim_strength)),
                    sigma_t,
                )
            else:
                effective_sigma = sigma_t

            growth = effective_sigma.unsqueeze(0) + g_vec
            x_temp = x + (x * growth + coupling_x) * dt
            y_temp = y + (y * growth + coupling_y) * dt

            if noise_amp > 0:
                x_temp = x_temp + noise_amp * sqrt_dt * torch.randn(
                    x.shape, generator=generator, dtype=torch_dtype, device=device
                )
                y_temp = y_temp + noise_amp * sqrt_dt * torch.randn(
                    y.shape, generator=generator, dtype=torch_dtype, device=device
                )

            x = cos_w * x_temp - sin_w * y_temp
            y = sin_w * x_temp + cos_w * y_temp
            radius = torch.sqrt(x.square() + y.square())

            if n_keep_here:
                X_keep[:, :, step] = x[:n_keep_here]
                Y_keep[:, :, step] = y[:n_keep_here]
                R_keep[:, :, step] = radius[:n_keep_here]

            active = radius > node_threshold
            active_counter = torch.where(active, active_counter + 1, torch.zeros_like(active_counter))
            newly_switched = (~switched) & (active_counter >= min_samples)
            if newly_switched.any():
                switch_times[newly_switched] = (step - min_samples + 1) * dt
                switched |= newly_switched

            n_active = active.sum(dim=1)
            is_global = n_active >= required_nodes
            global_counter = torch.where(is_global, global_counter + 1, torch.zeros_like(global_counter))
            newly_global = torch.isnan(global_onset) & (global_counter >= min_samples)
            if newly_global.any():
                global_onset[newly_global] = (step - min_samples + 1) * dt

        switch_times_np = switch_times.detach().cpu().numpy()
        global_onset_np = global_onset.detach().cpu().numpy()
        if n_keep_here:
            X_keep_np = X_keep.detach().cpu().numpy()
            Y_keep_np = Y_keep.detach().cpu().numpy()
            R_keep_np = R_keep.detach().cpu().numpy()

        for local_idx in range(current_batch):
            run = batch_start + local_idx
            seed = int(chunk_base_seed + run)
            summary = _summary_from_switch_times(
                run=run,
                seed=seed,
                tau_n=switch_times_np[local_idx],
                global_onset_time=global_onset_np[local_idx],
                stim_nodes=stim_nodes,
                stim_strength=stim_strength,
            )
            summaries.append(summary)

            if local_idx < n_keep_here:
                kept_results.append({
                    "X": X_keep_np[local_idx],
                    "Y": Y_keep_np[local_idx],
                    "R": R_keep_np[local_idx],
                    "switch_times": summary["switch_times"],
                    "first_onset_time": summary["first_onset_time"],
                    "global_onset_time": summary["global_onset_time"],
                    "switched": np.isfinite(summary["switch_times"]),
                    "dt": dt,
                    "seed": seed,
                    "stim_nodes": stim_nodes,
                    "stim_strength": stim_strength,
                    "switching_order": summary["switching_order"],
                    "recruitment_times": summary["recruitment_times"],
                })

    return kept_results, pd.DataFrame(summaries)


def run_many_synthetic_seizures(A,sigma,n_runs=100,base_seed=42,chunk_base_seed=None,dt=1e-3,max_search_time=60.0,post_onset_time=None,k=0.1,noise_amp=0.05,
                                n_jobs=None,stim_nodes=None,stim_strength=0.0,stim_start=0.0,stim_end=np.inf,keep_trajectories=0,batch_size=128,device=None,
                                dtype="float64"):
    """This function is used in run_baseline_gpu.py for the batched synthetic seizure simulation.
    
    Parameters
    ----------
    A
        connectivity matrix.
    sigma
        Node excitability.
    n_runs
        Number of runs.
    base_seed
        Base seed.
    chunk_base_seed
        Chunk base seed.
    dt
        Simulation time step in seconds.
    max_search_time
        Max search time.
    post_onset_time
        Post onset time.
    k
        K.
    noise_amp
        Noise amplitude.
    n_jobs
        Parallel-worker setting retained for compatibility or CPU preprocessing.
    stim_nodes
        Nodes to stimulate.
    stim_strength
        Magnitude of the stimulation-induced excitability change. Set to 1 everywhere in the analyses.
    stim_start
        Stim start.
    stim_end
        Stim end.
    keep_trajectories
        Whether to keep trajectories.
    batch_size
        Number of simulations processed per GPU batch.
    device
        PyTorch device specification; if None, use the default.
    dtype
        Floating-point precision for PyTorch computation.
    
    Returns
    -------
    object
        Stored example trajectories and per-run summary DataFrame."""
    del post_onset_time, n_jobs
    if chunk_base_seed is None:
        chunk_base_seed = base_seed
    return simulate_many_synthetic_seizures_gpu(A=A,sigma=sigma,n_runs=n_runs,chunk_base_seed=chunk_base_seed,dt=dt,max_search_time=max_search_time,k=k,
                                                noise_amp=noise_amp,stim_nodes=stim_nodes,stim_strength=stim_strength,stim_start=stim_start,stim_end=stim_end,
                                                batch_size=batch_size,keep_trajectories=keep_trajectories,device=device,dtype=dtype,)


def run_stimulation(A,sigma,stim_strength=1.0,n_runs_per_node=100,chunk_base_seed=42,dt=1e-3,max_search_time=60.0,post_onset_time=None,k=0.1,noise_amp=0.05,
                    n_jobs=None,stim_start=0.0,stim_end=np.inf,stim_nodes=None,batch_size=128,device=None,dtype="float64",):

    """This function is used in run_stim_regime_gpu.py for the batched synthetic seizure simulation with single node stimulation.
    
    Parameters
    ----------
    A
        Connectivity matrix.
    sigma
        Node excitability.
    stim_strength
        Magnitude of the stimulation; 1 for all analyses.
    n_runs_per_node
        N runs per node.
    chunk_base_seed
        Chunk base seed.
    dt
        Simulation time step in seconds.
    max_search_time
        Max search time.
    post_onset_time
        Post onset time.
    k
        K.
    noise_amp
        Noise amplitude.
    n_jobs
        Parallel-worker setting retained for compatibility or CPU preprocessing.
    stim_start
        Stimulation start.
    stim_end
        Stimulationi end.
    stim_nodes
        Node indices to stimulate; None uses the function-specific default.
    batch_size
        Number of simulations processed per GPU batch.
    device
        PyTorch device specification; if None, use the default.
    dtype
        Floating-point precision for PyTorch computation.
    
    Returns
    -------
    object
        Concatenated stimulation results for all stimulated nodes."""
    del post_onset_time, n_jobs
    n_nodes = len(sigma)
    if stim_nodes is None:
        stim_nodes = list(range(n_nodes))

    all_summaries = []
    for stim_node in stim_nodes:
        _, df = simulate_many_synthetic_seizures_gpu(A=A,sigma=sigma,n_runs=n_runs_per_node,chunk_base_seed=chunk_base_seed,dt=dt,
                                                     max_search_time=max_search_time,k=k,noise_amp=noise_amp,stim_nodes=[int(stim_node)],
                                                     stim_strength=stim_strength,stim_start=stim_start,stim_end=stim_end,batch_size=batch_size,
                                                     keep_trajectories=0,device=device,dtype=dtype,)
        df["stim_node"] = int(stim_node)
        all_summaries.append(df)

    return pd.concat(all_summaries, ignore_index=True)


def summarize_stimulation_screen(stim_df):
    """This function summarizes the node-wise stimulation simulations for the node-wise results.
    
    Parameters
    ----------
    stim_df
        Stimulation dataframe.
    
    Returns
    -------
    object
        Node-wise stimulation summary sorted by the screening criteria."""
    stim_summary = (
        stim_df
        .groupby("stim_node")
        .agg(
            onset_rate=("first_onset_time", lambda x: x.notna().mean()),
            global_rate=("global_onset_time", lambda x: x.notna().mean()),

            mean_n_switched=("n_switched", "mean"),
            median_n_switched=("n_switched", "median"),

            mean_first_onset_time=("first_onset_time", "mean"),
            mean_last_onset_time=("last_onset_time", "mean"),

            mean_total_recruitment_time=("total_recruitment_time", "mean"),
            median_total_recruitment_time=("total_recruitment_time", "median"),

            mean_global_time=("global_onset_time", "mean"),
            median_global_time=("global_onset_time", "median"),
            std_global_time=("global_onset_time", "std"),

            n_global_onsets=("global_onset_time", lambda x: x.notna().sum()),
            n_runs=("run", "count"),
        )
        .reset_index()
    )

    stim_summary = stim_summary.sort_values(
        ["global_rate", "mean_n_switched", "mean_global_time"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    stim_summary["rank"] = np.arange(1, len(stim_summary) + 1)

    return stim_summary


def plot_node_trajectories(result,signal="R",nodes=None,use_abs=False,max_nodes=25,save_path=None,dpi=600,):
    """This function can plot selected node trajectories from a stored synthetic seizure simulation. Used for visualization.
    
    Parameters
    ----------
    result
        Result.
    signal
        Signal.
    nodes
        Nodes.
    use_abs
        Use abs.
    max_nodes
        Max nodes.
    save_path
        Optional path at which to save the generated figure.
    dpi
        Resolution used when saving a figure.
    
    Returns
    -------
    object
        Plotted node trajectories"""
    data = result[signal]

    N, T = data.shape
    t = np.arange(T) * result["dt"]

    if nodes is None:
        nodes = list(range(min(max_nodes, N)))

    plt.figure(figsize=(12, 5))

    for n in nodes:
        y = np.abs(data[n]) if use_abs else data[n]
        plt.plot(t, y, label=f"node {n}")

    plt.xlabel("Time (s)")
    plt.ylabel(signal)
    plt.title(f"{signal} trajectories")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    plt.show()


def plot_adjacency_matrix(A, title="A matrix"):
    """This function is used to plot a adjacency or effective-connectivity matrix. Mostly used for sanity_checks.
    
    Parameters
    ----------
    A
        Connectivity matrix.
    title
        Title.
    
    Returns
    -------
    object
        The plotted matrix"""
    plt.figure(figsize=(6, 5))
    plt.imshow(A, cmap="viridis", aspect="auto")
    plt.colorbar(label="weight")
    plt.title(title)
    plt.xlabel("source")
    plt.ylabel("targe")
    plt.tight_layout()
    plt.show()

def plot_mean_controllability_by_channel(ctrl_ranking,metric="mean_controllability",channel_col="channel",title="Mean controllability per channel",
                                         ylabel="Mean controllability",save_path=None,dpi=600,figsize=None,):
    """This function plots and returns the channel-wise controllability values, ordered from high to low.
    
    Parameters
    ----------
    ctrl_ranking
        Controllability ranking.
    metric
        Metric.
    channel_col
        Channel col.
    title
        Title.
    ylabel
        Ylabel.
    save_path
        Optional path at which to save the generated figure.
    dpi
        Resolution used when saving a figure.
    figsize
        Figsize.
    
    Returns
    -------
    object
        Mean controllability"""
    df = ctrl_ranking.copy()
    df = df[[channel_col, metric]].dropna()
    df = df.sort_values(metric, ascending=False).reset_index(drop=True)

    if figsize is None:
        figsize = (max(10, 0.35 * len(df)), 5)

    plt.figure(figsize=figsize)

    x = np.arange(len(df))
    y = df[metric].values

    plt.bar(x, y)

    plt.xticks(x,df[channel_col],rotation=90,fontsize=8,)

    plt.ylabel(ylabel)
    plt.xlabel("Channel")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    plt.show()

    return df


def plot_controllability_soz_violin(ctrl_ranking,metric="mean_controllability",channel_col="channel",soz_col="is_soz",title=None,ylabel=None,save_path=None,dpi=600,):
    """This function compares SOZ and non-SOZ controllability values with channel labels; based on the paper by Ding et al.
    
    Parameters
    ----------
    ctrl_ranking
        Controllability ranking.
    metric
        Metric.
    channel_col
        Channel colom.
    soz_col
        Soz colom.
    title
        Title.
    ylabel
        Ylabel.
    save_path
        Optional path at which to save the generated figure.
    dpi
        Resolution used when saving a figure.
    
    Returns
    -------
    object
        Violin plot of SOZ vs N-SOZ controllability values"""
    df = ctrl_ranking.copy()

    df = df[[channel_col, soz_col, metric]].dropna()
    df[soz_col] = df[soz_col].astype(bool)

    soz_vals = df.loc[df[soz_col], metric].values
    nsoz_vals = df.loc[~df[soz_col], metric].values

    data = [nsoz_vals, soz_vals]
    labels = ["NSOZ", "SOZ"]
    positions = [0, 1]

    plt.figure(figsize=(7, 6))

    violin = plt.violinplot(data,positions=positions,showmeans=True,showmedians=True,showextrema=True,)

    rng = np.random.default_rng(0)

    for group_idx, is_soz_group in enumerate([False, True]):
        df_group = df[df[soz_col] == is_soz_group].copy()

        x_jitter = (np.full(len(df_group), positions[group_idx])+ rng.normal(0, 0.035, size=len(df_group)))

        y = df_group[metric].values

        plt.scatter(x_jitter,y,s=35,alpha=0.8,edgecolor="black",linewidth=0.5,zorder=3,)

        for x_i, (_, row) in zip(x_jitter, df_group.iterrows()):
            plt.text(x_i + 0.025,row[metric],str(row[channel_col]),fontsize=8,va="center",)

    plt.xticks(positions, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    plt.show()

def association_h2(target_signal, source_signal, n_bins=100):

    """This function computes the h² association from a source signal to a target signal.
    
    Parameters
    ----------
    target_signal
        Target signal.
    source_signal
        Source signal.
    n_bins
        Number of bins used for h² association estimation.
    
    Returns
    -------
    object
        The directed h² association coefficient clipped to [0, 1]."""
    x = np.asarray(source_signal, dtype=float)
    y = np.asarray(target_signal, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < n_bins or np.var(y) <= 0:
        return 0.0

    order = np.argsort(x)
    y_sorted = y[order]

    bins = np.array_split(np.arange(len(y_sorted)), n_bins)

    numerator = 0.0

    for idx in bins:
        if len(idx) == 0:
            continue

        y_bin = y_sorted[idx]
        y_bin_mean = np.mean(y_bin)

        numerator += np.sum((y_bin - y_bin_mean) ** 2)

    h2 = 1.0 - numerator / (len(y) * np.var(y))

    return float(np.clip(h2, 0.0, 1.0))


def compute_association_matrix(data_window, n_bins=100):
    """This function is used to compute the full h² association matrix for a multichannel data window.
    
    Parameters
    ----------
    data_window
        Data window.
    n_bins
        Number of bins used for h² association estimation.
    
    Returns
    -------
    object
        Directed channel-by-channel h² association matrix."""
    data_window = np.asarray(data_window, dtype=float)

    n_samples, n_channels = data_window.shape

    A = np.zeros((n_channels, n_channels), dtype=float)

    for target in range(n_channels):
        for source in range(n_channels):
            if target == source:
                continue

            A[target, source] = association_h2(target_signal=data_window[:, target],source_signal=data_window[:, source],n_bins=n_bins)
    A[~np.isfinite(A)] = 0.0
    A[A < 0] = 0.0
    np.fill_diagonal(A, 0.0)

    return A

def compute_static_association_A_from_seizure(raw,orig_sfreq,onset_sample,offset_sample,tmin=-5,tmax=10,l_freq=30.0,h_freq=90.0,notch_freq=60,n_bins=100,):
    """This function uses the last functions to construct a h² association matrix over the seizure interval. Not used in analysis
    
    Parameters
    ----------
    raw
        Loaded MNE Raw iEEG recording.
    orig_sfreq
        Original sampling frequency in Hz.
    onset_sample
        Onset sample.
    offset_sample
        Offset sample.
    tmin
        Tmin.
    tmax
        Tmax.
    l_freq
        L freq.
    h_freq
        H freq.
    notch_freq
        Notch freq.
    n_bins
        Number of bins used for h² association estimation.
    
    Returns
    -------
    object
        The h2 association matrix for the seizure"""
    onset = onset_sample / orig_sfreq
    offset = offset_sample / orig_sfreq
    
    start_time = onset
    stop_time = offset 

    if start_time < raw.times[0] or stop_time > raw.times[-1]:
        raise ValueError("Seizure epoch outside recording bounds")

    if stop_time <= start_time:
        raise ValueError("Offset must be after onset")

    raw_filt = raw.copy()

    if notch_freq is not None:
        raw_filt.notch_filter(notch_freq,notch_widths=2.0,method="spectrum_fit",verbose=False)

    raw_filt.filter(l_freq,h_freq,method="iir",iir_params=dict(order=4, ftype="butter"),verbose=False)

    epoch = raw_filt.copy().crop(start_time, stop_time)
    data = epoch.get_data()  # channels x samples

    if np.isnan(data).any():
        raise ValueError("NaN values detected in epoch")

    X = data.T  # samples x channels

    A = compute_association_matrix(X, n_bins=n_bins)

    return A, epoch, data

def plot_A_matrix(A, ch_names=None, title="Static association matrix A"):
    """This function lots an h² association matrix.
    
    Parameters
    ----------
    A
        Connectivity matrix.
    ch_names
        Channel names in matrix/node order.
    title
        Title.
    
    Returns
    -------
    object
        Plot of the A matrix"""
    plt.figure(figsize=(7, 6))
    im = plt.imshow(A, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, label="h² association")

    plt.xlabel("source node m")
    plt.ylabel("target node n")
    plt.title(title)

    if ch_names is not None:
        plt.xticks(range(len(ch_names)), ch_names, rotation=90)
        plt.yticks(range(len(ch_names)), ch_names)

    plt.tight_layout()
    plt.show()

def normalize_A_spectral(A, beta=0.95):
    """This function normalizes an adjacency matrix by its spectral radius and scales it by beta to ensure stability.
    
    Parameters
    ----------
    A
        Connectivity matrix.
    beta
        Scaling parameter.
    
    Returns
    -------
    object
        The Normalized matrix."""
    A = np.asarray(A, dtype=float).copy()

    A[~np.isfinite(A)] = 0.0
    A[A < 0] = 0.0
    np.fill_diagonal(A, 0.0)

    eigvals = np.linalg.eigvals(A)
    spectral_radius = np.max(np.abs(eigvals))

    if spectral_radius > 0:
        A = A / spectral_radius

    A = beta * A

    return A


def plot_adjacency_matrix(A, ch_names=None, title="A matrix"):
    """This plots ana effective-connectivity matrix.
    
    Parameters
    ----------
    A
        Connectivity matrix.
    ch_names
        Channel names in matrix/node order.
    title
        Title.
    
    Returns
    -------
    Object
            The plot of the a matrix"""
    plt.figure(figsize=(7, 6))
    plt.imshow(A, cmap="viridis", aspect="auto")
    plt.colorbar(label="weight")

    plt.title(title)
    plt.xlabel("source")
    plt.ylabel("target")

    if ch_names is not None:
        plt.xticks(range(len(ch_names)), ch_names, rotation=90)
        plt.yticks(range(len(ch_names)), ch_names)

    plt.tight_layout()
    plt.show()


def diagnose_A(A, ch_names=None, low_input_threshold=0.05):
    """This function is a sanity-test that prints basic connectivity diagnostics and visualizes an adjacency matrix.
    
    Parameters
    ----------
    A
        Connectivity matrix.
    ch_names
        Channel names in matrix/node order.
    low_input_threshold
        Low input threshold.
    
    Returns
    -------
    object
        The A matrix diagnostics"""
    A = np.asarray(A)
    #ingoing
    in_strength = A.sum(axis=1)     
    #outgoing
    out_strength = A.sum(axis=0)   

    print("A shape:", A.shape)
    print("A min/max:", np.min(A), np.max(A))
    print("density:", np.mean(A > 0))
    print("mean weight:", np.mean(A[A > 0]) if np.any(A > 0) else 0.0)

    print("\nInput strength")
    print("min input:", in_strength.min())
    print("mean input:", in_strength.mean())
    print("max input:", in_strength.max())

    low_input_nodes = np.where(in_strength < low_input_threshold)[0]
    print("nodes with really low input:", low_input_nodes)

    if ch_names is not None and len(low_input_nodes) > 0:
        print("low input channels:", [ch_names[i] for i in low_input_nodes])

    print("\nOutput strength")
    print("min output:", out_strength.min())
    print("mean output:", out_strength.mean())
    print("max output:", out_strength.max())

    plot_adjacency_matrix(A)


def threshold_A_density(A, target_density=0.2, min_in_edges=3):
    """This function reduces the density, while keeping a target edge density and minimum incoming support.
    
    Parameters
    ----------
    A
        Connectivity matrix.
    target_density
        Target density.
    min_in_edges
        Minimum incoming edges.
    
    Returns
    -------
    object
        The thresholded a matrix."""
    A = np.asarray(A, dtype=float).copy()
    np.fill_diagonal(A, 0.0)

    A_new = np.zeros_like(A)
    N = A.shape[0]

    # keeps strongest edges
    vals = A[A > 0]
    cutoff = np.quantile(vals, 1 - target_density)
    A_new[A >= cutoff] = A[A >= cutoff]

    # guarantees enough incoming edges
    for target in range(N):
        incoming = A[target, :].copy()
        incoming[target] = 0.0

        strongest_sources = np.argsort(incoming)[::-1][:min_in_edges]

        for source in strongest_sources:
            if incoming[source] > 0:
                A_new[target, source] = A[target, source]

    np.fill_diagonal(A_new, 0.0)
    return A_new

def compute_energy_based_sigma(data,sfreq,window_size=1.0,overlap=0.5,sigma_range=(-0.20, -0.10),):
    """This function maps the channel-wise signal energy to the sigma excitability values; based on Creaser paper.
    
    Parameters
    ----------
    data
        Data.
    sfreq
        Sampling frequency in Hz.
    window_size
        Window size.
    overlap
        Overlap.
    sigma_range
        Sigma range.
    
    Returns
    -------
    object
        Tuple of sigma values, channel energies, and normalized energies."""
    n_channels, n_samples = data.shape
    win_samp = int(window_size * sfreq)
    step_samp = int(win_samp * (1 - overlap))

    energies = np.zeros(n_channels)

    for start in range(0, n_samples - win_samp + 1, step_samp):
        stop = start + win_samp
        window = data[:, start:stop]
        energies += np.sum(window ** 2, axis=1)

    e_min = np.min(energies)
    e_max = np.max(energies)

    if e_max > e_min:
        energy_norm = (energies - e_min) / (e_max - e_min)
    else:
        energy_norm = np.full(n_channels, 0.5)

    sigma_min, sigma_max = sigma_range

    sigma = sigma_min + energy_norm * (sigma_max - sigma_min)

    return sigma, energies, energy_norm

def compare_controllability_to_stim(stim_summary,ctrl_summary,ctrl_metric,ctrl_label,title_prefix,stim_metric="reduction_n_switched",
                                    stim_node_col="stim_node",ctrl_node_col="node_idx",figsize=(6, 6),save_path=None,dpi=600,):
    """This function compares controllability and stimulation rankings using Spearman rank correlation.
    
    Parameters
    ----------
    stim_summary
        Stim summary.
    ctrl_summary
        Ctrl summary.
    ctrl_metric
        Ctrl metric.
    ctrl_label
        Ctrl label.
    title_prefix
        Title prefix.
    stim_metric
        Stim metric.
    stim_node_col
        Stim node col.
    ctrl_node_col
        Ctrl node col.
    figsize
        Figsize.
    save_path
        Optional path at which to save the generated figure.
    dpi
        Resolution.
    
    Returns
    -------
    object
        The comparison between controllability and stimulation"""
    comparison = stim_summary.merge(ctrl_summary[[ctrl_node_col, "channel", ctrl_metric]],left_on=stim_node_col,right_on=ctrl_node_col,
                                    how="left",suffixes=("_stim", "_ctrl"),)

    comparison["stim_rank"] = (comparison[stim_metric].rank(ascending=False, method="min"))

    comparison["ctrl_rank"] = (comparison[ctrl_metric].rank(ascending=False, method="min"))

    valid = comparison[["stim_rank", "ctrl_rank"]].notna().all(axis=1)

    if valid.sum() >= 2:
        rho, p = spearmanr(comparison.loc[valid, "stim_rank"],comparison.loc[valid, "ctrl_rank"],)
    else:
        rho, p = np.nan, np.nan

    print(f"\n{title_prefix}: {ctrl_label}")
    print("spearman rho:", rho)
    print("p value:", p)

    comparison["spearman_rho"] = rho
    comparison["spearman_p"] = p
    comparison["ctrl_metric"] = ctrl_metric
    comparison["ctrl_label"] = ctrl_label

    if "channel_stim" in comparison.columns:
        comparison["channel_plot"] = comparison["channel_stim"]
    elif "channel_x" in comparison.columns:
        comparison["channel_plot"] = comparison["channel_x"]
    elif "channel_ctrl" in comparison.columns:
        comparison["channel_plot"] = comparison["channel_ctrl"]
    elif "channel_y" in comparison.columns:
        comparison["channel_plot"] = comparison["channel_y"]
    else:
        comparison["channel_plot"] = comparison["channel"]

    plt.figure(figsize=figsize)

    x = comparison.loc[valid, "ctrl_rank"]
    y = comparison.loc[valid, "stim_rank"]

    plt.scatter(x, y)

    if valid.sum() >= 2:
        m, b = np.polyfit(x, y, 1)
        plt.plot(x,m * x + b,linewidth=2,label=f"fit (slope={m:.2f})",)

    for _, row in comparison.loc[valid].iterrows():
        plt.text(row["ctrl_rank"],row["stim_rank"],row["channel_plot"],fontsize=8,)

    plt.xlabel(f"{ctrl_label} rank")
    plt.ylabel("Synthetic stimulation rank")
    plt.title(f"{title_prefix}: {ctrl_label} vs stimulation\n"f"Spearman rho={rho:.2f}, p={p:.3g}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    plt.show()

    return comparison

def compute_network_similarity_h2(conn_mats, use_absolute=False, zero_negative=True):
    """This function computes network similarity from correlations between h² matrices for community detection.
    
    Parameters
    ----------
    conn_mats
        Conn mats.
    use_absolute
        Use absolute.
    zero_negative
        Zero negative.
    
    Returns
    -------
    object
        The  network similarity matrix."""
    W, N, _ = conn_mats.shape

    offdiag_idx = ~np.eye(N, dtype=bool)

    vectors = np.array([conn_mats[w][offdiag_idx]for w in range(W)])

    S = np.corrcoef(vectors)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

    if use_absolute:
        S = np.abs(S)

    if zero_negative:
        S[S < 0] = 0.0

    np.fill_diagonal(S, 1.0)

    return S

def apply_temporal_weighting(S, beta=0.01):

    """This function is used to apply temporal weighting between temporally distant windows; based on Scheid paper.
    
    Parameters
    ----------
    S
        Similarity matrix.
    beta
        Temporal-weighting parameter.
    
    Returns
    -------
    object
        The temporally weighted similarity matrix."""
    W = S.shape[0]
    weighted_S = S.copy()

    for i in range(W):
        for j in range(W):
            weight = (1 - beta * (abs(i - j) / W)) ** 2
            weighted_S[i, j] *= weight

    weighted_S = np.nan_to_num(weighted_S, nan=0.0, posinf=0.0, neginf=0.0)
    weighted_S[weighted_S < 0] = 0.0

    np.fill_diagonal(weighted_S, 1.0)

    return weighted_S

def run_louvain_multiple(S_weighted, resolution=1.0, n_runs=100, seed=0):
    """This function runs Louvain community detection repeatedly and retains the partition with highest modularity.
    
    Parameters
    ----------
    S_weighted
        weighted similarity matrix.
    resolution
        Resolution.
    n_runs
        Number of runs.
    seed
        Seed.
    
    Returns
    -------
    object
        The best community labels and the corresponding modularity value."""
    S = S_weighted.copy()
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    S[S < 0] = 0.0

    np.fill_diagonal(S, 0.0)

    G = nx.from_numpy_array(S)

    best_Q = -np.inf
    best_labels = None

    for r in range(n_runs):
        partition = community_louvain.best_partition(G,weight="weight",resolution=resolution,random_state=seed + r,)
        labels = np.array([partition[i]for i in range(len(partition))])
        communities = {}

        for node, comm in partition.items():
            communities.setdefault(comm, []).append(node)

        Q = nx.algorithms.community.quality.modularity(G,list(communities.values()),weight="weight",resolution=resolution)

        if Q > best_Q:
            best_Q = Q
            best_labels = labels

    return best_labels, best_Q

def find_three_communities(S_weighted,seed=0,gamma_range=np.arange(0.8, 1.05, 0.05),n_runs=100,verbose=True,prefer_three=True,):

    """This function loops through Louvain resolution values and tries to find the best partition containing three communities.
    
    Parameters
    ----------
    S_weighted
        Weighted similarity matrix.
    seed
        Seed.
    gamma_range
        Gamma range.
    n_runs
        Number of runs.
    verbose
        Verbose.
    prefer_three
        Prefer three.
    
    Returns
    -------
    object
       The Selected labels, resolution parameter, and modularity value for preferably 3 communities."""
    best_three_labels = None
    best_three_gamma = None
    best_three_Q = -np.inf

    best_any_labels = None
    best_any_gamma = None
    best_any_Q = -np.inf

    for gamma in gamma_range:
        labels, Q = run_louvain_multiple(S_weighted,resolution=gamma,n_runs=n_runs,seed=seed)

        labels = relabel_communities_by_time(labels)

        n_comm = len(np.unique(labels))

        if verbose:
            print(f"gamma={gamma:.2f}, communities={n_comm}, Q={Q:.4f}")

        if Q > best_any_Q:
            best_any_labels = labels
            best_any_gamma = gamma
            best_any_Q = Q

        if n_comm == 3 and Q > best_three_Q:
            best_three_labels = labels
            best_three_gamma = gamma
            best_three_Q = Q

    if prefer_three and best_three_labels is not None:
        return best_three_labels, best_three_gamma, best_three_Q

    if prefer_three:
        print("No solution with 3 communities found, returning best partition")

    return best_any_labels, best_any_gamma, best_any_Q

def relabel_communities_by_time(labels):
    """This function relabels the communities in chronological order by their mean window positions.
    
    Parameters
    ----------
    labels
        Labels.
    
    Returns
    -------
    object
        Chronologically relabeled community labels."""
    unique = np.unique(labels)
    mean_positions = {c: np.mean(np.where(labels == c)[0])for c in unique}

    ordered = sorted(unique, key=lambda c: mean_positions[c])

    mapping = {old: new for new, old in enumerate(ordered)}

    return np.array([mapping[x] for x in labels])


def plot_regimes_h2(window_times,regime_labels,title="Ictal h² communities",onset_time=0.0,result=None,):
    """This function can plot detected ictal regimes over time. Used for visualization.
    
    Parameters
    ----------
    window_times
        Window times.
    regime_labels
        Regime labels.
    title
        Title.
    onset_time
        Onset time.
    result
        Result.
    
    Returns
    -------
    object
        Plot of the h2 regimes."""
    plt.figure(figsize=(11, 3))

    plt.scatter(window_times,regime_labels,c=regime_labels,s=35,)

    plt.axvline(onset_time,linestyle="--",label="seizure onset",)

    if result is not None:
        if "first_onset_time" in result and result["first_onset_time"] is not None:
            plt.axvline(result["first_onset_time"],linestyle="--",label="first synthetic onset",)

        if "global_onset_time" in result and not np.isnan(result["global_onset_time"]):
            plt.axvline(result["global_onset_time"],linestyle=":",label="global synthetic onset",)

    plt.xlabel("Time from seizure onset (s)")
    plt.ylabel("Community / regime")
    plt.title(title)

    unique_labels = np.unique(regime_labels)

    if len(unique_labels) == 3:
        plt.yticks([0, 1, 2],["early ictal", "middle ictal", "late ictal"],)
    else:
        plt.yticks(unique_labels)

    plt.legend()
    plt.grid(True)
    plt.show()

def make_regimes_contiguous_by_majority(labels):

    """This function is used to convert temporally interweaving community labels into contiguous chronological regimes;
    Different method than Scheid is used, see the text.
    
    Parameters
    ----------
    labels
        Labels.
    
    Returns
    -------
    object
        The contiguous labels."""
    labels = np.asarray(labels)
    W = len(labels)

    unique_labels = np.unique(labels)
    n_regimes = len(unique_labels)

    if n_regimes == 1:
        return np.zeros(W, dtype=int), []

    # Temporal center of each  community
    centers = []
    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        centers.append((lab, np.median(idx)))

    centers = sorted(centers, key=lambda x: x[1])

    ordered_centers = np.array([c for _, c in centers])

    # Boundaries in the middle
    boundaries = []
    for i in range(n_regimes - 1):
        b = int(np.round((ordered_centers[i] + ordered_centers[i + 1]) / 2))
        b = max(1, min(W - 1, b))
        boundaries.append(b)

    boundaries = sorted(list(set(boundaries)))
    if len(boundaries) != n_regimes - 1:
        boundaries = [int(np.round(W * k / n_regimes))for k in range(1, n_regimes)]

    contiguous = np.zeros(W, dtype=int)

    start = 0
    for reg, end in enumerate(boundaries):
        contiguous[start:end] = reg
        start = end

    contiguous[start:] = len(boundaries)
    return contiguous, boundaries

def get_regime_representative_A(A_seq):
    """This function computes the mean connectivity matrix for a regime.
    
    Parameters
    ----------
    A_seq
        A seq.
    
    Returns
    -------
    object
        The Mean A matrix for regime and also the index of  the central window."""
        
    A_rep = np.nanmean(A_seq, axis=0)
    rep_window_idx = A_seq.shape[0] // 2
    A_rep = np.nan_to_num(A_rep, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(A_rep, 0.0)
    return A_rep, rep_window_idx


def detect_h2_ictal_regimes_one_seizure(A_ictal,ictal_times,beta=0.01,gamma_range=np.arange(0.8, 1.05, 0.05),n_runs=100,seed=0,prefer_three=True,
                                        use_absolute=False,zero_negative=True,verbose=True,):

    """This function combines the last functions to detect the temporally weighted Louvain regimes for the h² networks of one seizure.
    
    Parameters
    ----------
    A_ictal
        A ictal.
    ictal_times
        Ictal times.
    beta
        Temporal-weighting parameter.
    gamma_range
        Gamma range.
    n_runs
        Number of runs.
    seed
        Seed.
    prefer_three
        Prefer three.
    use_absolute
        Use absolute.
    zero_negative
        Zero negative.
    verbose
        Verbose.
    
    Returns
    -------
    object
        A Dictionary containing  the labels, similarity matrices, gamma, modularity, and timing metadata."""
    if A_ictal.shape[0] < 3:
        raise ValueError("Need at least 3 ictal windows for regime detection.")

    S = compute_network_similarity_h2(A_ictal,use_absolute=use_absolute,zero_negative=zero_negative,)

    S_weighted = apply_temporal_weighting(S,beta=beta)

    labels, gamma, Q = find_three_communities(S_weighted,seed=seed,gamma_range=gamma_range,n_runs=n_runs,verbose=verbose,prefer_three=prefer_three,)

    labels = relabel_communities_by_time(labels)
    
    labels_raw = labels.copy()
    
    labels_contiguous, boundaries = make_regimes_contiguous_by_majority(labels_raw)

    regime_result = {
        "labels": labels,
        "gamma": gamma,
        "Q": Q,
        "S": S,
        "S_weighted": S_weighted,
        "ictal_times": ictal_times,
        "beta": beta,
        "n_runs": n_runs,
        "prefer_three": prefer_three,
    }
    return regime_result

def detect_h2_ictal_regimes_all(all_ec_ictal_h2,all_ictal_times_h2,beta=0.01,gamma_range=np.arange(0.8, 1.05, 0.05),n_runs=100,seed=0,prefer_three=True,
                                verbose=True,make_contiguous=True,):
    """This function detects the h² ictal regimes for all seizures and summarizes their timing and modularity.
    
    Parameters
    ----------
    all_ec_ictal_h2
        All ec ictal h2.
    all_ictal_times_h2
        All ictal times h2.
    beta
        Temporal-weighting parameter.
    gamma_range
        Gamma range.
    n_runs
        Number of runs.
    seed
        Seed.
    prefer_three
        Prefer three.
    verbose
        Verbose.
    make_contiguous
        Make contiguous.
    
    Returns
    -------
    object
        Per-seizure regime result dictionaries and a regime summary DataFrame."""
    all_results = []
    rows = []

    for sz_idx, (A_ictal, ictal_times) in enumerate(zip(all_ec_ictal_h2, all_ictal_times_h2)):
        print(f"\n=== Seizure {sz_idx + 1} ===")
        print("A_ictal shape:", A_ictal.shape)

        if A_ictal.shape[0] < 3:
            print("Skipping: fewer than 3 ictal windows")
            continue

        result = detect_h2_ictal_regimes_one_seizure(A_ictal=A_ictal,ictal_times=ictal_times,beta=beta,gamma_range=gamma_range,n_runs=n_runs,
                                                     seed=seed + 1000 * sz_idx,prefer_three=prefer_three,verbose=verbose,)

        labels_raw = result["labels"].copy()

        if make_contiguous:
            labels_contiguous, boundaries = make_regimes_contiguous_by_majority(labels_raw)
            labels_used = labels_contiguous
        else:
            labels_contiguous = labels_raw.copy()
            boundaries = []
            labels_used = labels_raw

        result["labels_raw"] = labels_raw
        result["labels_contiguous"] = labels_contiguous
        result["labels"] = labels_used
        result["contiguous_boundaries"] = boundaries
        result["make_contiguous"] = make_contiguous

        print("chosen gamma:", result["gamma"])
        print("modularity Q:", result["Q"])
        print("number of raw communities:", len(np.unique(labels_raw)))
        print("number of used regimes:", len(np.unique(labels_used)))

        if make_contiguous:
            print("contiguous boundaries:", boundaries)

        for lab in np.unique(labels_used):
            mask = labels_used == lab

            rows.append({
                "seizure_idx": sz_idx,
                "regime": int(lab),
                "n_windows": int(mask.sum()),
                "start_time": float(ictal_times[mask][0]),
                "end_time": float(ictal_times[mask][-1]),
                "gamma": float(result["gamma"]),
                "Q": float(result["Q"]),
                "beta": float(beta),
                "n_raw_communities": int(len(np.unique(labels_raw))),
                "n_used_regimes": int(len(np.unique(labels_used))),
                "make_contiguous": bool(make_contiguous),
            })

            print(f"  regime {lab}: "f"{mask.sum()} windows, "f"{ictal_times[mask][0]:.1f}–{ictal_times[mask][-1]:.1f}s")

        all_results.append(result)

    summary = pd.DataFrame(rows)

    return all_results, summary

def average_patient_seizures(association_A_all, preprocessed_epochs_assoc):
    """This function averages association matrices and preprocessed signals across a patient’s seizures.
    
    Parameters
    ----------
    association_A_all
        Association a all.
    preprocessed_epochs_assoc
        Preprocessed epochs assoc.
    
    Returns
    -------
    object
        The info averaged over seizures"""
    if len(association_A_all) != len(preprocessed_epochs_assoc):
        raise ValueError("A list and epoch list must have same length")

    ch_ref = association_A_all[0]["ch_names"]

    for i, item in enumerate(association_A_all):
        if item["ch_names"] != ch_ref:
            raise ValueError(f"Channel mismatch in A matrix {i}")

    for i, epoch in enumerate(preprocessed_epochs_assoc):
        if epoch.ch_names != ch_ref:
            raise ValueError(f"Channel mismatch in epoch {i}")
    A_stack = np.stack([item["A"] for item in association_A_all], axis=0)
    A_mean = np.nanmean(A_stack, axis=0)
    A_std = np.nanstd(A_stack, axis=0)
    np.fill_diagonal(A_mean, 0.0)

    data_stack = np.stack([ep.get_data() for ep in preprocessed_epochs_assoc], axis=0)
    data_mean = np.nanmean(data_stack, axis=0)

    avg_info = {
        "run": "average",
        "seizure_idx": "average",
        "A": A_mean,
        "A_std": A_std,
        "ch_names": ch_ref,
        "sfreq": preprocessed_epochs_assoc[0].info["sfreq"],
        "onset_sample": association_A_all[0]["onset_sample"],
        "offset_sample": association_A_all[0]["offset_sample"],
        "filter": association_A_all[0]["filter"],
        "n_bins": association_A_all[0]["n_bins"],
        "n_seizures_averaged": len(association_A_all),
        "data_mean": data_mean,
    }

    return avg_info, A_stack, data_stack

def average_ec_across_seizures(all_ec, all_times, check_times=True):

    """This function averages effective-connectivity across seizures after truncating to a common length; sanity check
    
    Parameters
    ----------
    all_ec
        All ec.
    all_times
        All times.
    check_times
        Check times.
    
    Returns
    -------
    object
        The mean effective connectivity and common window times."""
    shapes = [x.shape for x in all_ec]
    print("Shapes:", shapes)

    min_windows = min(x.shape[0] for x in all_ec)

    ec_stack = np.stack([x[:min_windows]for x in all_ec], axis=0)

    times_stack = np.stack([t[:min_windows]for t in all_times], axis=0)

    if check_times:
        max_time_diff = np.max(np.abs(times_stack - times_stack[0]))
        print("max time difference:", max_time_diff)

        if max_time_diff > 1e-6:
            print("Warning: window times are not perfectly aligned.")

    ec_mean = np.mean(ec_stack, axis=0)
    common_times = times_stack[0]

    return ec_mean, common_times

def split_ec_preictal_ictal(ec_matrices_all):
    """This function is used to split each sequence into preictal and ictal windows.
    
    Parameters
    ----------
    ec_matrices_all
        Ec matrices all.
    
    Returns
    -------
    object
        The ec matrices split in pre-ictal and ictal data"""
    all_ec_ictal = []
    all_ictal_times = []

    all_ec_preictal = []
    all_preictal_times = []

    split_summary = []

    for sz_idx, seizure_dict in enumerate(ec_matrices_all):
        ec_data = seizure_dict["ec"]
        window_times = seizure_dict["window_times"]

        ictal_mask = window_times >= 0
        preictal_mask = window_times < 0

        ec_ictal = ec_data[ictal_mask]
        ictal_times = window_times[ictal_mask]

        ec_preictal = ec_data[preictal_mask]
        preictal_times = window_times[preictal_mask]

        print(f"\nSeizure {sz_idx + 1}")
        print("amount of preictal windows:", ec_preictal.shape[0])
        print("amount of ictal windows:", ec_ictal.shape[0])

        all_ec_ictal.append(ec_ictal)
        all_ictal_times.append(ictal_times)

        all_ec_preictal.append(ec_preictal)
        all_preictal_times.append(preictal_times)

        split_summary.append({
            "seizure_idx": sz_idx,
            "n_preictal": ec_preictal.shape[0],
            "n_ictal": ec_ictal.shape[0],
            "preictal_start": preictal_times[0] if len(preictal_times) > 0 else np.nan,
            "preictal_end": preictal_times[-1] if len(preictal_times) > 0 else np.nan,
            "ictal_start": ictal_times[0] if len(ictal_times) > 0 else np.nan,
            "ictal_end": ictal_times[-1] if len(ictal_times) > 0 else np.nan,
        })

    split_summary = pd.DataFrame(split_summary)

    return (all_ec_preictal,all_preictal_times,all_ec_ictal,all_ictal_times,split_summary,)

def plot_mean_A_by_regime(A_ictal,ictal_times,regime_labels,ch_names=None,title="Mean h² A matrix by regime",):
    """This function plots the mean h² connectivity matrix for each detected regime; for visualization
    
    Parameters
    ----------
    A_ictal
        A ictal.
    ictal_times
        Ictal times.
    regime_labels
        Regime labels.
    ch_names
        Channel names.
    title
        Title.
    
    Returns
    -------
    object
        The mean h2 connectivity matrices"""
    unique_labels = np.unique(regime_labels)

    mean_mats = []
    regime_titles = []

    for lab in unique_labels:
        mask = regime_labels == lab

        A_mean = np.mean(A_ictal[mask], axis=0)
        mean_mats.append(A_mean)

        regime_titles.append(f"Regime {lab}\n"f"{np.sum(mask)} windows\n"f"{ictal_times[mask][0]:.1f}–{ictal_times[mask][-1]:.1f}s")

    vmax = max(np.max(A) for A in mean_mats)

    n = len(mean_mats)

    fig, axes = plt.subplots(1,n,figsize=(4.5 * n, 4),constrained_layout=True,)

    if n == 1:
        axes = [axes]

    for ax, A, regime_title in zip(axes, mean_mats, regime_titles):
        im = ax.imshow(A,aspect="auto",interpolation="nearest",vmin=0,vmax=vmax,)

        ax.set_title(regime_title)

        if ch_names is not None:
            ax.set_xticks(range(len(ch_names)))
            ax.set_yticks(range(len(ch_names)))
            ax.set_xticklabels(ch_names, rotation=90, fontsize=7)
            ax.set_yticklabels(ch_names, fontsize=7)

    fig.suptitle(title)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Mean A weight")
    plt.show()

def preprocess_ding_lssm(raw,sfreq,freq_bands=None,notch_freq=60.0,window_size=2.5,step_size=0.125,):
    """This function does the preprocessing to create the Hilbert-envelope sliding windows for the Ding pipeline.
    
    Parameters
    ----------
    raw
        Loaded MNE Raw iEEG recording.
    sfreq
        Sampling frequency in Hz.
    freq_bands
        Frequency bands.
    notch_freq
        Notch freq.
    window_size
        Window size.
    step_size
        Step size.
    
    Returns
    -------
    object
        The preprocessed feature maps and epoch list"""
    if freq_bands is None:
        freq_bands = {
            "low_gamma": (30.0, 90.0)
        }

    raw_path = Path(raw.filenames[0])
    events_file_path = raw_path.parent / f"{raw_path.stem.replace('_ieeg', '')}_events.tsv"

    if not events_file_path.exists():
        raise FileNotFoundError(f"Could not find events file at {events_file_path}")

    #events_df = pd.read_csv(events_file_path, sep="\t")

    sfreq = float(raw.info["sfreq"])
    onset_time, offset_time = get_onset_offset(events_file_path)
    onset_sample = int(round(onset_time * sfreq))
    offset_sample = int(round(offset_time * sfreq))

    feature_maps = []
    epochs_list = []

    raw_notched = raw.copy()

    if notch_freq is not None:
        raw_notched.notch_filter(freqs=notch_freq,notch_widths=2.0,picks="all",method="spectrum_fit",verbose=False,)

    print(f"\nProcessing seizure onset-to-offset "
        f"({onset_time:.2f}s → {offset_time:.2f}s, "
        f"duration={offset_time - onset_time:.2f}s)"
    )

    band_data = {}

    for band_name, (fmin, fmax) in freq_bands.items():
        fmax = min(fmax, sfreq / 2 - 1)

        print(f"  [{band_name}] filtering {fmin}-{fmax} Hz ...", end="", flush=True)

        filtered = raw_notched.copy().filter(fmin,fmax,method="iir",iir_params=dict(order=4, ftype="butter"),verbose=False,)

        data = filtered.get_data()

        analytic = hilbert(data, axis=1)
        amp_env = np.abs(analytic)

        start_samp = int(round(onset_time * sfreq))
        end_samp = int(round(offset_time * sfreq))

        start_samp = max(start_samp, 0)
        end_samp = min(end_samp, amp_env.shape[1])

        if end_samp <= start_samp:
            raise ValueError("Invalid onset-to-offset crop: end_samp <= start_samp")

        cropped = amp_env[:, start_samp:end_samp]

        # Sliding windows across full seizure
        win_samp = int(round(window_size * sfreq))
        step_samp = int(round(step_size * sfreq))

        if win_samp <= 0:
            raise ValueError("window_size is too small.")

        if step_samp <= 0:
            raise ValueError("step_size is too small.")

        n_win = max(0, (cropped.shape[1] - win_samp) // step_samp + 1)

        if n_win == 0:
            raise ValueError(
                f"Seizure duration is too short for window_size={window_size}s "
                f"at sfreq={sfreq}."
            )

        windows = []

        for w in range(n_win):
            s = w * step_samp
            e = s + win_samp
            windows.append(cropped[:, s:e])

        band_data[band_name] = windows

        print(f" done ({n_win} windows)")

    raw_epoch = raw_notched.copy().crop(tmin=onset_time,tmax=offset_time)

    epochs_list.append(raw_epoch)
    feature_maps.append(band_data)

    return feature_maps, epochs_list

def compute_gramian_controllability(A, horizon=100, reg=0.0):
    """
    This function computes the controllability Gramian for each node using discrete Lyapunov equation with regularization.
    """
    n = A.shape[0]
    Li = np.zeros(n)
    
    for i in range(n):
        Q = np.zeros((n, n))
        Q[i, i] = 1.0
        try:
            W_i = solve_discrete_lyapunov(A, Q)
        except Exception as e:
            logging.warning(f"Lyapunov solve failed for node {i}: {e}. Falling back to summation.")
            W_i = np.zeros((n, n))
            A_t = np.eye(n)
            for t in range(horizon):
                W_i += A_t @ Q @ A_t.T
                A_t = A_t @ A
        
        #Regularize the gramian
        W_i += reg * np.eye(n)
        Li[i] = np.trace(W_i)
    return Li


def process_window(w_data, lambdas, horizon, reg, max_eig_tol):
    """
    This function fits the LSSM and computes controllability for a single window.
    """ 
    X_t = w_data[:, :-1].T
    X_t1 = w_data[:, 1:].T

    stable_A = None
    A = None
    max_abs_eig = None

    for lam in lambdas:
        # Fit LSSM using current ridge regularization
        ridge = Ridge(alpha=lam, fit_intercept=False)
        ridge.fit(X_t, X_t1)
        A = ridge.coef_

        eigvals = np.linalg.eigvals(A)
        max_abs_eig = np.max(np.abs(eigvals))

        if max_abs_eig <= max_eig_tol:
            stable_A = A

            logging.debug(f"Stable A found with λ={lam:.1e}, "f"spectral radius={max_abs_eig:.4f}")
            break

    if stable_A is None:
        scaling_factor = max_eig_tol / max_abs_eig
        stable_A = A * scaling_factor

        logging.warning(f"No stable A found naturally. "f"Using λ={lam:.1e} and rescaling spectral radius "f"from {max_abs_eig:.4f} to {max_eig_tol:.4f}.")

    Li = compute_gramian_controllability(stable_A,horizon=horizon,reg=reg)

    return Li

def fit_LSSM_and_compute_controllability(feature_maps,ch_names,soz_channels=None,method_name="ding_lssm",lambdas=None,horizon=100,reg=1e-4,
                                         max_eig_tol=0.99,n_jobs=-1,):

    """This function fits ridge-regularized LSSMs to the feature windows and computes node controllability.
    
    Parameters
    ----------
    feature_maps
        Feature maps.
    ch_names
        Channel names.
    soz_channels
        Names of seizure-onset-zone channels.
    method_name
        Method name.
    lambdas
        Lambdas.
    horizon
        Horizon.
    reg
        Reg.
    max_eig_tol
        Max eig tol.
    n_jobs
        Parallel-worker setting retained for compatibility or CPU preprocessing.
    
    Returns
    -------
    object
        The controllability results for the nodes"""
    if lambdas is None:
        lambdas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    if soz_channels is None:
        soz_channels = []

    soz_set = set(soz_channels)

    controllability_results = []
    rows = []

    for sz_idx, seizure_data in enumerate(feature_maps):
        logging.info(
            f"Seizure {sz_idx + 1}/{len(feature_maps)}: fitting LSSM across bands"
        )

        result = {"C_time_resolved": {}}

        for band_name, windows in seizure_data.items():
            logging.info(f"  Band: {band_name} ({len(windows)} epochs)")

            if len(windows) == 0:
                logging.warning(
                    f"  No windows found for seizure {sz_idx + 1}, band {band_name}"
                )
                continue

            n_channels = windows[0].shape[0]
            n_windows = len(windows)

            if n_channels != len(ch_names):
                raise ValueError(
                    f"Number of channels in windows ({n_channels}) does not match "
                    f"len(ch_names) ({len(ch_names)})."
                )

            results = Parallel(n_jobs=n_jobs)(
                delayed(process_window)(w_data,lambdas,horizon,reg,max_eig_tol)
                for w_data in tqdm(
                    windows,
                    desc=f"Seizure {sz_idx + 1} | {band_name}",
                    leave=False
                )
            )

            C_time = np.vstack(results)
            result["C_time_resolved"][band_name] = C_time

            mean_per_node = np.nanmean(C_time, axis=0)
            median_per_node = np.nanmedian(C_time, axis=0)
            std_per_node = np.nanstd(C_time, axis=0)
            min_per_node = np.nanmin(C_time, axis=0)
            max_per_node = np.nanmax(C_time, axis=0)
            n_valid_windows = np.sum(np.isfinite(C_time), axis=0)

            for node_idx, ch in enumerate(ch_names):
                rows.append({
                    "method": method_name,
                    "seizure_idx": sz_idx,
                    "seizure_number": sz_idx + 1,
                    "band": band_name,
                    "node_idx": node_idx,
                    "channel": ch,
                    "mean_controllability": mean_per_node[node_idx],
                    "median_controllability": median_per_node[node_idx],
                    "std_controllability": std_per_node[node_idx],
                    "min_controllability": min_per_node[node_idx],
                    "max_controllability": max_per_node[node_idx],
                    "n_windows": n_windows,
                    "n_valid_windows": int(n_valid_windows[node_idx]),
                    "is_soz": ch in soz_set,
                })

        controllability_results.append(result)

    df_seizure_nodes = pd.DataFrame(rows)

    if len(df_seizure_nodes) > 0:
        df_seizure_nodes["rank_within_seizure"] = (
            df_seizure_nodes
            .groupby(["seizure_idx", "band"])["mean_controllability"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        group_cols = [
            "method",
            "band",
            "node_idx",
            "channel",
            "is_soz",
        ]

        df_avg_nodes = (
            df_seizure_nodes
            .groupby(group_cols, as_index=False)
            .agg(
                mean_controllability=("mean_controllability", "mean"),
                median_controllability=("mean_controllability", "median"),
                std_across_seizures=("mean_controllability", "std"),
                mean_rank_across_seizures=("rank_within_seizure", "mean"),
                n_seizures=("seizure_idx", "nunique"),
                total_valid_windows=("n_valid_windows", "sum"),
            )
        )

        df_avg_nodes["rank"] = (
            df_avg_nodes
            .groupby("band")["mean_controllability"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        df_avg_nodes = (
            df_avg_nodes
            .sort_values(["band", "rank"])
            .reset_index(drop=True)
        )

    else:
        df_avg_nodes = pd.DataFrame()

    logging.info("Finished fitting LSSM and computing controllability")

    return controllability_results, df_seizure_nodes, df_avg_nodes

def average_feature_maps(preprocessed_feature_maps):
    """This funcion averages the Ding feature-map windows across seizures.
    
    Parameters
    ----------
    preprocessed_feature_maps
        Preprocessed feature maps.
    
    Returns
    -------
    object
        Function-specific result; see the description and downstream callers for its structure."""
    band_names = list(preprocessed_feature_maps[0].keys())
    averaged_seizure = {}

    for band_name in band_names:
        all_band_windows = [
            seizure_map[band_name]
            for seizure_map in preprocessed_feature_maps
            if band_name in seizure_map
        ]

        min_n_windows = min(len(windows) for windows in all_band_windows)
        averaged_windows = []

        for w_idx in range(min_n_windows):
            windows_this_time = [
                windows[w_idx]
                for windows in all_band_windows
            ]

            windows_this_time = np.stack(windows_this_time, axis=0)

            avg_window = np.nanmean(windows_this_time, axis=0)

            averaged_windows.append(avg_window)

        averaged_seizure[band_name] = averaged_windows

    return [averaged_seizure]


def split_ding_feature_maps_by_regime(preprocessed_feature_maps,regime_df,step_size=0.125,window_size=2.5,band_names=None,):
    """This function splits the ding feature maps into regime-specific chunks.
    
    Parameters
    ----------
    preprocessed_feature_maps
        Preprocessed feature maps.
    regime_df
        Regime df.
    step_size
        Step size.
    window_size
        Window size.
    band_names
        Band names.
    
    Returns
    -------
    object
        The regime-specific feature maps."""
    regime_feature_maps = []
    regime_metadata = []

    if band_names is None:
        band_names = list(preprocessed_feature_maps[0].keys())

    regime_names = {
        0: "early",
        1: "middle",
        2: "late",
    }

    for _, row in regime_df.iterrows():

        seizure_idx = int(row["seizure_idx"])
        regime = int(row["regime"])

        start_time = float(row["start_time"])
        end_time = float(row["end_time"])

        seizure_map = (preprocessed_feature_maps[seizure_idx])
        regime_map = {}

        for band in band_names:

            windows = seizure_map[band]
            n_windows = len(windows)

            window_center_times = (np.arange(n_windows) * step_size+ (window_size / 2))

            keep_mask = ((window_center_times >= start_time)&(window_center_times <= end_time))

            selected_windows = [windows[i]for i in np.where(keep_mask)[0]]

            regime_map[band] = selected_windows

        n_selected = len(regime_map[band_names[0]])

        if n_selected == 0:
            print(
                f"Skipping seizure {seizure_idx}, "
                f"regime {regime}: "
                "no windows selected."
            )
            continue

        regime_feature_maps.append(regime_map)

        regime_metadata.append({
            "original_seizure_idx":seizure_idx,
            "seizure_number":seizure_idx + 1,
            "regime":regime,
            "regime_name": regime_names.get(regime,f"regime_{regime}",),
            "regime_start_time":start_time,
            "regime_end_time":end_time,
            "n_regime_windows":n_selected,})

    metadata_df = pd.DataFrame(regime_metadata)

    if len(metadata_df) > 0:
        metadata_df["_feature_map_idx"] = np.arange(len(metadata_df))

        metadata_df = (metadata_df.sort_values(["regime","original_seizure_idx",]).reset_index(drop=True))
        #fix when order is misaligned
        order = (metadata_df["_feature_map_idx"].astype(int).tolist())

        regime_feature_maps = [regime_feature_maps[i]for i in order]

        metadata_df = (metadata_df.drop(columns="_feature_map_idx"))
    return (regime_feature_maps,metadata_df,)

def build_ding_feature_maps(runs,sub,ses,task,acq,bids_root,freq_bands=None,notch_freq=60.0,window_size=2.5,step_size=0.125,):
    """This function combines the last functions to build the Ding feature maps for each seizure.
    
    Parameters
    ----------
    runs
        Sequence of BIDS run identifiers.
    sub
        BIDS subject identifier.
    ses
        BIDS session identifier.
    task
        BIDS task label.
    acq
        BIDS acquisition label.
    bids_root
        Root directory of the BIDS dataset.
    freq_bands
        Freq bands.
    notch_freq
        Notch freq.
    window_size
        Window size.
    step_size
        Step size.
    
    Returns
    -------
    object
        The Ding feature maps for the seizures"""
    if freq_bands is None:
        freq_bands = {"low_gamma": (30.0, 90.0)}

    preprocessed_feature_maps = []
    preprocessed_epochs = []
    all_channels = None

    for run in runs:
        print(f"\n=== Ding preprocessing: run {run} ===")

        raw, sfreq = load_raw_ieeg(sub=sub,ses=ses,task=task,acq=acq,run=run,bids_root=bids_root)

        all_channels = raw.ch_names

        feature_maps, epochs_list = preprocess_ding_lssm(raw,sfreq,freq_bands=freq_bands,notch_freq=notch_freq,window_size=window_size,step_size=step_size,)

        preprocessed_feature_maps.extend(feature_maps)
        preprocessed_epochs.extend(epochs_list)
    return preprocessed_feature_maps, preprocessed_epochs, all_channels

def build_ding_full_metadata(preprocessed_feature_maps, h2_scheid_all):
    """Construct full-seizure metadata aligning Ding outputs with the h² seizure records.
    
    Parameters
    ----------
    preprocessed_feature_maps
        Preprocessed feature maps.
    h2_scheid_all
        H2 scheid all.
    
    Returns
    -------
    object
        Function-specific result; see the description and downstream callers for its structure."""
    rows = []

    for sz_idx, seizure_map in enumerate(preprocessed_feature_maps):
        first_band = next(iter(seizure_map.keys()))
        n_windows = len(seizure_map[first_band])

        rows.append({
            "original_seizure_idx": sz_idx,
            "seizure_number": sz_idx + 1,
            "run": h2_scheid_all[sz_idx].get("run", None),
            "regime": -1,
            "regime_name": "full",
            "regime_start_time": 0.0,
            "regime_end_time": h2_scheid_all[sz_idx].get("duration", None),
            "n_regime_windows": n_windows,
        })
    return pd.DataFrame(rows)

def process_window_h2(w,data,win_samp,n_bins=100,threshold_density=0.2,min_in_edges=3,normalize=True,beta=0.95):

    """This function creates the h² connectivity matrix for the scheid method; not used in final analysis.
    
    Parameters
    ----------
    w
        W.
    data
        Data.
    win_samp
        Win samp.
    n_bins
        Number of bins used for h² association estimation.
    threshold_density
        Threshold density.
    min_in_edges
        Min in edges.
    normalize
        Normalize.
    beta
        Scaling or temporal-weighting parameter, depending on the function.
    
    Returns
    -------
    object
        Function-specific result; see the description and downstream callers for its structure."""
    try:
        start = w * win_samp
        end = start + win_samp

        X_win = data[:, start:end].T 

        if X_win.shape[0] < 2:
            return {"h2": None,"reason": "too_few_samples","window": w,}

        if np.isnan(X_win).any():
            return {"h2": None,"reason": "nan_in_window","window": w,}

        A_raw = compute_association_matrix(X_win,n_bins=n_bins,)

        np.fill_diagonal(A_raw, 0.0)

        A = A_raw.copy()

        if threshold_density is not None:
            A = threshold_A_density(A,target_density=threshold_density,min_in_edges=min_in_edges,)

        np.fill_diagonal(A, 0.0)

        if normalize:
            A = normalize_A_spectral(A,beta=beta,)

        np.fill_diagonal(A, 0.0)

        n = A.shape[0]
        offdiag = ~np.eye(n, dtype=bool)

        density = np.count_nonzero(A[offdiag]) / offdiag.sum()
        mean_h2 = np.mean(A[offdiag])
        max_h2 = np.max(A[offdiag])

        raw_density = np.count_nonzero(A_raw[offdiag]) / offdiag.sum()
        raw_mean_h2 = np.mean(A_raw[offdiag])
        raw_max_h2 = np.max(A_raw[offdiag])

        eigvals = np.linalg.eigvals(A)
        spectral_radius = np.max(np.abs(eigvals))

        return {
            "h2": A,
            "h2_raw": A_raw,
            "reason": None,
            "window": w,
            "density": density,
            "mean_h2": mean_h2,
            "max_h2": max_h2,
            "raw_density": raw_density,
            "raw_mean_h2": raw_mean_h2,
            "raw_max_h2": raw_max_h2,
            "spectral_radius": spectral_radius,
        }

    except Exception as e:
        return {
            "h2": None,
            "reason": repr(e),
            "window": w,
        }
    
def preprocess_for_ec_h2(raw,window_size=1.0,n_jobs=-1,l_freq=30.0,h_freq=90.0,notch_freq=60.0,n_bins=100,
                         threshold_density=0.2,min_in_edges=3,normalize=True,beta=0.95,keep_raw_h2=True,):
    """This function is used to preprocess the iEEG data and estimates h² effective-connectivity matrices for Scheid pipeline;
    Not used in the final analysis
    
    Parameters
    ----------
    raw
        Loaded MNE Raw iEEG recording.
    orig_sfreq
        Original sampling frequency in Hz.
    window_size
        Window size.
    n_jobs
        Parallel-worker setting retained for compatibility or CPU preprocessing.
    l_freq
        L freq.
    h_freq
        H freq.
    notch_freq
        Notch freq.
    n_bins
        Number of bins used for h² association estimation.
    threshold_density
        Threshold density.
    min_in_edges
        Min in edges.
    normalize
        Normalize.
    beta
        Temporal weighting parameter.
    keep_raw_h2
        Keep raw h2.
    
    Returns
    -------
    object
        Preprocessed h2 windows for scheid"""
    print("Start preprocessing with h²:")
    print("amount of channels:", len(raw.ch_names))
    print("Duration in seconds:", raw.times[-1])
    print("ch:", raw.ch_names)
    print("ch count:", len(raw.ch_names))

    total_start = time.time()

    raw_path = Path(raw.filenames[0])
    events_file = raw_path.parent / f"{raw_path.stem.replace('_ieeg', '')}_events.tsv"

   #events_df = pd.read_csv(events_file, sep="\t")
    sfreq = float(raw.info["sfreq"])
    onset_time, offset_time = get_onset_offset(events_file)
    onset_sample = int(round(onset_time * sfreq))
    offset_sample = int(round(offset_time * sfreq))

    print("\nStart filtering")
    raw_filt = raw.copy()

    if notch_freq is not None:
        raw_filt.notch_filter(notch_freq,notch_widths=2.0,method="spectrum_fit",verbose=False,)

    raw_filt.filter(l_freq,h_freq,method="iir",iir_params=dict(order=4, ftype="butter"),verbose=False,)

    ec_all = []
    epochs_all = []

    print("\nseizure 1")

    seizure_start = time.time()

    start_time = onset_time
    stop_time = offset_time

    if start_time < raw_filt.times[0] or stop_time > raw_filt.times[-1]:
        print(
            f"skip seizure: "
            f"[{start_time}, {stop_time}] outside recording bounds"
        )
        return ec_all, epochs_all

    epoch = raw_filt.copy().crop(start_time, stop_time)
    data = epoch.get_data()  # channels x samples

    if np.isnan(data).any():
        print("NaNs detected, skipping")
        return ec_all, epochs_all

    win_samp = int(round(window_size * sfreq))
    n_win = data.shape[1] // win_samp

    if n_win == 0:
        print("No windows available")
        return ec_all, epochs_all

    window_times = np.arange(n_win) * window_size + window_size / 2

    print("Time relative to onset:", window_times[0], "to", window_times[-1])
    print("Absolute time:", start_time, "to", stop_time)
    print("Epoch shape:", data.shape)
    print("amount of windows:", n_win)
    print("Samples for every window:", win_samp)
    print("Processing windows with h²")

    window_start = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(process_window_h2)(w=w,data=data,win_samp=win_samp,n_bins=n_bins,threshold_density=threshold_density,min_in_edges=min_in_edges,
                                   normalize=normalize,beta=beta,)
        for w in range(n_win)
    )

    valid_results = [r for r in results if r["h2"] is not None]
    valid_idx = [idx for idx, r in enumerate(results) if r["h2"] is not None]

    seizure_h2 = (
        np.array([r["h2"] for r in valid_results])
        if len(valid_results) > 0
        else np.empty((0, data.shape[0], data.shape[0]))
    )

    if keep_raw_h2:
        seizure_h2_raw = (
            np.array([r["h2_raw"] for r in valid_results])
            if len(valid_results) > 0
            else np.empty((0, data.shape[0], data.shape[0]))
        )
    else:
        seizure_h2_raw = None

    valid_window_times = (
        window_times[valid_idx]
        if len(valid_idx) > 0
        else np.array([])
    )

    elapsed = time.time() - window_start

    print("Valid windows amount:", len(valid_results), "/", n_win)
    print("Window processing time:", round(elapsed, 2), "sec")

    if len(valid_results) > 0:
        densities = np.array([r["density"] for r in valid_results], dtype=float)
        means = np.array([r["mean_h2"] for r in valid_results], dtype=float)
        maxes = np.array([r["max_h2"] for r in valid_results], dtype=float)
        raw_means = np.array([r["raw_mean_h2"] for r in valid_results], dtype=float)
        raw_maxes = np.array([r["raw_max_h2"] for r in valid_results], dtype=float)
        spectral_radii = np.array([r["spectral_radius"] for r in valid_results], dtype=float)

        all_finite = np.all(np.isfinite(seizure_h2))
        all_diag_zero = np.all([
            np.allclose(np.diag(m), 0.0, atol=1e-8)
            for m in seizure_h2
        ])
        any_negative = np.any(seizure_h2 < -1e-10)
        mean_asymmetry = np.mean([
            np.mean(np.abs(m - m.T))
            for m in seizure_h2
        ])

        print("Global sanity checks:")
        print("  all finite:", all_finite)
        print("  all zero diagonal:", all_diag_zero)
        print("  any negative h²:", any_negative)
        print("  mean asymmetry:", round(float(mean_asymmetry), 6))

    failed_results = [r for r in results if r["h2"] is None]

    if failed_results:
        reasons = pd.Series([r["reason"] for r in failed_results]).value_counts()
        print("Failed window reasons:")
        print(reasons.to_string())

    seizure_dict = {
        "ec": seizure_h2,
        "window_times": valid_window_times,
        "results": results,
        "sfreq": sfreq,
        "ch_names": epoch.ch_names,
        "method": "h2",
        "window": "onset_to_offset",
        "onset_sample": onset_sample,
        "offset_sample": offset_sample,
        "onset_time": onset_time,
        "offset_time": offset_time,
        "duration": offset_time - onset_time,
        "window_size": window_size,
        "l_freq": l_freq,
        "h_freq": h_freq,
        "n_bins": n_bins,
        "threshold_density": threshold_density,
        "min_in_edges": min_in_edges,
        "normalize": normalize,
        "beta": beta,
    }

    if keep_raw_h2:
        seizure_dict["ec_raw"] = seizure_h2_raw

    ec_all.append(seizure_dict)
    epochs_all.append(epoch)

    print(
        f"Seizure 1 total time: "
        f"{round(time.time() - seizure_start, 2)} sec"
    )

    print(f"\nFinished all seizures in {round(time.time() - total_start, 2)} sec")

    return ec_all, epochs_all


def stabilize_matrix_continuous(EC):
    """This function transforms an effective-connectivity matrix into a continuous-time system matrix.
    
    Parameters
    ----------
    EC
        Ec.
    
    Returns
    -------
    object
        A Continuous-time system matrix."""
    eigvals_ = np.linalg.eigvals(EC)
    lambda_max = np.max(np.real(eigvals_))
    A = EC / (1 + lambda_max) - np.eye(EC.shape[0])
    if np.max(np.real(np.linalg.eigvals(A))) >= 0:
        print("Watch out not stable")
    return A
   
def build_control_matrix(N, control_node):
    """This function constructs a single-node control-input matrix with weak background input to other nodes, according to the Scheid paper.
    
    Parameters
    ----------
    N
        N.
    control_node
        Control node.
    
    Returns
    -------
    object
        Control-input matrix."""
    B = np.zeros((N, N))
    B[control_node, control_node] = 1.0
    return B
    
def average_controllability_continuous(A, B):
    """This function computes the average controllability like the Scheid paper does, not used in final analysis.
    
    Parameters
    ----------
    A
        A matrix.
    B
        B.
    
    Returns
    -------
    object
        Average controllability score."""
    W = solve_continuous_lyapunov(A, -B @ B.T)
    return np.trace(W)

def modal_metrics(A, percent=0.15):
    """This function computes modal, persistent and transient controllability, like Scheid et al.; not used in final analysis.
    
    Parameters
    ----------
    A
        Network/system matrix.
    percent
        Percent.
    
    Returns
    -------
    object
        Modal, persistent-modal, and transient-modal controllability vectors."""
    eigvals_, eigvecs = eig(A)
    eigvals_ = np.real(eigvals_)
    eigvecs = np.real(eigvecs)
    N = len(eigvals_)
    phi = np.zeros(N)
    for i in range(N):
        for j in range(N):
            phi[i] += (1 - (np.exp(eigvals_[j]))**2) * (eigvecs[i, j]**2)

    n_select = int(np.ceil(percent * N))
    idx_sorted = np.argsort(eigvals_)
    transient_idx = idx_sorted[:n_select]
    persistent_idx = idx_sorted[-n_select:]
    phi_persistent = np.zeros(N)
    phi_transient = np.zeros(N)
    
    for i in range(N):
        for j in persistent_idx:
            phi_persistent[i] += (1 - (np.exp(eigvals_[j]))**2) * (eigvecs[i, j]**2)
        for j in transient_idx:
            phi_transient[i] += (1 - (np.exp(eigvals_[j]))**2) * (eigvecs[i, j]**2)
    return phi, phi_persistent, phi_transient

def compute_controllability_window_continuous(A):
    """This function combines the past functions and computes average, modal, persistent, and transient controllability for every node.
    
    Parameters
    ----------
    A
        Network/system matrix.
    
    Returns
    -------
    object
        Average, modal, persistent, and transient node-controllability."""
    A = stabilize_matrix_continuous(A)
    N = A.shape[0]
    avg_ctrl = np.zeros(N)
    for node in range(N):
        B = build_control_matrix(N, node) 
        avg_ctrl[node] = average_controllability_continuous(A, B)
    modal_ctrl, persistent_ctrl, transient_ctrl = modal_metrics(A)
    return avg_ctrl, modal_ctrl, persistent_ctrl, transient_ctrl

def compute_scheid_controllability(h2_scheid_all,ch_names=None,soz_channels=None,ec_key="ec",time_key="window_times",method_name="scheid_h2",):

    """This function computes the Scheid-style controllability over all h² connectivity windows.
    
    Parameters
    ----------
    h2_scheid_all
        H2 scheid all.
    ch_names
        Channel names in matrix/node order.
    soz_channels
        Names of seizure-onset-zone channels.
    ec_key
        Ec key.
    time_key
        Time key.
    method_name
        Method name.
    
    Returns
    -------
    object
        The controllability results"""
    if soz_channels is None:
        soz_channels = []

    soz_set = set(soz_channels)

    all_results = []
    rows = []

    for seizure_counter, seizure in enumerate(h2_scheid_all):
        A_seq = seizure[ec_key]
        times = seizure[time_key]

        if ch_names is None:
            if "ch_names" in seizure:
                ch_names_used = seizure["ch_names"]
            else:
                ch_names_used = [f"node_{i}" for i in range(A_seq.shape[1])]
        else:
            ch_names_used = ch_names

        n_windows, n_channels, _ = A_seq.shape

        avg_all = np.zeros((n_windows, n_channels))
        modal_all = np.zeros((n_windows, n_channels))
        persistent_all = np.zeros((n_windows, n_channels))
        transient_all = np.zeros((n_windows, n_channels))

        for t in range(n_windows):
            avg, modal, pers, trans = compute_controllability_window_continuous(A_seq[t])

            avg_all[t] = avg
            modal_all[t] = modal
            persistent_all[t] = pers
            transient_all[t] = trans

        seizure_id = seizure.get("seizure_idx", seizure_counter)

        result = {
            "method": method_name,
            "run": seizure.get("run", None),
            "seizure_idx": seizure_id,
            "times": times,
            "window_level": {
                "average": avg_all,
                "modal": modal_all,
                "persistent": persistent_all,
                "transient": transient_all,
            },
            "ch_names": ch_names_used,
        }

        all_results.append(result)

        for node_idx, ch in enumerate(ch_names_used):
            rows.append({
                "method": method_name,
                "run": seizure.get("run", None),
                "seizure_idx": seizure_id,
                "node_idx": node_idx,
                "channel": ch,
                "is_soz": ch in soz_set,

                "mean_average_controllability": np.nanmean(avg_all[:, node_idx]),
                "median_average_controllability": np.nanmedian(avg_all[:, node_idx]),

                "mean_modal_controllability": np.nanmean(modal_all[:, node_idx]),
                "median_modal_controllability": np.nanmedian(modal_all[:, node_idx]),

                "mean_persistent_controllability": np.nanmean(persistent_all[:, node_idx]),
                "mean_transient_controllability": np.nanmean(transient_all[:, node_idx]),

                "n_windows": n_windows,
            })

    df_seizure_nodes = pd.DataFrame(rows)

    df_avg_nodes = (
        df_seizure_nodes.groupby(["method", "node_idx", "channel", "is_soz"],as_index=False,).agg(
            mean_average_controllability=("mean_average_controllability", "mean"),
            median_average_controllability=("mean_average_controllability", "median"),
            mean_modal_controllability=("mean_modal_controllability", "mean"),
            median_modal_controllability=("mean_modal_controllability", "median"),
            mean_persistent_controllability=("mean_persistent_controllability", "mean"),
            mean_transient_controllability=("mean_transient_controllability", "mean"),
            n_seizures=("seizure_idx", "nunique"),
            total_windows=("n_windows", "sum"),))

    df_avg_nodes["rank_average"] = (df_avg_nodes["mean_average_controllability"].rank(ascending=False, method="min").astype(int))

    df_avg_nodes["rank_modal"] = (df_avg_nodes["mean_modal_controllability"].rank(ascending=False, method="min").astype(int))

    df_avg_nodes["rank_persistent"] = (df_avg_nodes["mean_persistent_controllability"].rank(ascending=False, method="min").astype(int))

    df_avg_nodes["rank_transient"] = (df_avg_nodes["mean_transient_controllability"].rank(ascending=False, method="min").astype(int))

    df_avg_nodes = df_avg_nodes.sort_values("rank_average").reset_index(drop=True)

    return all_results, df_seizure_nodes, df_avg_nodes

def build_h2_scheid_all(runs,sub,ses,task,acq,bids_root,window_size=1.0,n_jobs=-1,l_freq=30.0,h_freq=90.0,notch_freq=60.0,n_bins=100,
                        threshold_density=0.2,min_in_edges=3,normalize=True,beta=0.95,keep_raw_h2=True,):
    """This function loads in all the runs and constructs the full-seizure h² connectivity sequences for Scheid analysis. In the end unused.
    
    Parameters
    ----------
    runs
        The sequence of BIDS run identifiers.
    sub
        BIDS subject identifier.
    ses
        BIDS session identifier.
    task
        BIDS task label.
    acq
        BIDS acquisition label.
    bids_root
        Root directory of the BIDS dataset.
    orig_sfreq
        Original sampling frequency in Hz.
    window_size
        Window size.
    n_jobs
        Parallel-worker setting retained for compatibility or CPU preprocessing.
    l_freq
        L freq.
    h_freq
        H freq.
    notch_freq
        Notch freq.
    n_bins
        Number of bins used for h² association estimation.
    threshold_density
        Threshold density.
    min_in_edges
        Min in edges.
    normalize
        Normalize.
    beta
        Temporal-weighting parameter.
    keep_raw_h2
        Keep raw h2.
    
    Returns
    -------
    object
        All of the scheid h2 matrices for a single patient."""
    h2_scheid_all = []
    all_channels = None

    for run in runs:
        print(f"\n=== h² preprocessing: run {run} ===")

        raw, sfreq = load_raw_ieeg(sub=sub,ses=ses,task=task,acq=acq,run=run,bids_root=bids_root)

        if all_channels is None:
            all_channels = raw.ch_names.copy()

        ec_matrices_h2_all, epochs_h2_all = preprocess_for_ec_h2(raw=raw,window_size=window_size,n_jobs=n_jobs,l_freq=l_freq,
                                                                 h_freq=h_freq,notch_freq=notch_freq,n_bins=n_bins,threshold_density=threshold_density,
                                                                 min_in_edges=min_in_edges,normalize=normalize,beta=beta,keep_raw_h2=keep_raw_h2,)

        for r in ec_matrices_h2_all:
            r["run"] = run

        h2_scheid_all.extend(ec_matrices_h2_all)

    return h2_scheid_all, all_channels

def build_h2_regime_dicts(regime_results_h2_ictal,all_ec_ictal_h2,all_ictal_times_h2,h2_scheid_all,all_channels,):
    """This function is used to extract the regime-specific h² connectivity chunks and the associated metadata.
    
    Parameters
    ----------
    regime_results_h2_ictal
        Regime results h2 ictal.
    all_ec_ictal_h2
        All ec ictal h2.
    all_ictal_times_h2
        All ictal times h2.
    h2_scheid_all
        H2 scheid all.
    all_channels
        All channels.
    
    Returns
    -------
    object
        The regime connectivity dictionaries and their metadata table."""
    h2_regime_dicts = []

    regime_names = {
        0: "early",
        1: "middle",
        2: "late",
    }

    for sz_idx, result in enumerate(regime_results_h2_ictal):
        A_ictal = all_ec_ictal_h2[sz_idx]
        times = all_ictal_times_h2[sz_idx]
        labels = result["labels"]

        if not (len(A_ictal) == len(times) == len(labels)):
            raise ValueError(
                f"Length mismatch for seizure {sz_idx}: "
                f"A={len(A_ictal)}, times={len(times)}, labels={len(labels)}"
            )

        for regime in np.unique(labels):
            mask = labels == regime

            h2_regime_dicts.append({
                "ec": A_ictal[mask],
                "window_times": times[mask],
                "ch_names": all_channels,

                "original_seizure_idx": sz_idx,
                "seizure_idx": sz_idx,
                "seizure_number": sz_idx + 1,
                "run": h2_scheid_all[sz_idx].get("run", None),

                "regime": int(regime),
                "regime_name": regime_names.get(int(regime), f"regime_{regime}"),

                "gamma": result["gamma"],
                "Q": result["Q"],
                "n_windows": int(mask.sum()),
                "start_time": float(times[mask][0]),
                "end_time": float(times[mask][-1]),
            })

    regime_order = {
        "early": 0,
        "middle": 1,
        "late": 2,
    }

    h2_regime_dicts = sorted(h2_regime_dicts,key=lambda d: (regime_order.get(d["regime_name"], 999),d["original_seizure_idx"],),)

    for chunk_idx, d in enumerate(h2_regime_dicts):
        d["chunk_idx"] = chunk_idx

    h2_regime_metadata = pd.DataFrame([
        {
            "chunk_idx": d["chunk_idx"],
            "original_seizure_idx": d["original_seizure_idx"],
            "seizure_number": d["seizure_number"],
            "run": d["run"],
            "regime": d["regime"],
            "regime_name": d["regime_name"],
            "regime_start_time": d["start_time"],
            "regime_end_time": d["end_time"],
            "n_regime_windows": d["n_windows"],
            "gamma": d["gamma"],
            "Q": d["Q"],
        }
        for d in h2_regime_dicts
    ])

    return h2_regime_dicts, h2_regime_metadata

def build_h2_full_seizure_dicts(h2_scheid_all,all_ec_ictal_h2,all_ictal_times_h2,all_channels,):

    """This function constructs the full-seizure h² dictionaries with the channel and timing metadata.
    
    Parameters
    ----------
    h2_scheid_all
        H2 scheid all.
    all_ec_ictal_h2
        All ec ictal h2.
    all_ictal_times_h2
        All ictal times h2.
    all_channels
        All channels.
    
    Returns
    -------
    object
        The full-seizure h² connectivity dictionaries."""
    h2_full_dicts = []

    for sz_idx, (A_ictal, ictal_times) in enumerate(
        zip(all_ec_ictal_h2, all_ictal_times_h2)):
        h2_full_dicts.append({
            "ec": A_ictal,
            "window_times": ictal_times,
            "ch_names": all_channels,
            "original_seizure_idx": sz_idx,
            "seizure_idx": sz_idx,
            "seizure_number": sz_idx + 1,
            "run": h2_scheid_all[sz_idx].get("run", None),
            "regime": -1,
            "regime_name": "full",
            "gamma": None,
            "Q": None,
            "n_windows": int(len(ictal_times)),
            "start_time": float(ictal_times[0]),
            "end_time": float(ictal_times[-1]),
        })

    return h2_full_dicts

def build_seizure_infos_regime(h2_regime_dicts,h2_scheid_all,sub,ses,task,acq,bids_root,sigma_range=(-0.20, -0.10)):
    """This function is used for our simulation pipeline, to build regime networks and get the energy-derived oscillator parameters.
    
    Parameters
    ----------
    h2_regime_dicts
        H2 regime dicts.
    h2_scheid_all
        H2 scheid all.
    sub
        BIDS subject identifier.
    ses
        BIDS session identifier.
    task
        BIDS task label.
    acq
        BIDS acquisition label.
    bids_root
        Root directory of the BIDS dataset.
    sigma_range
        Sigma range.
    
    Returns
    -------
    object
        The seizure info per regime, including the bistable oscillator parameters."""
    seizure_infos_regime = []

    for chunk_idx, d in enumerate(h2_regime_dicts):
        A_seq = d["ec"]
        A_rep, rep_window_idx = get_regime_representative_A(A_seq)

        original_seizure_idx = d["original_seizure_idx"]
        run = d.get("run", None)

        regime_start_time = d.get("start_time", np.nan)
        regime_end_time = d.get("end_time", np.nan)

        raw, sfreq = load_raw_ieeg(sub=sub,ses=ses,task=task,acq=acq,run=run,bids_root=bids_root)

        onset_time = h2_scheid_all[original_seizure_idx].get("onset_time", 0.0)

        crop_start = onset_time + regime_start_time - 0.5
        crop_end = onset_time + regime_end_time + 0.5

        crop_start = max(crop_start, raw.times[0])
        crop_end = min(crop_end, raw.times[-1])

        raw_regime = raw.copy().crop(tmin=crop_start,tmax=crop_end,)

        data_regime = raw_regime.get_data()

        sigma, energies, energy_norm = compute_energy_based_sigma(data=data_regime,sfreq=sfreq,window_size=1.0,overlap=0.5,sigma_range=sigma_range,)

        info = {
            "chunk_idx": chunk_idx,
            "original_seizure_idx": original_seizure_idx,
            "seizure_idx": original_seizure_idx,
            "seizure_number": original_seizure_idx + 1,
            "run": run,
            "regime": d["regime"],
            "regime_name": d.get("regime_name", f"regime_{d['regime']}"),
            "A": A_rep,
            "A_seq": A_seq,
            "representative_window_idx": rep_window_idx,
            "representative_window_time": float(d["window_times"][rep_window_idx]),
            "sigma": sigma,
            "energies": energies,
            "energy_norm": energy_norm,
            "ch_names": d["ch_names"],
            "regime_start_time": regime_start_time,
            "regime_end_time": regime_end_time,
            "crop_start_time": crop_start,
            "crop_end_time": crop_end,
            "n_regime_windows": d.get("n_windows", len(A_seq)),
        }

        seizure_infos_regime.append(info)

    return seizure_infos_regime


def _kashima_initial_state_torch(torch, batch_size, n_nodes, dtype, device, start_point, generator):
    """This function starts the oscillator states for Kashima perturbation simulations. 
        In the end only equilibrium was used, the others were used earlier as sanity checks.
    
    Parameters
    ----------
    torch
        Torch.
    batch_size
        Number of simulations processed per GPU batch.
    n_nodes
        Number of nodes.
    dtype
        Floating-point precision for PyTorch computation.
    device
        PyTorch device specification; if None, use the configured default.
    start_point
        Start point.
    generator
        Generator.
    
    Returns
    -------
    object
        The x and y starting value for running the Kashima pipeline."""
    if start_point == "equilibrium":
        x = torch.zeros((batch_size, n_nodes), dtype=dtype, device=device)
        y = torch.zeros_like(x)
    elif start_point == "small_random":
        x = 0.01 * torch.randn((batch_size, n_nodes), dtype=dtype, device=device, generator=generator)
        y = 0.01 * torch.randn((batch_size, n_nodes), dtype=dtype, device=device, generator=generator)
    elif start_point == "limit_cycle":
        x = torch.ones((batch_size, n_nodes), dtype=dtype, device=device)
        y = torch.ones_like(x)
    else:
        raise ValueError("start_point must be 'equilibrium', 'small_random', or 'limit_cycle'")
    return x, y


def _simulate_kashima_terminal_batch_gpu(*, A_t, omega_t, sigma_t, alpha_t, beta_t, input_nodes,
                                         dt, steps, k, noise_amp, start_point, generator,):
    """This function simulates a single batch of Kashima noise realizations and returns the terminal states.
    """
    torch = _require_torch()
    batch_size = int(input_nodes.numel())
    n_nodes = int(A_t.shape[0])
    x, y = _kashima_initial_state_torch(torch, batch_size, n_nodes, A_t.dtype, A_t.device, start_point, generator)

    row_sums = A_t.sum(dim=1)
    cos_w = torch.cos(omega_t * dt)
    sin_w = torch.sin(omega_t * dt)
    sqrt_dt = float(np.sqrt(dt))
    row_idx = torch.arange(batch_size, device=A_t.device)

    for _ in range(int(steps)):
        r2 = x.square() + y.square()
        g_vec = 2.0 * alpha_t.unsqueeze(0) * beta_t.unsqueeze(0) * r2 - beta_t.unsqueeze(0) * r2.square()

        coupling_x = k * (torch.matmul(x, A_t.T) - x * row_sums)
        coupling_y = k * (torch.matmul(y, A_t.T) - y * row_sums)
        growth = sigma_t.unsqueeze(0) + g_vec

        x_temp = x + (x * growth + coupling_x) * dt
        y_temp = y + (y * growth + coupling_y) * dt

        if noise_amp > 0:
            nx = noise_amp * sqrt_dt * torch.randn((batch_size,), dtype=A_t.dtype, device=A_t.device, generator=generator)
            ny = noise_amp * sqrt_dt * torch.randn((batch_size,), dtype=A_t.dtype, device=A_t.device, generator=generator)
            x_temp[row_idx, input_nodes] += nx
            y_temp[row_idx, input_nodes] += ny

        x = cos_w * x_temp - sin_w * y_temp
        y = sin_w * x_temp + cos_w * y_temp

    return torch.cat((x, y), dim=1)


def nonlinear_controllability_from_h2_ec_matrix_gpu(a,band_center_hz,sigma_val=-0.05,alpha_val=1.0,beta_val=1.0,k=0.1,dt=0.001,steps=5000,
                                                    noise_amp=0.05,num_realizations=20,random_seed=42,start_point="equilibrium",batch_size=512,
                                                    device=None,dtype="float64",return_gramians=False,):
    """This function calculates the GPU-batched nonlinear Kashima-based controllability value for one single adjacency matrix.
    """
    torch, resolved_device = _resolve_torch_device(device)
    torch_dtype = _torch_dtype(torch, dtype)

    a_used = np.asarray(a, dtype=float).copy()
    a_used[~np.isfinite(a_used)] = 0.0
    a_used[a_used < 0] = 0.0
    np.fill_diagonal(a_used, 0.0)

    if a_used.ndim != 2 or a_used.shape[0] != a_used.shape[1]:
        raise ValueError(f"Expected a square matrix, got {a_used.shape}.")
    if num_realizations <= 0 or steps <= 0 or batch_size <= 0:
        raise ValueError("num_realizations, steps and batch_size must be positive.")

    n_nodes = a_used.shape[0]
    state_dim = 2 * n_nodes
    A_t = torch.as_tensor(a_used, dtype=torch_dtype, device=resolved_device)
    omega_t = torch.full((n_nodes,), 2 * np.pi * band_center_hz, dtype=torch_dtype, device=resolved_device)
    sigma_t = torch.full((n_nodes,), sigma_val, dtype=torch_dtype, device=resolved_device)
    alpha_t = torch.full((n_nodes,), alpha_val, dtype=torch_dtype, device=resolved_device)
    beta_t = torch.full((n_nodes,), beta_val, dtype=torch_dtype, device=resolved_device)

    all_input_nodes = torch.arange(n_nodes, device=resolved_device).repeat_interleave(num_realizations)
    gramian_sums = torch.zeros((n_nodes, state_dim, state_dim), dtype=torch_dtype, device=resolved_device)
    counts = torch.zeros(n_nodes, dtype=torch.int64, device=resolved_device)

    total = int(all_input_nodes.numel())
    for batch_start in range(0, total, batch_size):
        input_nodes = all_input_nodes[batch_start:batch_start + batch_size]
        generator = torch.Generator(device=resolved_device)
        generator.manual_seed(int(random_seed + batch_start))
        terminal = _simulate_kashima_terminal_batch_gpu(A_t=A_t, omega_t=omega_t, sigma_t=sigma_t, alpha_t=alpha_t, beta_t=beta_t,
                                                        input_nodes=input_nodes, dt=dt, steps=steps, k=k, noise_amp=noise_amp,
                                                        start_point=start_point, generator=generator,)

        finite = torch.isfinite(terminal).all(dim=1)
        for node_idx in torch.unique(input_nodes):
            mask = (input_nodes == node_idx) & finite
            z = terminal[mask]
            if z.shape[0] > 0:
                gramian_sums[node_idx] += z.T @ z
                counts[node_idx] += z.shape[0]

    gramians_t = torch.full_like(gramian_sums, torch.nan)
    valid = counts > 0
    gramians_t[valid] = gramian_sums[valid] / counts[valid, None, None]

    # Now unused code for if we start in limit cycle and want to subtract baseline
    if start_point == "limit_cycle" and bool(valid.any().item()):
        baseline_nodes = torch.zeros(1, dtype=torch.long, device=resolved_device)
        baseline_gen = torch.Generator(device=resolved_device)
        baseline_gen.manual_seed(int(random_seed))
        z0 = _simulate_kashima_terminal_batch_gpu(
            A_t=A_t, omega_t=omega_t, sigma_t=sigma_t, alpha_t=alpha_t, beta_t=beta_t,
            input_nodes=baseline_nodes, dt=dt, steps=steps, k=k, noise_amp=0.0,
            start_point=start_point, generator=baseline_gen,
        )[0]
        baseline_W = torch.outer(z0, z0)
        gramians_t[valid] -= baseline_W

    scores_t = torch.diagonal(gramians_t, dim1=1, dim2=2).sum(dim=1)
    scores = scores_t.detach().cpu().numpy()
    gramians = ([w for w in gramians_t.detach().cpu().numpy()]if return_gramians else None)
    return scores, gramians, a_used


def nonlinear_controllability_from_h2_ec_matrix(a, band_center_hz, sigma_val=-0.5, alpha_val=1.0, beta_val=1.0,k=0.1, dt=0.001, steps=5000,
                                                noise_amp=0.05,num_realizations=20, n_jobs=None, random_seed=42,start_point="equilibrium",
                                                return_example_trajs=False, example_nodes=None,batch_size=512, device=None, dtype="float64", return_gramians=True,):
    """Compatibility wrapper using the GPU Kashima implementation."""
    #hotfix to remove these because they are no longer needed
    del n_jobs, example_nodes
    result = nonlinear_controllability_from_h2_ec_matrix_gpu(a=a, band_center_hz=band_center_hz, sigma_val=sigma_val,alpha_val=alpha_val,
                                                             beta_val=beta_val, k=k, dt=dt, steps=steps,noise_amp=noise_amp, 
                                                             num_realizations=num_realizations,random_seed=random_seed, 
                                                             start_point=start_point,batch_size=batch_size, device=device, dtype=dtype,
                                                             return_gramians=return_gramians,)
    if return_example_trajs:
        return (*result, {})
    return result


def run_kashima_on_scheid_h2_ec(ec_all,seizure_idx,band_center_hz,sigma_val=-0.8,alpha_val=1.0,beta_val=1.0,k=1.0,dt=0.001,steps=5000,noise_amp=0.08,
                                num_realizations=50,n_jobs=None,random_seed=42,start_point="equilibrium",window_times=None,batch_size=512,device=None,
                                dtype="float64",return_gramians=True,):

    """This function computes the nonlinear Kashima-based controllability for every window of one h² connectivity sequence.
    
    Parameters
    ----------
    ec_all
        Sequence of effective-connectivity arrays.
    seizure_idx
        Index of the seizure or connectivity chunk to process.
    band_center_hz
        Oscillator center frequency in Hz.
    sigma_val
        Sigma val.
    alpha_val
        Alpha val.
    beta_val
        Beta val.
    k
        K.
    dt
        Simulation time step in seconds.
    steps
        Steps.
    noise_amp
        Noise amp.
    num_realizations
        Num realizations.
    n_jobs
        Parallel-worker setting retained for compatibility or CPU preprocessing.
    random_seed
        Random seed.
    start_point
        Start point.
    window_times
        Window times.
    batch_size
        Number of simulations processed per GPU batch.
    device
        PyTorch device specification; if None, use the configured default.
    dtype
        Floating-point precision for PyTorch computation.
    return_gramians
        Whether to retain and return full empirical Gramians in addition to their trace scores.
    
    Returns
    -------
    object
        The nonlinear controllability value for all windows in a single seizure."""
    h2_windows = np.asarray(ec_all[seizure_idx],dtype=float)

    if h2_windows.ndim != 3:
        raise ValueError(f"Expected (n_windows, N, N), got "f"{h2_windows.shape}.")

    n_windows, n_nodes, n_nodes_2 = h2_windows.shape

    if n_nodes != n_nodes_2:
        raise ValueError(f"Expected square matrices, got "f"{h2_windows.shape}.")

    window_times_this = (np.arange(n_windows)if window_times is None else np.asarray(window_times[seizure_idx]))

    controllability = np.full((n_nodes, n_windows),np.nan,dtype=float,)

    A_list = []
    gramians_all = []

    for w, A_raw in enumerate(h2_windows):
        scores, gramians, A_final = (
            nonlinear_controllability_from_h2_ec_matrix_gpu(a=A_raw,band_center_hz=band_center_hz,sigma_val=sigma_val,alpha_val=alpha_val,beta_val=beta_val,
                                                            k=k,dt=dt,steps=steps,noise_amp=noise_amp,num_realizations=num_realizations,
                                                            random_seed=(random_seed+ 10_000 * seizure_idx+ 100 * w),start_point=start_point,
                                                            batch_size=batch_size,device=device,dtype=dtype,return_gramians=return_gramians,))

        controllability[:, w] = scores
        A_list.append(A_final)

        if return_gramians:
            gramians_all.append(gramians)

    return {
        "seizure_idx": seizure_idx,
        "controllability": controllability,
        "window_times": window_times_this,
        "A_list": A_list,
        "gramians": (
            gramians_all
            if return_gramians
            else None
        ),
    }

def run_kashima_on_all_scheid_h2_ec(ec_all,ch_names,band_center_hz,sigma_val=-0.5,alpha_val=1.0,beta_val=1.0,k=1.0,dt=0.001,steps=5000,noise_amp=0.08,
                                    num_realizations=50,n_jobs=None,random_seed=42,start_point="equilibrium",batch_size=512,device=None,dtype="float64",
                                    return_gramians=True,print_chunk_progress=False):

    """This function finally computes the nonlinear Kashima-based controllability for all chunks and creates a table withthe results.
    
    Parameters
    ----------
    ec_all
        Sequence of effective-connectivity arrays.
    ch_names
        Channel names in matrix/node order.
    band_center_hz
        Oscillator center frequency in Hz.
    sigma_val
        Sigma val.
    alpha_val
        Alpha val.
    beta_val
        Beta val.
    k
        K.
    dt
        Simulation time step in seconds.
    steps
        Steps.
    noise_amp
        Noise amp.
    num_realizations
        Num realizations.
    n_jobs
        Parallel-worker setting retained for compatibility or CPU preprocessing.
    random_seed
        Random seed.
    start_point
        Start point.
    batch_size
        Number of simulations processed per GPU batch.
    device
        PyTorch device specification; if None, use the configured default.
    dtype
        Floating-point precision for PyTorch computation.
    return_gramians
        Whether to retain and return full empirical Gramians in addition to their trace scores.
    print_chunk_progress
        Print chunk progress.
    
    Returns
    -------
    object
        Results for all different chunks."""
    all_results = []
    rows = []

    n_chunks = len(ec_all)

    for seizure_idx in range(n_chunks):
        chunk_start = time.perf_counter()

        if print_chunk_progress:
            n_windows = len(ec_all[seizure_idx])

            print(
                "\n"
                f"[Kashima chunk "
                f"{seizure_idx + 1}/{n_chunks}] "
                f"STARTING "
                f"({n_windows} windows)",
                flush=True,
            )

        res = run_kashima_on_scheid_h2_ec(ec_all=ec_all,seizure_idx=seizure_idx,band_center_hz=band_center_hz,sigma_val=sigma_val,alpha_val=alpha_val,
                                          beta_val=beta_val,k=k,dt=dt,steps=steps,noise_amp=noise_amp,num_realizations=num_realizations,
                                          random_seed=random_seed,start_point=start_point,batch_size=batch_size,device=device,dtype=dtype,
                                          return_gramians=return_gramians,)

        all_results.append(res)

        C = res["controllability"]
        times = res["window_times"]

        for node_idx, ch in enumerate(ch_names):
            for w, t in enumerate(times):
                rows.append({
                    "seizure_idx": seizure_idx,
                    "node_idx": node_idx,
                    "channel": ch,
                    "window_idx": w,
                    "window_time": t,
                    "kashima_score": C[node_idx, w],
                })

        if print_chunk_progress:
            chunk_elapsed = (
                time.perf_counter() - chunk_start
            )

            print(
                f"[Kashima chunk "
                f"{seizure_idx + 1}/{n_chunks}] "
                f"FINISHED in "
                f"{chunk_elapsed / 60:.2f} minutes",
                flush=True,
            )

    return {
        "all_results": all_results,
        "df": pd.DataFrame(rows),
    }

def prepare_patient_regime_inputs(patient_id,bids_root=BIDS_ROOT,n_jobs=-1,):

    """This function runs the full preparation pipeline, with all different methods combined for a single specified patient.
    
    Parameters
    ----------
    patient_id
        Internal patient identifier used to select configuration and output paths.
    bids_root
        Root directory of the BIDS dataset.
    n_jobs
        Parallel-worker setting retained for CPU preprocessing.
    
    Returns
    -------
    object
        The full prepared data for a single patient."""
    cfg = get_patient_config(patient_id)

    sub = cfg["sub"]
    ses = cfg["ses"]
    task = cfg["task"]
    acq = cfg["acq"]
    runs = cfg["runs"]
    soz_channels = cfg["soz_channels"]
    exclude_channels = cfg.get("exclude_channels",[],)

    print("=" * 80)
    print(f"Preparing patient: {patient_id}")
    print(f"Subject: {sub}")
    print(f"Session: {ses}")
    print(f"Runs: {runs}")
    print(f"SOZ channels: {soz_channels}")
    print(f"Excluded channels: {exclude_channels}")
    print("=" * 80)
    print("Detecting usable patient channels...")

    eeg_channels = get_patient_channels(sub=sub,ses=ses,task=task,acq=acq,run=runs[0],bids_root=bids_root,)

    print("Numberr of analysis channels:",len(eeg_channels),)

    print("Analysis channels: ",eeg_channels,)
    print("Building Ding feature maps...")

    (preprocessed_feature_maps,preprocessed_epochs,all_channels_ding,) = build_ding_feature_maps(runs=runs,sub=sub,ses=ses,task=task,acq=acq,
                                                                                                 bids_root=bids_root,freq_bands={"low_gamma": (30.0, 90.0)},
                                                                                                 notch_freq=60.0,window_size=2.5,step_size=0.125,)

    print("Building h² matrices...")
    (h2_scheid_all,all_channels_h2,) = build_h2_scheid_all(runs=runs,sub=sub,ses=ses,task=task,acq=acq,bids_root=bids_root,
                                                           window_size=1.0,n_jobs=n_jobs,l_freq=30.0,h_freq=90.0,notch_freq=60.0,n_bins=100,
                                                           threshold_density=0.2,min_in_edges=3,normalize=True,beta=0.95,keep_raw_h2=True,)
    all_channels = all_channels_h2
    if list(all_channels_ding) != list(all_channels_h2):
        raise ValueError(
            "Channel mismatch between Ding "
            "preprocessing and h² preprocessing."
        )

    print("Splitting h² into preictal and ictal...")

    (all_ec_preictal_h2,all_preictal_times_h2,all_ec_ictal_h2,all_ictal_times_h2,split_summary_h2,) = split_ec_preictal_ictal(h2_scheid_all)

    print("Detecting h² ictal regimes...")

    (regime_results_h2_ictal,regime_summary_h2_ictal,) = detect_h2_ictal_regimes_all(all_ec_ictal_h2=all_ec_ictal_h2,all_ictal_times_h2=all_ictal_times_h2,
                                                                                     beta=0.01,gamma_range=np.arange(0.80,1.101,0.002,),n_runs=50,seed=0,
                                                                                     prefer_three=True,verbose=True,make_contiguous=True,)

    print("Building Ding regime feature maps...")
    (ding_regime_feature_maps,ding_regime_metadata,) = split_ding_feature_maps_by_regime(preprocessed_feature_maps=preprocessed_feature_maps,
                                                                                         regime_df=regime_summary_h2_ictal,step_size=0.125,window_size=2.5,
                                                                                         band_names=["low_gamma"],)

    print("Building h² regime dictionaries...")

    (h2_regime_dicts,h2_regime_metadata,) = build_h2_regime_dicts(regime_results_h2_ictal=regime_results_h2_ictal,all_ec_ictal_h2= all_ec_ictal_h2,
                                                                  all_ictal_times_h2= all_ictal_times_h2,h2_scheid_all= h2_scheid_all,
                                                                  all_channels=all_channels,)

    print("Building h² full-seizure dictionaries...")

    h2_full_dicts = (build_h2_full_seizure_dicts(h2_scheid_all=h2_scheid_all,all_ec_ictal_h2=all_ec_ictal_h2,all_ictal_times_h2=all_ictal_times_h2,
                                                 all_channels=all_channels,))

    h2_all_analysis_dicts = (h2_full_dicts+ h2_regime_dicts)

    for chunk_idx, d in enumerate(h2_all_analysis_dicts):
        d["chunk_idx"] = chunk_idx

    h2_all_metadata = pd.DataFrame([
        {"chunk_idx":d["chunk_idx"],
        "original_seizure_idx":d["original_seizure_idx"],
        "seizure_number":d["seizure_number"],
        "run":d["run"],
        "regime":d["regime"],
        "regime_name":d["regime_name"],
        "regime_start_time":d["start_time"],
        "regime_end_time":d["end_time"],
        "n_regime_windows":d["n_windows"],
        "gamma":d["gamma"],
        "Q":d["Q"],}
        for d in h2_all_analysis_dicts])

    print("Building Ding full-seizure metadata...")

    ding_full_metadata = (build_ding_full_metadata(preprocessed_feature_maps=preprocessed_feature_maps,h2_scheid_all=h2_scheid_all,))

    ding_analysis_feature_maps = (preprocessed_feature_maps+ ding_regime_feature_maps)

    ding_analysis_metadata = (pd.concat([ding_full_metadata,ding_regime_metadata,],ignore_index=True,))

    ding_analysis_metadata["chunk_idx"] = (np.arange(len(ding_analysis_metadata)))

    # sanity check the lengths

    ding_keys = (ding_analysis_metadata[["chunk_idx","original_seizure_idx","regime","regime_name",]].reset_index(drop=True))

    h2_keys = (h2_all_metadata[["chunk_idx","original_seizure_idx","regime","regime_name",]].reset_index(drop=True))

    if not ding_keys.equals(h2_keys):
        raise ValueError(
            "Ding and h² chunk ordering "
            "do not match!\n\n"
            f"Ding:\n{ding_keys}\n\n"
            f"h²:\n{h2_keys}"
        )

    print("Ding and h² chunk ordering ""match exactly.")
    print("Building synthetic simulation ""inputs for full + regime chunks...")

    seizure_infos_analysis = (build_seizure_infos_regime(h2_regime_dicts=h2_all_analysis_dicts,h2_scheid_all=h2_scheid_all,sub=sub,ses=ses,
                                                         task=task,acq=acq,bids_root=bids_root,sigma_range=(-0.20,-0.10,),))
    print(
        "Building synthetic simulation "
        "inputs for regime-only chunks..."
    )

    seizure_infos_regime = (build_seizure_infos_regime(h2_regime_dicts=h2_regime_dicts,h2_scheid_all=h2_scheid_all,sub=sub,ses=ses,task=task,acq=acq,
                                                       bids_root=bids_root,sigma_range=(-0.20,-0.10,),))


    return {
        "patient_id": patient_id,
        "patient_config": cfg,
        "preprocessed_feature_maps":preprocessed_feature_maps,
        "preprocessed_epochs":preprocessed_epochs,
        "h2_scheid_all":h2_scheid_all,
        "all_ec_preictal_h2":all_ec_preictal_h2,
        "all_preictal_times_h2":all_preictal_times_h2,
        "all_ec_ictal_h2":all_ec_ictal_h2,
        "all_ictal_times_h2":all_ictal_times_h2,
        "split_summary_h2":split_summary_h2,
        "regime_results_h2_ictal":regime_results_h2_ictal,
        "regime_summary_h2_ictal":regime_summary_h2_ictal,
        "ding_regime_feature_maps":ding_regime_feature_maps,
        "ding_regime_metadata":ding_regime_metadata,
        "ding_analysis_feature_maps":ding_analysis_feature_maps,
        "ding_analysis_metadata":ding_analysis_metadata,
        "h2_regime_dicts":h2_regime_dicts,
        "h2_regime_metadata":h2_regime_metadata,
        "analysis_h2_dicts":h2_all_analysis_dicts,
        "analysis_metadata":h2_all_metadata,
        "seizure_infos_regime":seizure_infos_regime,
        "seizure_infos_analysis":seizure_infos_analysis,
        "all_channels":all_channels,
        "soz_channels":soz_channels,
        "runs":runs,}


def get_patient_config(patient_id):
    """This function obtains the configuration for a single patient.
    
    Parameters
    ----------
    patient_id
        Internal patient identifier used to select configuration and output paths.
    
    Returns
    -------
    object
        The patient configuration."""
    if patient_id not in PATIENT_CONFIGS:
        raise ValueError(f"Unknown patient_id: {patient_id}")
    return PATIENT_CONFIGS[patient_id]
    
PATIENT_CONFIGS = {
    "pt01": {
        "sub": "pt01",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["01", "02", "03", "04"],
        "soz_channels": [
            "PD1", "PD2", "PD3", "PD4",
            "AD1", "AD2", "AD3", "AD4",
            "ATT1", "ATT2",
        ],
        "exclude_channels": ["$G11", "$G12"],
        "orig_sfreq": 1000,
    },

    "pt02": {
        "sub": "pt2",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["01", "02", "03"],
        "soz_channels": [
            "PST1", "PST2", "PST3", "PST4",
            "AST1", "AST2",
            "MST1", "MST2",
        ],
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },

    "pt03": {
        "sub": "pt3",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["01","02"],
        "soz_channels": [
            "OF1", "OF2", "OF3", "OF4",
            "MFP2", "MFP3",
            "IFP1", "IFP2", "IFP3",
            "SFP1", "SFP2", "SFP3",
            *[f"FG{i}" for i in range(1, 26)],
        ],
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },

    "pt06": {
        "sub": "pt6",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["01","02","03"],
        "soz_channels": [
            "LA1", "LA2", "LA3", "LA4",
            "LAH1", "LAH2", "LAH3", "LAH4",
            "LPH1", "LPH2", "LPH3", "LPH4",
        ],
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },

    "pt07": {
        "sub": "pt7",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["01","02","03"],
        "soz_channels": [
            "LFP3", "MFP1",
            "PT2", "PT3", "PT4", "PT5",
            "MT2", "MT3",
            "AT3", "AT4",
            "G29", "G30", "G39", "G40", "G45", "G46",
        ],
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },

    "pt08": {
        "sub": "pt8",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["01","02","03"],
        "soz_channels": [
            "MST3", "MST4",
            "TO5", "TO6",
            "G22", "G23",
            "G29", "G30", "G31",
            "G19", "G20", "G21", "G28",
            "PO8", "PO9", "PO10",
        ],
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },

    "pt10": {
        "sub": "pt10",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["01","02","03"],
        "soz_channels": [
            "TT1", "TT2", "TT3", "TT4", "TT5", "TT6",
            "AST1", "AST2",
            "MST1", "MST2",
        ],
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },

    "pt12": {
        "sub": "pt12",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["01","02"],
        "soz_channels": [
            "TT1", "TT2", "TT3", "TT4", "TT5",
            "AST1", "AST2",
            "MST1", "MST2",
        ],
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },

    "pt13": {
        "sub": "pt13",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["01","02","03","04"],
        "soz_channels": [
            "G1", "G2", "G9", "G10", "G17", "G18",
        ],
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },

    "pt14": {
        "sub": "pt14",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["01","02","03"],
        "soz_channels": [
            "TT1", "TT2", "TT3",
            "AST1", "AST2",
            "MST1", "MST2",
        ],
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },

    "pt15": {
        "sub": "pt15",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["01","02","03","04"],
        "soz_channels": [
            "TT1", "TT2", "TT3", "TT4",
            "MST1", "MST2",
            "AST1", "AST2", "AST3", "AST4",
            "G2", "G3", "G4", "G5",
            "G10", "G11", "G12", "G13",
        ],
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },

    "pt16": {
        "sub": "pt16",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["01","02","03"],
        "soz_channels": [
            "TT1", "TT2", "TT3", "TT4", "TT5",
            "AST1",
        ],
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },

    "pt17": {
        "sub": "pt17",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["02"],
        "soz_channels": ["TT1", "TT2"],
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },
    "ummc003": {
        "sub": "ummc003",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["01","02","03"],
        "soz_channels": [
            "MEST3", "MEST4", "MEST5",
            "G4", "G10", "G12",
            "G18", "G19", "G20",
            "G26", "G27", "G28",
        ],
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },

    "ummc004": {
        "sub": "ummc004",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["01","02","03"],
        "soz_channels": ["AT1", "AT2"],
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },

    "ummc005": {
        "sub": "ummc005",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["01","02"],
        "soz_channels": [
            "AT1", "AT2", "G17", "G19", "G25", "G27",
        ],
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },

    "ummc006": {
        "sub": "ummc006",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["01","02","03"],
        "soz_channels": [
            "MT1", "MT2", "MT3", "MT4",
            "MesT1", "MesT2", "MesT3", "MesT4", "MesT5",
        ],
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },
    "ummc008": {
        "sub": "ummc008",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["01","02","03"],
        "soz_channels": [
            "G1", "G2", "G3", "G4", "G5",
            "G9", "G10", "G11", "G12", "G13",
            "G17", "G18", "G19", "G20", "G21",
            "AT1", "AT2", "AT3", "AT4",
            "MT1", "MT2", "MT3", "MT4",
        ],
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },

    "ummc009": {
        "sub": "ummc009",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["01","02","03"],
        "soz_channels": [
            "G4", "G5", "G6", "G7", "G12", "G14",
            "PT1", "PT2", "PT3", "PT4",
            "AT1", "AT2", "AT3", "AT4",
        ],
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },

    "jh101": {
        "sub": "jh101",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["01","02","03","04"],
        "soz_channels": [
            "LAT1", "LAT2", "LAT6", "LAT7", "LAH6",
        ],
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },
    "jh103": {
        "sub": "jh103",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["02","03"],    
        "soz_channels": [
            "RTG40", "rTG48",
            "RAD1", "RAD2", "RAD3", "RAD4",
            "RAD5", "RAD6", "RAD7", "RAD8",
            "RHD1", "RHD2", "RHD3", "RHD4", "RHD5",
            "RHD6", "RHD7", "RHD8", "RHD9",
        ],
    
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },
    
    "jh105": {
        "sub": "jh105",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",    
        "runs": ["02","04"],
        "soz_channels": [
            "RPG4", "RPG5",
            "RPG12", "RPG13", "RPG14",
            "RPG20", "RPG21",
            "APD1", "APD2", "APD3", "APD4",
            "APD5", "APD6", "APD7", "APD8",
            "PPD1", "PPD2", "PPD3", "PPD4",
            "PPD5", "PPD6", "PPD7", "PPD8",
            "ASI3",
            "PSI5", "PSI6",
        ],   
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },
    
    "jh108": {
        "sub": "jh108",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",    
        "runs": ["04","05","06"],
        "soz_channels": [
            "RDI1", "RDI2", "RDI3", "RDI4",
            "RSI1", "RSI2", "RSI3", "RSI4",
        ],
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },
    
    "umf001": {
        "sub": "umf001",
        "ses": "presurgery",
        "task": "ictal",
        "acq": "ecog",
        "runs": ["01"],
        "soz_channels": [
            "BTM1", "BTM2", "BTM3", "BTM4", "BTM5", "BTM6",
            "BTP1", "BTP2", "BTP3", "BTP4", "BTP5", "BTP6",
            "C33", "C34",
            "C41", "C42", "C43",
            "C49", "C50", "C51", "C52",
            "C53", "C54", "C55", "C56",
            "C57", "C58", "C59", "C60",
            "C61", "C62", "C63", "C64",
        ],
    
        "exclude_channels": [],
        "orig_sfreq": 1000,
    },

}

SUCCESSFUL_SURGERY = [
    "pt1",
    "pt2",
    "pt3",
    "pt8",
    "pt13",
    "pt15",
    "pt16",
    "pt17",
    "ummc003",
    "ummc004",
    "ummc005",
    "ummc006",
    "ummc008",
    "ummc009",
    "jh105",
    "umf001",
]

UNSUCCESSFUL_SURGERY = [
    "pt6",
    "pt7",
    "pt10",
    "pt12",
    "pt14",
    "jh101",
    "jh103",
    "jh108",
]