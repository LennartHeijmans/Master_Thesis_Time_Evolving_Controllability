import os
import sys
import pickle
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from thesis_common_gpu import (PATIENT_ID,PREPARED_DIR,STIM_DIR,make_output_dirs,run_stimulation,BASE_SEED,)

def main():
    make_output_dirs()

    print("Patient:", PATIENT_ID)
    print("Prepared directory:", PREPARED_DIR)
    print("Stim directory:", STIM_DIR)

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    n_runs_per_node = int(os.environ.get("N_RUNS_PER_NODE", os.environ.get("N_RUNS", "500")))
    stim_strength = float(os.environ.get("STIM_STRENGTH", "1.0"))
    gpu_batch_size = int(os.environ.get("GPU_BATCH_SIZE", "128"))

    dt = float(os.environ.get("SIM_DT", "0.001"))
    max_search_time = float(os.environ.get("MAX_SEARCH_TIME", "60.0"))
    k = float(os.environ.get("SIM_K", "0.2"))
    noise_amp = float(os.environ.get("NOISE_AMP", "0.05"))
    sim_device = os.environ.get("SIM_DEVICE", "cuda")
    sim_dtype = os.environ.get("SIM_DTYPE", "float64")

    prepared_env = os.environ.get("PREPARED_FILE")

    if prepared_env:
        input_file = Path(prepared_env)
    else:
        candidates = [PREPARED_DIR / f"{PATIENT_ID}_regime_inputs.pkl",PREPARED_DIR / "regime_inputs.pkl",]
        input_file = next((p for p in candidates if p.exists()), candidates[0])

    if not input_file.exists():
        raise FileNotFoundError(
            f"Could not find prepared input for {PATIENT_ID}: {input_file}"
        )

    print("Prepared file:", input_file)

    with open(input_file, "rb") as f:
        data = pickle.load(f)

    seizure_infos = data.get(
        "seizure_infos_analysis",
        data["seizure_infos_regime"]
    )

    node_counts = [len(info.get("ch_names", info["sigma"])) for info in seizure_infos]
    unique_node_counts = sorted(set(node_counts))
    if len(unique_node_counts) != 1:
        raise ValueError(f"Expected a constant node count, found {unique_node_counts}.")

    n_nodes = unique_node_counts[0]
    n_chunks = len(seizure_infos)
    n_tasks = n_chunks * n_nodes
    if task_id < 0 or task_id >= n_tasks:
        raise IndexError(f"task_id={task_id} is invalid; valid range is 0..{n_tasks - 1}.")

    chunk_idx = task_id // n_nodes
    stim_node = task_id % n_nodes
    info = seizure_infos[chunk_idx]
    A = info["A"]
    sigma = info["sigma"]
    ch_names = info.get("ch_names", [f"node_{i}" for i in range(len(sigma))])

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got {A.shape}.")
    if A.shape[0] != len(sigma) or len(ch_names) != len(sigma):
        raise ValueError("A, sigma and channel names have incompatible node counts.")

    chunk_base_seed = BASE_SEED + 100_000 * chunk_idx

    print("=" * 80)
    print("Running fixed-duration GPU stimulation simulation")
    print(f"task_id: {task_id}")
    print(f"chunk_idx: {chunk_idx}")
    print(f"stim_node: {stim_node} ({ch_names[stim_node]})")
    print(f"n_runs_per_node: {n_runs_per_node}")
    print(f"duration: {max_search_time} s")
    print(f"device: {sim_device}")
    print(f"dtype: {sim_dtype}")
    print(f"GPU batch size: {gpu_batch_size}")
    print(f"chunk_base_seed: {chunk_base_seed}")
    print("=" * 80)

    stim_df = run_stimulation(A=A,sigma=sigma,stim_strength=stim_strength,n_runs_per_node=n_runs_per_node,chunk_base_seed=chunk_base_seed,dt=dt,max_search_time=max_search_time,k=k,
                              noise_amp=noise_amp,stim_start=0.0,stim_end=float("inf"),stim_nodes=[stim_node],batch_size=gpu_batch_size,device=sim_device,dtype=sim_dtype,)

    stim_df["task_id"] = task_id
    stim_df["chunk_idx"] = chunk_idx
    stim_df["original_seizure_idx"] = info["original_seizure_idx"]
    stim_df["seizure_number"] = info.get("seizure_number", info["original_seizure_idx"] + 1)
    stim_df["run_name"] = info.get("run", None)
    stim_df["regime"] = info["regime"]
    stim_df["regime_name"] = info["regime_name"]
    stim_df["regime_start_time"] = info.get("regime_start_time", None)
    stim_df["regime_end_time"] = info.get("regime_end_time", None)
    stim_df["n_regime_windows"] = info.get("n_regime_windows", None)
    stim_df["stim_channel"] = ch_names[stim_node]
    stim_df["stim_strength"] = stim_strength
    stim_df["dt"] = dt
    stim_df["max_search_time"] = max_search_time
    stim_df["k"] = k
    stim_df["noise_amp"] = noise_amp
    stim_df["chunk_base_seed"] = chunk_base_seed
    stim_df["sim_device"] = sim_device
    stim_df["sim_dtype"] = sim_dtype
    stim_df["gpu_batch_size"] = gpu_batch_size

    out_path = STIM_DIR / f"stim_summary_chunk_{chunk_idx:02d}_node_{stim_node:03d}.parquet"
    stim_df.to_parquet(out_path, index=False)
    print("Saved stimulation summary:", out_path)


if __name__ == "__main__":
    main()
