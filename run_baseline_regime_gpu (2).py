import os
import sys
import pickle
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from thesis_common_gpu import (PREPARED_DIR,PATIENT_ID,BASELINE_DIR,make_output_dirs,run_many_synthetic_seizures,BASE_SEED,)


def main():
    make_output_dirs()

    print("Patient:", PATIENT_ID)
    print("Prepared directory:", PREPARED_DIR)
    print("Baseline directory:", BASELINE_DIR)

    chunk_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    n_runs = int(os.environ.get("N_RUNS", "500"))
    keep_trajectories = int(os.environ.get("KEEP_TRAJECTORIES", "3"))
    gpu_batch_size = int(os.environ.get("GPU_BATCH_SIZE", "128"))

    dt = float(os.environ.get("SIM_DT", "0.001"))
    max_search_time = float(os.environ.get("MAX_SEARCH_TIME", "60.0"))
    k = float(os.environ.get("SIM_K", "0.1"))
    noise_amp = float(os.environ.get("NOISE_AMP", "0.05"))
    sim_device = os.environ.get("SIM_DEVICE", "cuda")
    sim_dtype = os.environ.get("SIM_DTYPE", "float64")
    prepared_env = os.environ.get("PREPARED_FILE")

    if prepared_env:
        input_file = Path(prepared_env)
    else:
        # Fallback for manual/older runs
        candidates = [PREPARED_DIR / f"{PATIENT_ID}_regime_inputs.pkl",PREPARED_DIR / "regime_inputs.pkl",]
        input_file = next((p for p in candidates if p.exists()),candidates[0],)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Could not find prepared input for {PATIENT_ID}: {input_file}"
        )

    print("Prepared file:", input_file)

    with open(input_file, "rb") as f:
        data = pickle.load(f)

    seizure_infos = data.get("seizure_infos_analysis",data["seizure_infos_regime"],)

    if chunk_idx < 0 or chunk_idx >= len(seizure_infos):
        raise IndexError(
            f"chunk_idx={chunk_idx} is invalid; "
            f"found {len(seizure_infos)} chunks."
        )

    info = seizure_infos[chunk_idx]
    chunk_base_seed = BASE_SEED + 100_000 * chunk_idx

    print("=" * 80)
    print("Running fixed-duration GPU baseline simulation")
    print(f"chunk_idx: {chunk_idx}")
    print(f"n_runs: {n_runs}")
    print(f"duration: {max_search_time} s")
    print(f"device: {sim_device}")
    print(f"dtype: {sim_dtype}")
    print(f"GPU batch size: {gpu_batch_size}")
    print(f"chunk_base_seed: {chunk_base_seed}")
    print("=" * 80)

    kept_runs, summary_df = run_many_synthetic_seizures(A=info["A"],sigma=info["sigma"],n_runs=n_runs,chunk_base_seed=chunk_base_seed,dt=dt,max_search_time=max_search_time,k=k,noise_amp=noise_amp,
                                                        stim_nodes=None,stim_strength=0.0,stim_start=0.0,stim_end=float("inf"),keep_trajectories=keep_trajectories,batch_size=gpu_batch_size,
                                                        device=sim_device,dtype=sim_dtype,)

    summary_df["chunk_idx"] = chunk_idx
    summary_df["original_seizure_idx"] = info["original_seizure_idx"]
    summary_df["seizure_number"] = info.get("seizure_number", info["original_seizure_idx"] + 1)
    summary_df["run_name"] = info.get("run", None)
    summary_df["regime"] = info["regime"]
    summary_df["regime_name"] = info["regime_name"]
    summary_df["regime_start_time"] = info.get("regime_start_time", None)
    summary_df["regime_end_time"] = info.get("regime_end_time", None)
    summary_df["n_regime_windows"] = info.get("n_regime_windows", None)
    summary_df["dt"] = dt
    summary_df["max_search_time"] = max_search_time
    summary_df["k"] = k
    summary_df["noise_amp"] = noise_amp
    summary_df["chunk_base_seed"] = chunk_base_seed
    summary_df["sim_device"] = sim_device
    summary_df["sim_dtype"] = sim_dtype
    summary_df["gpu_batch_size"] = gpu_batch_size

    summary_path = BASELINE_DIR / f"baseline_summary_chunk_{chunk_idx:02d}.parquet"
    traj_path = BASELINE_DIR / f"baseline_trajectories_chunk_{chunk_idx:02d}.pkl"
    summary_df.to_parquet(summary_path, index=False)

    with open(traj_path, "wb") as f:
        pickle.dump(
            {
                "chunk_idx": chunk_idx,
                "metadata": {
                    "original_seizure_idx": info["original_seizure_idx"],
                    "seizure_number": info.get("seizure_number", info["original_seizure_idx"] + 1),
                    "run": info.get("run", None),
                    "regime": info["regime"],
                    "regime_name": info["regime_name"],
                    "regime_start_time": info.get("regime_start_time", None),
                    "regime_end_time": info.get("regime_end_time", None),
                    "ch_names": info.get("ch_names", None),
                    "n_runs": n_runs,
                    "keep_trajectories": keep_trajectories,
                    "dt": dt,
                    "max_search_time": max_search_time,
                    "chunk_base_seed": chunk_base_seed,
                    "sim_device": sim_device,
                    "sim_dtype": sim_dtype,
                    "gpu_batch_size": gpu_batch_size,
                },
                "kept_trajectories": kept_runs,
                "runs_light": summary_df.to_dict("records"),
            },
            f,
        )

    print("Saved summary:", summary_path)
    print("Saved trajectories:", traj_path)


if __name__ == "__main__":
    main()
