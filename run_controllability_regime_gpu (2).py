import os
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path

from thesis_common_gpu import (
    PATIENT_ID,
    PREPARED_DIR,
    CONTROLLABILITY_DIR,
    make_output_dirs,
    fit_LSSM_and_compute_controllability,
    compute_scheid_controllability,
    run_kashima_on_all_scheid_h2_ec,
)


def timed_stage(name, function, *args, **kwargs):
    print("\n" + "=" * 80, flush=True)
    print(f"STARTING: {name}", flush=True)
    print("=" * 80, flush=True)
    start = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print("\n" + "-" * 80, flush=True)
    print(f"FINISHED: {name} in {elapsed / 60:.2f} minutes", flush=True)
    print("-" * 80, flush=True)
    return result


def assert_unique_columns(df, name):
    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(
            f"{name} contains duplicate columns: {duplicated}\n"
            f"All columns: {df.columns.tolist()}"
        )
    print(f"{name}: unique column names OK.", flush=True)


def parquet_safety_test(df, name):
    assert_unique_columns(df, name)
    test_path = CONTROLLABILITY_DIR / f"_TEST_{name}.parquet"
    try:
        df.to_parquet(test_path, index=False)
        check = pd.read_parquet(test_path)
        if len(check) != len(df):
            raise RuntimeError(
                f"{name}: parquet round-trip row mismatch {len(df)} -> {len(check)}"
            )
        print(
            f"{name}: parquet write/read test PASSED "
            f"({len(df)} rows, {len(df.columns)} columns).",
            flush=True,
        )
    finally:
        if test_path.exists():
            test_path.unlink()


def save_pickle(obj, path, label):
    print(f"Saving {label} pickle -> {path}", flush=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path, "rb") as f:
        pickle.load(f)
    print(f"{label} pickle saved and verified.", flush=True)


def merge_ding_metadata(df_ding_nodes, ding_metadata):
    df = df_ding_nodes.copy()
    meta = ding_metadata.copy().reset_index(drop=True)

    meta.insert(0, "seizure_idx", np.arange(len(meta), dtype=int))

    if "seizure_idx" not in df.columns:
        raise KeyError(
            "Ding node dataframe has no seizure_idx: "
            f"{df.columns.tolist()}"
        )

    overlaps = [c for c in meta.columns if c != "seizure_idx" and c in df.columns]
    if overlaps:
        print("Ding metadata overlaps; keeping metadata versions:", overlaps, flush=True)
        df = df.drop(columns=overlaps)

    df = df.merge(meta,on="seizure_idx",how="left",validate="many_to_one",)
    df = df.drop(columns=["seizure_idx"])

    if "chunk_idx" not in df.columns:
        raise KeyError("Ding merge did not produce chunk_idx from ding_metadata.")

    assert_unique_columns(df, "Ding dataframe")
    return df


def merge_chunk_metadata(df_nodes, metadata, name):
    df = df_nodes.copy()
    meta = metadata.copy().reset_index(drop=True)

    if "seizure_idx" in df.columns:
        df = df.rename(columns={"seizure_idx": "chunk_idx"})

    if "chunk_idx" not in df.columns:
        raise KeyError(
            f"{name}: expected seizure_idx or chunk_idx in {df.columns.tolist()}"
        )

    if "chunk_idx" not in meta.columns:
        meta.insert(0, "chunk_idx", np.arange(len(meta), dtype=int))

    overlaps = [c for c in meta.columns if c != "chunk_idx" and c in df.columns]
    if overlaps:
        print(f"{name} metadata overlaps; keeping metadata versions:", overlaps, flush=True)
        df = df.drop(columns=overlaps)

    df = df.merge(meta,on="chunk_idx",how="left",validate="many_to_one",)

    assert_unique_columns(df, f"{name} dataframe")
    return df


def main():
    total_start = time.perf_counter()
    make_output_dirs()

    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    sim_device = os.environ.get("SIM_DEVICE", "cuda")
    sim_dtype = os.environ.get("SIM_DTYPE", "float64")
    kashima_batch_size = int(os.environ.get("KASHIMA_BATCH_SIZE", "2048"))
    kashima_realizations = int(os.environ.get("KASHIMA_REALIZATIONS", "50"))
    save_kashima_gramians = os.environ.get("SAVE_KASHIMA_GRAMIANS", "0") == "1"

    prepared_env = os.environ.get("PREPARED_FILE")
    
    if prepared_env:
        input_file = Path(prepared_env)
    else:
        candidates = [PREPARED_DIR / f"{PATIENT_ID}_regime_inputs.pkl",PREPARED_DIR / "regime_inputs.pkl",]
        input_file = next((p for p in candidates if p.exists()), candidates[0])
    
    if not input_file.exists():raise FileNotFoundError(f"Could not find prepared input for {PATIENT_ID}: {input_file}")
    
    print("Prepared file:", input_file)
    
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    all_channels = data["all_channels"]
    soz_channels = data["soz_channels"]

    ding_feature_maps = data.get("ding_analysis_feature_maps", data["ding_regime_feature_maps"])
    ding_metadata = data.get("ding_analysis_metadata", data["ding_regime_metadata"])
    h2_dicts = data.get("analysis_h2_dicts", data["h2_regime_dicts"])
    metadata = data.get("analysis_metadata", data["h2_regime_metadata"])

    print("Ding analysis items:", len(ding_feature_maps), flush=True)
    print("h2/Kashima chunks:", len(h2_dicts), flush=True)
    print("Channels:", len(all_channels), flush=True)
    print("Ding n_jobs:", n_jobs, flush=True)
    print("Kashima device:", sim_device, flush=True)
    print("Kashima dtype:", sim_dtype, flush=True)
    print("Kashima realizations/node:", kashima_realizations, flush=True)
    print("Kashima batch size:", kashima_batch_size, flush=True)
    print("Save Kashima gramians:", save_kashima_gramians, flush=True)

    ding_results, df_ding_nodes_raw, _ = timed_stage("Ding regime controllability",fit_LSSM_and_compute_controllability,feature_maps=ding_feature_maps,ch_names=all_channels,
                                                    soz_channels=soz_channels,method_name="ding_lssm_regime",lambdas=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],horizon=100,reg=1e-4,
                                                    max_eig_tol=0.95,n_jobs=n_jobs,)

    print("\nPreparing Ding dataframe safely...", flush=True)
    df_ding_nodes = merge_ding_metadata(df_ding_nodes_raw, ding_metadata)
    print("Ding final columns:", df_ding_nodes.columns.tolist(), flush=True)

    #Quick sanity test
    parquet_safety_test(df_ding_nodes, "ding_regime_nodes")

    ding_nodes_path = CONTROLLABILITY_DIR / "ding_regime_nodes.parquet"
    ding_results_path = CONTROLLABILITY_DIR / "ding_regime_results.pkl"
    df_ding_nodes.to_parquet(ding_nodes_path, index=False)
    save_pickle(ding_results, ding_results_path, "Ding")
    print("DING IS SAFELY ON DISK.", flush=True)

    scheid_results, df_scheid_nodes_raw, _ = timed_stage("Scheid regime controllability",compute_scheid_controllability,h2_dicts,ch_names=all_channels,soz_channels=soz_channels,ec_key="ec",
                                                        time_key="window_times",method_name="scheid_h2_regime",)

    print("\nPreparing Scheid dataframe safely...", flush=True)
    df_scheid_nodes = merge_chunk_metadata(df_scheid_nodes_raw, metadata, "Scheid")
    print("Scheid final columns:", df_scheid_nodes.columns.tolist(), flush=True)
    parquet_safety_test(df_scheid_nodes, "scheid_regime_nodes")

    scheid_nodes_path = CONTROLLABILITY_DIR / "scheid_regime_nodes.parquet"
    scheid_results_path = CONTROLLABILITY_DIR / "scheid_regime_results.pkl"
    df_scheid_nodes.to_parquet(scheid_nodes_path, index=False)
    save_pickle(scheid_results, scheid_results_path, "Scheid")
    print("SCHEID IS SAFELY ON DISK.", flush=True)

    kashima_regime_ec_all = [d["ec"] for d in h2_dicts]

    kashima_results = timed_stage("GPU Kashima regime controllability",run_kashima_on_all_scheid_h2_ec,ec_all=kashima_regime_ec_all,ch_names=all_channels,band_center_hz=60.0,sigma_val=-0.10,
    alpha_val=1.0,beta_val=1.0,k=1.0,dt=0.001,steps=5000,noise_amp=0.10,num_realizations=kashima_realizations,n_jobs=n_jobs,random_seed=42,start_point="equilibrium",
    batch_size=kashima_batch_size,device=sim_device,dtype=sim_dtype,return_gramians=save_kashima_gramians,print_chunk_progress=True,)

    kashima_results_path = CONTROLLABILITY_DIR / "kashima_regime_results.pkl"
    save_pickle(kashima_results, kashima_results_path, "Kashima")
    print("KASHIMA RAW RESULTS ARE SAFELY ON DISK BEFORE PARQUET FORMATTING.",flush=True,)

    print("\nPreparing Kashima dataframe safely...", flush=True)
    df_kashima = merge_chunk_metadata(kashima_results["df"], metadata, "Kashima")
    print("Kashima final columns:", df_kashima.columns.tolist(), flush=True)
    parquet_safety_test(df_kashima, "kashima_regime_nodes")

    kashima_nodes_path = CONTROLLABILITY_DIR / "kashima_regime_nodes.parquet"
    df_kashima.to_parquet(kashima_nodes_path, index=False)
    print("KASHIMA PARQUET IS SAFELY ON DISK.", flush=True)
    expected = [ding_nodes_path,ding_results_path,scheid_nodes_path,scheid_results_path,kashima_nodes_path,kashima_results_path,]

    print("\n" + "=" * 80, flush=True)
    print("FINAL OUTPUT VALIDATION", flush=True)
    print("=" * 80, flush=True)

    for path in expected:
        if not path.exists():
            raise FileNotFoundError(path)
        print(
            f"OK  {path.name}  ({path.stat().st_size / 1024**2:.2f} MB)",
            flush=True,
        )

    total_elapsed = time.perf_counter() - total_start
    print("\nALL CONTROLLABILITY OUTPUTS SAVED SUCCESSFULLY.", flush=True)
    print(f"Total runtime: {total_elapsed / 60:.2f} minutes", flush=True)


if __name__ == "__main__":
    main()
