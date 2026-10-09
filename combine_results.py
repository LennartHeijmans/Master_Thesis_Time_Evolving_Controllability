from pathlib import Path
import os
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT","/home/lheijmans/Master_Thesis",))

PATIENT_ID = os.environ["PATIENT_ID"]

RESULTS_DIR = PROJECT_ROOT / "Results" / PATIENT_ID
BASELINE_DIR = RESULTS_DIR / "Baseline"
STIM_DIR = RESULTS_DIR / "Stim"
COMBINED_DIR = RESULTS_DIR / "Combined"


def main():
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)

    combined_dir = COMBINED_DIR

    print(f"Patient: {PATIENT_ID}")
    print(f"Baseline directory: {BASELINE_DIR}")
    print(f"Stimulation directory: {STIM_DIR}")
    print(f"Combined directory: {COMBINED_DIR}")

    baseline_files = sorted(BASELINE_DIR.glob("baseline_summary_chunk_*.parquet"))

    stim_files = sorted(STIM_DIR.glob("stim_summary_chunk_*_node_*.parquet"))

    print(f"Found {len(baseline_files)} baseline files")
    print(f"Found {len(stim_files)} stimulation files")

    baseline_summary = None
    stim_summary = None
    
    if baseline_files:
        baseline_all = pd.concat([pd.read_parquet(f) for f in baseline_files],ignore_index=True,)

        baseline_all.to_parquet(combined_dir / "baseline_all.parquet",index=False,)

        baseline_summary = (baseline_all.groupby(["chunk_idx","original_seizure_idx","seizure_number","run_name","regime","regime_name",],as_index=False,).agg(
                onset_rate=("first_onset_time", lambda x: x.notna().mean()),
                global_rate=("global_onset_time", lambda x: x.notna().mean()),
                mean_n_switched=("n_switched", "mean"),
                median_n_switched=("n_switched", "median"),
                mean_first_onset_time=("first_onset_time", "mean"),
                mean_last_onset_time=("last_onset_time", "mean"),
                mean_total_recruitment_time=("total_recruitment_time", "mean"),
                mean_global_time=("global_onset_time", "mean"),
                n_runs=("run", "count"),))

        baseline_summary.to_parquet(combined_dir / "baseline_summary.parquet",index=False,)

        print("Saved baseline_all:", len(baseline_all))
        print("Saved baseline_summary:", len(baseline_summary))
    else:
        print("No baseline files found.")

    if stim_files:
        stim_all = pd.concat([pd.read_parquet(f) for f in stim_files],ignore_index=True,)

        stim_all.to_parquet(combined_dir / "stim_all.parquet",index=False,)

        stim_summary = (stim_all.groupby(["chunk_idx","original_seizure_idx","seizure_number","run_name","regime","regime_name","stim_node","stim_channel",],as_index=False,).agg(
                onset_rate=("first_onset_time", lambda x: x.notna().mean()),
                global_rate=("global_onset_time", lambda x: x.notna().mean()),
                mean_n_switched=("n_switched", "mean"),
                median_n_switched=("n_switched", "median"),
                mean_first_onset_time=("first_onset_time", "mean"),
                mean_last_onset_time=("last_onset_time", "mean"),
                mean_total_recruitment_time=("total_recruitment_time", "mean"),
                mean_global_time=("global_onset_time", "mean"),
                n_runs=("run", "count"),))

        stim_summary.to_parquet(combined_dir / "stim_summary.parquet",index=False,)

        print("Saved stim_all:", len(stim_all))
        print("Saved stim_summary:", len(stim_summary))
    else:
        print("No stimulation files found.")
    if baseline_summary is not None and stim_summary is not None:
        comparison_keys = ["chunk_idx","original_seizure_idx","seizure_number","run_name","regime","regime_name",]
    
        baseline_for_merge = baseline_summary[comparison_keys+ ["mean_n_switched","median_n_switched","onset_rate","global_rate","mean_first_onset_time",]].rename(
            columns={"mean_n_switched": "baseline_mean_n_switched","median_n_switched": "baseline_median_n_switched","onset_rate": "baseline_onset_rate",
            "global_rate": "baseline_global_rate","mean_first_onset_time":"baseline_mean_first_onset_time",})
    
        stim_comparison = stim_summary.merge(baseline_for_merge,on=comparison_keys,how="left",validate="many_to_one",)
    
        stim_comparison["reduced_n_switches"] = (stim_comparison["baseline_mean_n_switched"]- stim_comparison["mean_n_switched"])
    
        stim_comparison["proportional_reduction_n_switches"] = (stim_comparison["reduced_n_switches"]/ stim_comparison["baseline_mean_n_switched"]).where(stim_comparison["baseline_mean_n_switched"] > 0)
    
        stim_comparison["onset_rate_reduction"] = (stim_comparison["baseline_onset_rate"]- stim_comparison["onset_rate"])
    
        stim_comparison["global_rate_reduction"] = (stim_comparison["baseline_global_rate"]- stim_comparison["global_rate"])
    
        stim_comparison["first_onset_delay"] = (stim_comparison["mean_first_onset_time"]- stim_comparison["baseline_mean_first_onset_time"])
    
        stim_comparison.to_parquet(combined_dir / "stim_vs_baseline_summary.parquet",index=False,)
    
        print("Saved stim_vs_baseline_summary:",len(stim_comparison))


if __name__ == "__main__":
    main()