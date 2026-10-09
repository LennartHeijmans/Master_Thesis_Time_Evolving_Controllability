
import os
import pickle

from thesis_common_gpu import (BIDS_ROOT,PATIENT_ID,PREPARED_DIR,make_output_dirs,prepare_patient_regime_inputs,)

def main():
    patient_id = os.environ.get("PATIENT_ID")

    if patient_id is None:
        raise RuntimeError("PATIENT_ID is not set. ""Example: export PATIENT_ID=pt01")

    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK","4",))

    make_output_dirs()

    print(f"Preparing patient: {patient_id}")
    regime_inputs = (prepare_patient_regime_inputs(patient_id=patient_id,bids_root=BIDS_ROOT,n_jobs=n_jobs,))

    output_path = (PREPARED_DIR/ f"{patient_id}_regime_inputs.pkl")

    with open(output_path,"wb",) as f:
        pickle.dump(regime_inputs,f,protocol=pickle.HIGHEST_PROTOCOL,)

    summary_path = (PREPARED_DIR/ f"{patient_id}_regime_summary_h2_ictal.csv")

    regime_inputs["regime_summary_h2_ictal"].to_csv(summary_path,index=False,)

    print("Saved:",output_path,)


if __name__ == "__main__":
    main()