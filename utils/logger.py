import os
import csv
from datetime import datetime
import hashlib
import json

LOG_DIR = "logs"
MASTER_LOG = os.path.join(LOG_DIR, "experiments.csv")

def canonical_hparams(params_dict):
    """
    Deterministic JSON representation
    """
    return json.dumps(params_dict, sort_keys=True)

def make_grid_signature(params_dict):
    canonical = canonical_hparams(params_dict)
    return hashlib.md5(canonical.encode()).hexdigest()[:12]

def init_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(MASTER_LOG):
        with open(MASTER_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "model_name",
                "grid_signature",
                "hyperparameters_json",
                "epochs",
                "num_parameters",
                "test_accuracy",
                "resume_status",
                "run_time_sec"
            ])

def log_experiment(
    model_name,
    params_dict,
    epochs,
    num_parameters,
    test_accuracy,
    resumed,
    run_time_sec
):
    with open(MASTER_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            model_name,
            make_grid_signature(params_dict),
            canonical_hparams(params_dict),
            epochs,
            num_parameters,
            round(test_accuracy * 100, 2),
            "RESUMED" if resumed else "FRESH",
            round(run_time_sec, 2)
        ])
