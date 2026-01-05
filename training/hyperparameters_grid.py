import itertools
import yaml
import os
import json
from models.model_factory import build_custom_cnn
from training.train import train_model
from training.evaluate import evaluate_model
from utils.logger import init_logger, log_experiment
from utils.plotter import save_training_plots
import argparse
import numpy as np
from utils.hyperparam_plots import plot_accuracy_vs_params

from concurrent.futures import ProcessPoolExecutor

from utils.logger import make_grid_signature
import pandas as pd
import time

def get_completed_signatures(csv_path="logs/experiments.csv"):
    if not os.path.exists(csv_path):
        return set()

    df = pd.read_csv(csv_path)
    return set(df["grid_signature"].dropna().values)


def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_hyperparameter_grid(X_train, y_train, X_val, y_val, X_test, y_test, max_experiments=None):
    cfg = load_config()
    init_logger()

    # 🔧 Hyperparameter search space
    grid = {
        "conv_filters": [
            [16, 32],
            [32, 64, 128],
            [16, 32, 64, 128]
        ],
        "dropout": [True, False],
        "dropout_rate": [0.3, 0.5],
        "batchnorm": [True, False],
        "learning_rate": [1e-3, 5e-4]
    }

    combinations = list(itertools.product(
        grid["conv_filters"],
        grid["dropout"],
        grid["dropout_rate"],
        grid["batchnorm"],
        grid["learning_rate"]
    ))

    print(f"🔍 Total experiments: {len(combinations)}")

    best_acc = 0
    best_model = None
    best_config = None
    completed = get_completed_signatures()
    for idx, (filters, dropout, dr, bn, lr) in enumerate(combinations, 1):
        model_name = f"grid_{idx}"

        print(f"\n▶ Experiment {idx}/{len(combinations)}")
        params_dict = {
            "conv_filters": filters,
            "batchnorm": bn,
            "dropout": dropout,
            "dropout_rate": dr,
            "learning_rate": lr
        }
        signature = make_grid_signature(params_dict)
        if signature in completed:
            print(f"⏭️ Skipping completed grid: {signature}")
            continue
        print({
            "filters": filters,
            "dropout": dropout,
            "dropout_rate": dr,
            "batchnorm": bn,
            "lr": lr
        })

        model = build_custom_cnn(
            input_shape=tuple(cfg["input_shape"]),
            num_classes=cfg["num_classes"],
            conv_filters=filters,
            use_batchnorm=bn,
            use_dropout=dropout,
            dropout_rate=dr,
            model_name=model_name
        )


        # override LR per experiment
        local_cfg = cfg.copy()
        local_cfg["learning_rate"] = lr

        start_time = time.perf_counter()

        history,resumed = train_model(model, X_train, y_train, X_val, y_val, local_cfg)
        acc = evaluate_model(model, X_test, y_test)

        end_time = time.perf_counter()
        run_time_sec = end_time - start_time

        save_training_plots(history, model_name)

        log_experiment(
            model_name=model_name,
            params_dict=params_dict,
            epochs=cfg["epochs"],
            test_accuracy=acc,
            num_parameters=model.count_params(),           
            resumed=resumed,
            run_time_sec=run_time_sec
        )

        # Track best model
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_config = {
                "conv_filters": filters,
                "dropout": dropout,
                "dropout_rate": dr,
                "batchnorm": bn,
                "learning_rate": lr
            }

    # Save best model + config
    if best_model:
        best_model.save("output/best_grid_model.keras")
        with open("logs/best_grid_config.json", "w") as f:
            json.dump(best_config, f, indent=4)

        print("\n🏆 Best Grid Model Saved")
        print(best_config)

def run_single_experiment(args):
    (idx, filters, dropout, dr, bn, lr, cfg, X_train, y_train, X_val, y_val, X_test, y_test) = args

    model_name = f"grid_{idx}"

    model = build_custom_cnn(
        input_shape=tuple(cfg["input_shape"]),
        num_classes=cfg["num_classes"],
        conv_filters=filters,
        use_batchnorm=bn,
        use_dropout=dropout,
        dropout_rate=dr,
        model_name=model_name
    )

    local_cfg = cfg.copy()
    local_cfg["learning_rate"] = lr

    history = train_model(model, X_train, y_train, X_val, y_val, local_cfg)
    acc = evaluate_model(model, X_test, y_test)

    save_training_plots(history, model_name)
    log_experiment(model_name, acc, model.count_params(), cfg["epochs"])

    return {
        "model": model_name,
        "accuracy": acc,
        "filters": filters,
        "dropout": dropout,
        "dropout_rate": dr,
        "batchnorm": bn,
        "lr": lr
    }

def run_parallel_grid(X_train, y_train, X_val, y_val, X_test, y_test, max_workers=2):
    cfg = load_config()
    init_logger()

    grid = {
        "conv_filters": [[16,32], [32,64,128], [16,32,64,128]],
        "dropout": [True, False],
        "dropout_rate": [0.3, 0.5],
        "batchnorm": [True, False],
        "learning_rate": [1e-3, 5e-4]
    }

    combos = list(itertools.product(
        grid["conv_filters"],
        grid["dropout"],
        grid["dropout_rate"],
        grid["batchnorm"],
        grid["learning_rate"]
    ))
    print(f"🔍 Total experiments: {len(combos)}")
    args = [
        (i+1, *combo, cfg, X_train, y_train, X_val, y_val, X_test, y_test)
        for i, combo in enumerate(combos)
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_single_experiment, args))

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_experiments", type=int, default=None,
                        help="Limit number of grid experiments")
    args = parser.parse_args()

     # Load preprocessed data
    X_train = np.load(os.path.abspath('dataset/Processed/X_train.npy'))
    y_train = np.load(os.path.abspath('dataset/Processed/y_train.npy'))

    X_val = np.load(os.path.abspath('dataset/Processed/X_val.npy'))
    y_val = np.load(os.path.abspath('dataset/Processed/y_val.npy'))

    X_test = np.load(os.path.abspath('dataset/Processed/X_test.npy'))
    y_test = np.load(os.path.abspath('dataset/Processed/y_test.npy'))

    run_hyperparameter_grid(X_train, y_train, X_val, y_val, X_test, y_test, args.max_experiments)
    #run_parallel_grid(X_train, y_train, X_val, y_val, X_test, y_test, max_workers=4)
    plot_accuracy_vs_params()

if __name__ == "__main__":
    main()