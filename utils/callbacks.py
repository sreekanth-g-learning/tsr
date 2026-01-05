import tensorflow as tf
import os
import json

import time
import csv
import os
import tensorflow as tf

class EpochTimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.start_time = None
        self.epoch_start = None
        self.total_time = 0.0

        self.log_path = os.path.join(
            "logs", model_name, "epoch_times.csv"
        )
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Create CSV header once
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "epoch_time_sec",
                    "cumulative_time_sec",
                    "loss",
                    "accuracy",
                    "val_loss",
                    "val_accuracy"
                ])

    def on_train_begin(self, logs=None):
        self.start_time = time.perf_counter()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.perf_counter() - self.epoch_start
        self.total_time = time.perf_counter() - self.start_time

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                round(epoch_time, 3),
                round(self.total_time, 3),
                round(logs.get("loss", 0), 4),
                round(logs.get("accuracy", 0), 4),
                round(logs.get("val_loss", 0), 4),
                round(logs.get("val_accuracy", 0), 4),
            ])


class EpochTracker(tf.keras.callbacks.Callback):
    def __init__(self, model_name):
        super().__init__()
        self.meta_path = os.path.join("checkpoints", model_name, "meta.json")

    def on_epoch_end(self, epoch, logs=None):
        with open(self.meta_path, "w") as f:
            json.dump({"last_epoch": epoch + 1}, f)


def get_callbacks(model_name):
    ckpt_dir = os.path.join("checkpoints", model_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    return [
                # 🔁 Latest checkpoint (resume)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, "latest.keras"),
            monitor="val_loss",
            save_best_only=False,
            verbose=0
        ),
        # 🏆 Best checkpoint (deployment)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, "best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        ),
        # 🧠 Epoch tracker
        EpochTracker(model_name),
        # ⏱ Epoch timing logger
        EpochTimingCallback(model_name),
        # ⏹ Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        # 📉 LR scheduling
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]
