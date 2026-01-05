from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.callbacks import get_callbacks
import os
import json

def is_resumed(model_name):
    return os.path.exists(
        os.path.join("checkpoints", model_name, "latest.keras")
    )

def get_resume_state(model_name):
    ckpt_path = os.path.join("checkpoints", model_name, "latest.keras")
    meta_path = os.path.join("checkpoints", model_name, "meta.json")

    if not os.path.exists(ckpt_path) or not os.path.exists(meta_path):
        return None, 0

    with open(meta_path, "r") as f:
        last_epoch = json.load(f).get("last_epoch", 0)

    return ckpt_path, last_epoch

def get_latest_checkpoint(model_name):
    ckpt_dir = os.path.join("checkpoints", model_name)
    if not os.path.exists(ckpt_dir):
        return None

    checkpoints = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.endswith(".keras")
    ]

    if not checkpoints:
        return None

    return sorted(checkpoints)[-1]

def train_model(model, X_train, y_train, X_val, y_val, cfg):
    
    resumed = False

    model.compile(
        optimizer=Adam(cfg['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    ckpt_path, initial_epoch = get_resume_state(model.name)

    #latest_ckpt = get_latest_checkpoint(model.name)
    #initial_epoch = 0
    if ckpt_path:
        resumed = True
        print(f"🔁 RESUMING TRAINING")
        print(f"   ├─ Checkpoint : {ckpt_path}")
        print(f"   └─ Start epoch: {initial_epoch}")

        model = keras.models.load_model(ckpt_path)
    aug = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode="nearest")

    return model.fit(
        aug.flow(X_train, y_train, batch_size=32), 
        epochs=cfg['epochs'], 
        initial_epoch=initial_epoch,
        validation_data=(X_val, y_val), 
        callbacks=get_callbacks(model.name),
        verbose=1),resumed
    