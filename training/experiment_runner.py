import yaml
from models.model_factory import build_custom_cnn
from training.train import train_model
from training.evaluate import evaluate_model
from training.gradcam_comparison import compare_gradcam_across_models

from training.confusion_analysis import (
    compute_confusion_matrix,
    plot_confusion_matrix,
    get_top_confusions
)
from training.gradcam_error_analysis import analyze_misclassified_samples

def run_experiments(cfg, train_ds, val_ds, test_ds):
    results = {}

    for name, params in cfg["models"].items():
        print(f"\nTraining model: {name}")

        model = build_custom_cnn(
            input_shape=tuple(cfg["input_shape"]),
            num_classes=cfg["num_classes"],
            conv_filters=params["conv_filters"],
            batchnorm=params.get("batchnorm", False),
            dropout=params.get("dropout", False),
            dropout_rate=params.get("dropout_rate", 0.5)
        )

        train_model(model, train_ds, val_ds, cfg)
        metrics = evaluate_model(model, test_ds)

        results[name] = {
            "accuracy": metrics["test_accuracy"],
            "params": model.count_params(),
            "model": model
        }

        print("\nComparing Grad-CAM across models...")
        
        cm, y_true, y_pred = compute_confusion_matrix(
            model,
            test_ds,
            class_names
        )

        plot_confusion_matrix(
            cm,
            class_names,
            title=f"Confusion Matrix – {model_name}"
        )

        top_confusions = get_top_confusions(cm, class_names)

        print("Top Confused Classes:")
        for item in top_confusions:
            print(item)

        analyze_misclassified_samples(
            model,
            test_ds,
            class_names,
            last_conv_layer="last_conv"
        )
        
    models_for_cam = {
            name: res["model"]
            for name, res in results.items()
        }
    compare_gradcam_across_models(
        models_for_cam,
        cfg["gradcam"]["image_path"],
        last_conv_layer="last_conv_layer"
    )
    return results
