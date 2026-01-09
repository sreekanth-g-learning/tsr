import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from utils.class_labels import get_class_labels

def compute_classwise_metrics(
    y_true,
    y_pred,
    class_names,
    output_csv="logs/classwise_metrics.csv"
):
    """
    Computes per-class Precision, Recall, F1-score, Accuracy
    """

    # Classification report (precision, recall, f1, support)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    rows = []
    for i, class_name in enumerate(class_names):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        class_accuracy = (TP + TN) / cm.sum()

        rows.append({
            "class_id": i,
            "class_name": class_name,
            "precision": report[class_name]["precision"],
            "recall": report[class_name]["recall"],
            "f1_score": report[class_name]["f1-score"],
            "support": report[class_name]["support"],
            "accuracy": class_accuracy
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    return df

if __name__ == "__main__":
    # This part is for demonstration and should be replaced with actual model and data loading
    from tensorflow import keras
    import os
    model_trained = keras.models.load_model('checkpoints/grid_33/best.keras')
    X_test = np.load(os.path.abspath('dataset/Processed/X_test.npy'))
    y_test = np.load(os.path.abspath('dataset/Processed/y_test.npy'))
    # Label Overview
    class_names = [list(get_class_labels().values())]

    compute_classwise_metrics(model_trained, X_test, y_test, class_names)
