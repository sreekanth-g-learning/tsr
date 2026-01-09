import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd
from utils.class_labels import get_class_labels

def compute_classwise_metrics(
    cm,
    y_true,
    y_pred,
    class_names,
    output_csv="logs/bestmodel/classwise_metrics.csv"
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

def compute_confusion_matrix(model, X_test, y_test):
    y_true = []
    y_pred = []

    preds = model.predict(X_test, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    return cm, y_true, y_pred


def plot_confusion_matrix(cm, class_names, title):
    fig, ax = plt.subplots(figsize=(10, 10))
    d = os.path.join("logs", 'best_model')
    os.makedirs(d, exist_ok=True)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=90)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(d,"confusion_matrix.png")); 
    plt.show()

def get_top_confusions(cm, class_names, top_k=5):
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)

    flat_indices = np.argsort(cm_copy.flatten())[::-1]
    results = []

    for idx in flat_indices[:top_k]:
        i, j = divmod(idx, cm.shape[1])
        results.append({
            "true_class": class_names[i],
            "predicted_class": class_names[j],
            "count": cm[i, j]
        })

    return results


def print_top_confusions(confusions):
    print("Top Confusions:")
    for confusion in confusions:
        print(f"True: {confusion['true_class']}, Predicted: {confusion['predicted_class']}, Count: {confusion['count']}") 

def analyze_model_confusions(model, X_test, y_test, class_names, top_k=5):
    d = os.path.join("logs", 'best_model')
    os.makedirs(d, exist_ok=True)
    cm, y_true, y_pred = compute_confusion_matrix(model, X_test, y_test)
    plot_confusion_matrix(cm, class_names, "Confusion Matrix")
    compute_classwise_metrics(cm, y_true, y_pred, class_names, output_csv="logs/best_model/classwise_metrics.csv")
    top_confusions = get_top_confusions(cm, class_names, top_k)
    print_top_confusions(top_confusions)  


# Example usage:# analyze_model_confusions(model_trained, X_test, y_test, class_names, top_k=5) 

if __name__ == "__main__":
    # This part is for demonstration and should be replaced with actual model and data loading
    from tensorflow import keras
    import os
    model_trained = keras.models.load_model('checkpoints/grid_42/best.keras')
    X_test = np.load(os.path.abspath('dataset/Processed/X_test.npy'))
    y_test = np.load(os.path.abspath('dataset/Processed/y_test.npy'))
    # Label Overview
    class_names = list(get_class_labels().values())
    
    analyze_model_confusions(model_trained, X_test, y_test, class_names, top_k=10)