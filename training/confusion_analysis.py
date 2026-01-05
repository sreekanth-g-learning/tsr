import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
    cm, y_true, y_pred = compute_confusion_matrix(model, X_test, y_test)
    plot_confusion_matrix(cm, class_names, "Confusion Matrix")
    top_confusions = get_top_confusions(cm, class_names, top_k)
    print_top_confusions(top_confusions)  


# Example usage:# analyze_model_confusions(model_trained, X_test, y_test, class_names, top_k=5) 

if __name__ == "__main__":
    # This part is for demonstration and should be replaced with actual model and data loading
    from tensorflow import keras
    import os
    model_trained = keras.models.load_model('checkpoints/grid_33/best.keras')
    X_test = np.load(os.path.abspath('dataset/Processed/X_test.npy'))
    y_test = np.load(os.path.abspath('dataset/Processed/y_test.npy'))
    # Label Overview
    class_names = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }

    analyze_model_confusions(model_trained, X_test, y_test, class_names, top_k=5)