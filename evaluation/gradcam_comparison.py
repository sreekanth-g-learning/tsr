import math
import textwrap
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.grad_cam import make_gradcam_heatmap
from utils.visualization import overlay_gradcam, save_and_display_gradcam
import os
import PIL.Image as pilimage

'''
Compares Grad-CAM heatmaps across multiple models for a given input image.
inputs: 
- models_dict: A dictionary where keys are model names and values are tf.keras Model objects.
- img_path: Path to the input image file.
- last_conv_layer: Name of the last convolutional layer in the models.
'''
def compare_gradcam_across_models(
    models_dict,
    img_path,
    last_conv_layer="last_conv"
):
    d = os.path.join("logs", 'grad_cam_comparison')
    os.makedirs(d, exist_ok=True)
    #original_img = cv2.imread(img_path)
    #original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    #img_resized = cv2.resize(original_img, (30, 30))
    #img_array = np.expand_dims(img_resized / 255.0, axis=0)

    test_data=[]       
    test_image = cv2.imread(img_path)
    test_image_fromarray = pilimage.fromarray(test_image, 'RGB')
    test_resize_image = test_image_fromarray.resize((30, 30))
    test_data.append(np.array(test_resize_image))
        
    img_array = np.array(test_data)
    img_array = img_array/255

    num_models = len(models_dict)
    max_cols = 4
    cols = min(max_cols, num_models)
    rows = math.ceil(num_models / max_cols)

    plt.figure(figsize=(5 * cols, 5 * rows))

    for idx, (name, model) in enumerate(models_dict.items()):
        # extract the last conv layer name from the model architecture starting with conv2d_ instead of from parameter
        # get last layer name starting with conv2d_ 
        last_conv_layer = None
        for layer in model.layers:
            if layer.name.startswith("conv2d_"):
                last_conv_layer = layer.name  

        if last_conv_layer is None:
            raise ValueError(f"No Conv2D layer found in model: {name}")             
        
        print(f"Using last conv layer: {last_conv_layer} for model: {name}")

        heatmap,pred_clas = make_gradcam_heatmap(           
            img_array= img_array,
             model = model,  
             last_conv_layer_name=last_conv_layer
        )
        overlay = save_and_display_gradcam(img_path, heatmap)

        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(overlay)
        ax.axis("off")

        # Wrap long model names
        wrapped_title = "\n".join(textwrap.wrap(name, width=18))
        ax.set_title(wrapped_title, fontsize=10, pad=6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(d, "gradcam_comparison.png"), dpi=300, bbox_inches="tight")
    plt.show()

# Example usage:
# load all models with wtest accuracy > 90% from 'checkpoints' directory and compare their Grad-CAM heatmaps , get test accuracy from experiments.csv under logs directory

if __name__ == "__main__":
    import pandas as pd
    from tensorflow import keras

    models_dict = {}
    logs_dir = "logs"
    checkpoints_dir = "checkpoints"

    experiments_df = pd.read_csv(os.path.join(logs_dir, "experiments.csv"))

    for _, row in experiments_df.iterrows():
        accuracy = row['test_accuracy']
        best_model_name = "grid_33"  # Default model name
        model_path = os.path.join(checkpoints_dir, row['model_name'], 'best.keras')
        if  accuracy >= 98.0 and os.path.exists(model_path):
            model_name = f"{row['model_name']}_acc_{accuracy:.2f}"
            model = keras.models.load_model(model_path)
            models_dict[model_name] = model

    #test_image_path = "dataset/raw/gtsrb/Test/00117.png"  
    test_image_path = "dataset/raw/gtsrb/Train/29/00029_00008_00029.png"
    #test_image_path = "dataset/raw/gtsrb/Test/00149.png"
    if not os.path.exists(test_image_path) :
        raise FileNotFoundError(f"Test image not found at {test_image_path}")
   
   
    compare_gradcam_across_models(
        models_dict,
        test_image_path,
        last_conv_layer="conv2d_12"  # Adjust based on your model architecture  
    )