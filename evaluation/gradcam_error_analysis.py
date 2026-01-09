
import numpy as np
import cv2
import matplotlib.pyplot as plt
from models.grad_cam import generate_gradcam
from utils.visualization import overlay_gradcam
from models.grad_cam import make_gradcam_heatmap
from utils.visualization import overlay_gradcam, save_and_display_gradcam,get_superimposed_image
import pandas as pd
from utils.class_labels import get_class_labels
from PIL import Image
import keras

def analyze_misclassified_samples(
    model,
    X_test, 
    y_test,
    class_names,
    max_samples=5
):
    count = 0
    last_conv_layer = None
    for layer in model.layers:
        if layer.name.startswith("conv2d_"):
            last_conv_layer = layer.name  
    
    images = X_test
    labels = y_test

    
    preds = model.predict(X_test, verbose=0)

    for i in range(len(images)):
        true_cls = np.argmax(labels[i])
        pred_cls = np.argmax(preds[i])

        if true_cls != pred_cls:
            img = (images[i] * 255).astype("uint8")
            batched_img = np.expand_dims(img, axis=0)
            heatmap,pred_clas = make_gradcam_heatmap(           
                img_array= batched_img,
                model = model,  
                last_conv_layer_name=last_conv_layer
            )
            overlay = get_superimposed_image(img, heatmap)

            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f"True: {class_names[true_cls]}")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(overlay)
            plt.title(f"Pred: {class_names[pred_cls]}")
            plt.axis("off")

            #plt.show()
            plt.tight_layout()
            d = os.path.join("logs", 'best_model')
            os.makedirs(d, exist_ok=True)
            plt.savefig(os.path.join(d,f"misclassified_{count+1}.png"), dpi=300, bbox_inches="tight")
            plt.close()

            count += 1
            if count >= max_samples:
                return
            
def analyze_misclassified_samples2(model,data_dir='dataset/raw/gtsrb',root_dir='dataset', IMG_HEIGHT=30, IMG_WIDTH=30,class_names=None, max_samples=5):
    test_image_data = []
    test_image_labels = []
    test_org_image_data = []
    
    test = pd.read_csv(data_dir + '/Test.csv')
    # filter rows with following class ids only 21,22,42
    test = test[test['ClassId'].isin([21,22,42,18,17,30,38,5,12])]
    imgs = test["Path"].values
    labels = test["ClassId"].values
    count = 0
    for i in range(len(imgs)):
        try:
            image = cv2.imread(data_dir + '/' + imgs[i])
            image_fromarray = Image.fromarray(image, 'RGB')
            resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
            test_image_data.append(np.array(resize_image))
            test_image_labels.append(labels[i])
            test_org_image_data.append(image)
        except:
            print(f"Error in {imgs[i]}")
    
    test_image_data = np.array(test_image_data)
    test_image_labels = np.array(test_image_labels)
    test_image_data = test_image_data/255
    test_image_labels = keras.utils.to_categorical(test_image_labels, 43)

    images = test_image_data
    labels = test_image_labels

    last_conv_layer = None
    for layer in model.layers:
        if layer.name.startswith("conv2d_"):
            last_conv_layer = layer.name  

    preds = model.predict(test_image_data, verbose=0)
    for i in range(len(images)):
        true_cls = np.argmax(labels[i])
        pred_cls = np.argmax(preds[i])

        if true_cls != pred_cls:
            img = test_org_image_data[i] 
            test_proc_img = test_image_data[i]
            batched_img = np.expand_dims(test_proc_img, axis=0)
            heatmap,pred_clas = make_gradcam_heatmap(           
                img_array= batched_img,
                model = model,  
                last_conv_layer_name=last_conv_layer
            )
            overlay = get_superimposed_image(img, heatmap)

            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f"True: {class_names[true_cls]}")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(overlay)
            plt.title(f"Pred: {class_names[pred_cls]}")
            plt.axis("off")

            #plt.show()
            plt.tight_layout()
            d = os.path.join("logs", 'best_model')
            os.makedirs(d, exist_ok=True)
            plt.savefig(os.path.join(d,'correct',f"classified_{count+1}.png"), dpi=300, bbox_inches="tight")
            plt.close()

            count += 1
            if count >= max_samples:
                return

if __name__ == "__main__":
    from tensorflow import keras
    import os
    model_trained = keras.models.load_model('checkpoints/grid_42/best.keras')
    X_test = np.load(os.path.abspath('dataset/processed/X_test.npy'))
    y_test = np.load(os.path.abspath('dataset/processed/y_test.npy'))
    # Label Overview
    class_names = list(get_class_labels().values())
    analyze_misclassified_samples2(model=model_trained, data_dir='dataset/raw/gtsrb', root_dir='dataset', IMG_HEIGHT=30, IMG_WIDTH=30, class_names=class_names, max_samples=40)