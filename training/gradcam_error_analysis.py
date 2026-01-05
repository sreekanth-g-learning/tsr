import numpy as np
import cv2
import matplotlib.pyplot as plt
from models.grad_cam import generate_gradcam
from utils.visualization import overlay_gradcam

def analyze_misclassified_samples(
    model,
    test_ds,
    class_names,
    last_conv_layer="last_conv",
    max_samples=5
):
    count = 0

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)

        for i in range(len(images)):
            true_cls = np.argmax(labels[i])
            pred_cls = np.argmax(preds[i])

            if true_cls != pred_cls:
                img = (images[i].numpy() * 255).astype("uint8")

                heatmap = generate_gradcam(
                    model,
                    np.expand_dims(images[i], axis=0),
                    last_conv_layer,
                    pred_cls
                )

                overlay = overlay_gradcam(img, heatmap)

                plt.figure(figsize=(6, 3))
                plt.subplot(1, 2, 1)
                plt.imshow(img)
                plt.title(f"True: {class_names[true_cls]}")
                plt.axis("off")

                plt.subplot(1, 2, 2)
                plt.imshow(overlay)
                plt.title(f"Pred: {class_names[pred_cls]}")
                plt.axis("off")

                plt.show()

                count += 1
                if count >= max_samples:
                    return
