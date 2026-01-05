TSR - Traffic Signs Recognition

Project setup
-------------

conda create -n tsr python=3.12.11
conda activate tsr


Citation

J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In Proceedings of the IEEE International Joint Conference on Neural Networks, pages 1453–1460. 2011.

@inproceedings{Stallkamp-IJCNN-2011,
    author = {Johannes Stallkamp and Marc Schlipsing and Jan Salmen and Christian Igel},
    booktitle = {IEEE International Joint Conference on Neural Networks},
    title = {The {G}erman {T}raffic {S}ign {R}ecognition {B}enchmark: A multi-class classification competition},
    year = {2011},
    pages = {1453--1460}
}



🔍 Observations from the Plot

1️⃣ Effect of Number of Convolution Layers

Models with 2 convolution layers show high variability in accuracy, ranging from very low (~3–5%) to ~85%, indicating insufficient feature learning and unstable training.

Models with 3 convolution layers achieve consistently high accuracy (≈88–99%), suggesting this depth is optimal for the GTSRB dataset.

Models with 4 convolution layers show marginal improvement or saturation (≈95–99%), indicating diminishing returns with increased depth.

📌 Conclusion: Increasing depth improves accuracy up to a point, after which gains plateau.

2️⃣ Impact of Batch Normalization

Configurations with Batch Normalization (BN=True) consistently outperform those without BN at the same depth.

BN significantly stabilizes training for deeper models, particularly at 3 and 4 convolution layers.

Without BatchNorm, some models show severe accuracy degradation, especially at lower depths.

📌 Conclusion: Batch Normalization is a critical component for stable and high-performing CNN training.

3️⃣ Effect of Dropout Rate

Moderate dropout rates (0.2–0.3) correspond to the highest accuracy regions in the plot.

High dropout (≈0.5) sometimes leads to reduced accuracy, especially for shallow networks.

Very low or zero dropout can cause overfitting, reflected in inconsistent performance.

📌 Conclusion: A balanced dropout rate improves generalization, while excessive dropout harms learning.

4️⃣ Influence of Learning Rate

Learning rate = 0.001 (black edges) consistently produces higher accuracy across depths.

Lower learning rates (0.0005) show slower convergence and slightly reduced accuracy.

Extremely low learning rates offer no clear benefit for this task.

📌 Conclusion: A learning rate of 0.001 provides the best trade-off between convergence speed and accuracy.

5️⃣ Overall Best Configuration Trend

The best-performing models share the following characteristics:

3–4 convolution layers

Batch Normalization enabled

Dropout rate between 0.2 and 0.3

Learning rate = 0.001

These configurations consistently achieve ~98–99% test accuracy.
