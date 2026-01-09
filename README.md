🚦 Traffic Sign Recognition using Custom CNN & Grad-CAM
📌 Project Overview

This project implements a Traffic Sign Recognition (TSR) system using a Custom Convolutional Neural Network (CNN) trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset.
The system classifies traffic signs into 43 classes, integrates Grad-CAM for explainable AI, and deploys the trained model using a Streamlit web application.

The goal is to build a lightweight, accurate, and interpretable TSR system suitable for academic research and prototype ADAS applications.
----
🎯 Key Features

Custom CNN trained end-to-end (no transfer learning)

Hyperparameter grid experimentation

Class-wise evaluation (accuracy, precision, recall, F1-score)

Confusion matrix and error analysis

Grad-CAM visual explanations

Streamlit-based interactive deployment

Modular, production-ready code structure
---
🗂️ Project Folder Structure  
--
traffic-sign-recognition/  
│  
├── data/  
│   ├── raw/                     # Original GTSRB dataset  
│   ├── processed/               # Preprocessed & augmented data  
│  
├── models/    
│   ├── model_factory.py         # Custom CNN architectures  
│   ├── gradcam.py               # Grad-CAM implementation  
│  
├── training/  
│   ├── train.py                 # Training loop  
│   ├── hyperparameter_grid.py   # Grid search execution  
│   ├── callbacks.py             # Early stopping & checkpoints  
│  
├── evaluation/  
│   ├── metrics.py               # Accuracy, precision, recall  
│   ├── confusion_matrix.py      # Confusion matrix generation  
│   ├── error_analysis.py        # Misclassification analysis  
│  
├── experiments/  
│   ├── experiments.csv          # Experiment results log  
│   ├── classwise_metrics.csv    # Class-wise performance  
│  
├── app/  
│   ├── streamlit_app.py         # Streamlit web application  
│  
├── utils/  
│   ├── data_loader.py           # Dataset loading & preprocessing  
│   ├── visualization.py         # Plots & Grad-CAM overlays  
│  
├── checkpoints/  
│   ├── best_model.keras         # Best model checkpoint  
│  
├── run_pipeline.py              # End-to-end execution script  
├── requirements.txt             # Python dependencies  
└── README.md                    # Project documentation  

---
🧠 Methodology Summary  

Dataset Preparation  

1. GTSRB dataset (43 classes)  

2. Resize to 30×30 pixels  

3. Grayscale conversion & normalization

4. Data augmentation (rotation, zoom, shift, shear)

Model Training

Custom CNN with Conv → BatchNorm → Pool → GAP → Dense

Hyperparameter grid search

Early stopping and checkpointing

Evaluation

Accuracy, precision, recall, F1-score

Confusion matrix

Class-wise and error analysis

Explainability

Grad-CAM heatmaps to visualize model attention

Correlation with confusion matrix errors

Deployment

Streamlit app for image upload

Real-time prediction + Grad-CAM overlay
----
🚀 How to Run the Project
-------------
1️⃣ Install Dependencies
conda create -n tsr python=3.12.11
conda activate tsr
pip install -r requirements.txt

python -m dataprocessing.load_dataset
python -m dataprocessing.preprocessing
python -m training.grid_parallel
python -m evaluation.confusion_analysis
python -m evaluation.gradcam_error_analysis

streamlit run app/streamlit_app.py


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
