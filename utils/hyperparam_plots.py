import pandas as pd
import matplotlib.pyplot as plt
import json

def plot_accuracy_vs_params(csv_path="logs/experiments.csv"):
    df = pd.read_csv(csv_path)

    # Accuracy vs Parameters
    plt.figure(figsize=(8,5))
    plt.scatter(df["num_parameters"], df["test_accuracy"])
    plt.xlabel("Number of Parameters")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Model Size")
    plt.grid(True)
    plt.savefig("logs/accuracy_vs_params.png")
    plt.close()

    # Accuracy vs Epochs
    plt.figure(figsize=(8,5))
    plt.scatter(df["epochs"], df["test_accuracy"])
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Training Epochs")
    plt.grid(True)
    plt.savefig("logs/accuracy_vs_epochs.png")
    plt.close()

    df["hyperparameters"] = df["hyperparameters_json"].apply(json.loads)
    df["learning_rate"] = df["hyperparameters"].apply(lambda x: x["learning_rate"])
    df["dropout_rate"] = df["hyperparameters"].apply(lambda x: x["dropout_rate"])
    df["conv_filters"] = df["hyperparameters"].apply(lambda x: x["conv_filters"])
    df["batchnorm"] = df["hyperparameters"].apply(lambda x: x["batchnorm"])

    # Accuracy vs Learning Rate
    plt.figure(figsize=(8,5))
    plt.scatter(df["learning_rate"], df["test_accuracy"])
    plt.xlabel("Learning Rate")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Learning Rate")
    plt.grid(True)
    plt.savefig("logs/accuracy_vs_learning_rate.png")
    plt.close()

    # Accuracy vs Dropout Rate
    plt.figure(figsize=(8,5))
    plt.scatter(df["dropout_rate"], df["test_accuracy"])
    plt.xlabel("Dropout Rate")

    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Dropout Rate")
    plt.grid(True)
    plt.savefig("logs/accuracy_vs_dropout_rate.png")
    plt.close()

    # Accuracy vs Conv Filters
    #plt.figure(figsize=(8,5))
    #plt.scatter(df["conv_filters"], df["test_accuracy"])
    #plt.xlabel("Convolutional Filters")
    #plt.ylabel("Test Accuracy (%)")
    #plt.title("Accuracy vs Conv Filters")
    #plt.grid(True)
    #plt.savefig("logs/accuracy_vs_conv_filters.png")
    #plt.close()
    #print("✅ Saved hyperparameter plots in logs/ directory.")

    # Accuracy vs BatchNorm
    # filter only one value of conv_filters i.e [16, 32, 64, 128] to reduce clutter

    df_bn = df[df["conv_filters"] == str([16, 32, 64, 128])]

    # plot accuracy vs batchnorm where batchnorm is 0 or 1 and conv_filters is [16, 32, 64, 128] and dropout_rate is constant 0.5
    df_bn = df_bn[df_bn["dropout_rate"] == 0.5]
    
    plt.figure(figsize=(8,5))

    plt.scatter(df_bn["batchnorm"], df_bn["test_accuracy"])
    plt.xlabel("Batch Normalization (0=No, 1=Yes)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Batch Normalization for Conv Filters [16, 32, 64, 128]")
    plt.grid(True)

    plt.savefig("logs/accuracy_vs_batchnorm.png")
    plt.close()
    print("✅ Saved hyperparameter plots in logs/ directory.")


if __name__ == "__main__":
    plot_accuracy_vs_params()