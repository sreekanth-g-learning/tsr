import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_vs_hyperparams(csv_path="logs/experiments.csv"):
    df = pd.read_csv(csv_path)

    # Decode JSON hyperparameters
    df["hparams"] = df["hyperparameters_json"].apply(json.loads)

    # Extract variables
    df["num_conv_layers"] = df["hparams"].apply(
        lambda x: len(x["conv_filters"])
    )
    df["dropout_rate"] = df["hparams"].apply(
        lambda x: x["dropout_rate"]
    )
    df["batchnorm"] = df["hparams"].apply(
        lambda x: x["batchnorm"]
    )
    df["learning_rate"] = df["hparams"].apply(
        lambda x: x["learning_rate"]
    )

    df["dropout"] = df["hparams"].apply(
        lambda x: x["dropout"]
    )

    # update dropout rate to 0 if dropout is False
    df.loc[~df["dropout"], "dropout_rate"] = 0

    # Marker shapes for BatchNorm
    marker_map = {
        True: "o",    # BatchNorm ON
        False: "X"    # BatchNorm OFF
    }

    # Edge colors for learning rate
    lr_colors = {
        0.001: "black",
        0.0005: "red",
        0.0001: "blue"
    }

    plt.figure(figsize=(11, 7))

    handles = []
    labels = []

    for bn in [True, False]:
        subset_bn = df[df["batchnorm"] == bn]

        for lr, edge_col in lr_colors.items():
            subset = subset_bn[
                np.isclose(subset_bn["learning_rate"], lr)
            ]

            if subset.empty:
                continue

            sc = plt.scatter(
                subset["num_conv_layers"],
                subset["test_accuracy"],
                s=subset["num_parameters"] / 2000,
                c=subset["dropout_rate"],
                cmap="viridis",
                marker=marker_map[bn],
                edgecolors=edge_col,
                linewidths=1.5,
                alpha=0.75
            )

            handles.append(sc)
            labels.append(f"BN={bn}, LR={lr}")

    # Colorbar for dropout rate
    cbar = plt.colorbar(sc, pad=0.02)
    cbar.set_label("Dropout Rate")

    plt.xlabel("Number of Convolution Layers")
    plt.ylabel("Test Accuracy (%)")
    plt.title(
        "Accuracy vs Conv Layers vs BatchNorm vs Dropout Rate vs Learning Rate"
    )

    plt.grid(True)

    # 🔹 Legend placed BELOW plot (no overlap)
    plt.legend(
        handles,
        labels,
        title="BatchNorm & Learning Rate",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=True
    )

    plt.tight_layout()
    plt.savefig("logs/accuracy_vs_bn_conv_dropout_lr.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_accuracy_vs_hyperparams()