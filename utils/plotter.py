
import os, pandas as pd, matplotlib.pyplot as plt

def save_training_plots(history, model_name):
    d = os.path.join("logs", model_name)
    os.makedirs(d, exist_ok=True)
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(d, "history.csv"), index=False)

    plt.figure()
    plt.plot(df["accuracy"]); 
    plt.plot(df["val_accuracy"])
    plt.title("Accuracy - "+model_name)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["train","val"])
    plt.savefig(os.path.join(d,"accuracy.png")); 
    plt.close()

    plt.figure()
    plt.plot(df["loss"]); 
    plt.plot(df["val_loss"])
    plt.title("Loss - "+model_name)
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.legend(["train","val"])
    plt.savefig(os.path.join(d,"loss.png")); 
    plt.close()
