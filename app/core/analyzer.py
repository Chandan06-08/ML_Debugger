import json

def analyze_training(path):
    with open(path, "r") as f:
        logs = json.load(f)

    train = logs["train_acc"][-1]
    val = logs["val_acc"][-1]

    return {
        "train_acc": train,
        "val_acc": val,
        "gap": train - val
    }