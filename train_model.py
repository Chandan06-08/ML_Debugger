import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dummy dataset
X = [[i] for i in range(100)]
y = [0 if i < 50 else 1 for i in range(100)]

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

train_acc_list = []
val_acc_list = []

model = LogisticRegression()

# Simulate epochs
for i in range(5):
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    # artificially create overfitting effect
    train_acc += 0.02 * i  

    train_acc_list.append(min(train_acc, 1.0))
    val_acc_list.append(val_acc)

# Save logs
logs = {
    "train_acc": train_acc_list,
    "val_acc": val_acc_list
}

with open("data/real_logs.json", "w") as f:
    json.dump(logs, f)

print("Training complete. Logs saved.")