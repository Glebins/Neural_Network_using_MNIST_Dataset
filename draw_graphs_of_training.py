import re
import matplotlib.pyplot as plt

filename = "data.txt"

pattern = re.compile(r"I:\s*(\d+)\s+Train-Acc:\s*([\d.]+)\s+Test-Acc:\s*([\d.]+)\s+Error:\s*([\d.]+)")

epochs, train_acc, test_acc, losses = [], [], [], []

with open(filename, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            e, tr, te, lo = match.groups()
            epochs.append(int(e))
            train_acc.append(float(tr))
            test_acc.append(float(te))
            losses.append(float(lo))

fig, (axes_1, axes_2) = plt.subplots(1, 2)

axes_1.plot(epochs, losses, label="Loss", linewidth=3)
axes_1.legend()
axes_1.grid()

axes_2.plot(epochs, train_acc, label="Train Accuracy", linewidth=2)
axes_2.plot(epochs, test_acc, label="Test Accuracy", linewidth=2)
axes_2.legend()
axes_2.grid()

plt.tight_layout()
plt.show()
