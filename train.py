import pandas as pd
import numpy as np
import pennylane as qml
from sklearn.model_selection import train_test_split

print("Chargement du dataset sqkd_dataset.csv...")
df = pd.read_csv("sqkd_dataset.csv")

X = df[["QBER", "Gain_Shift"]].values
y = np.where(df["Label"].values == 0, 1.0, -1.0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

def feature_map(x):
    qml.RY(x[0] * np.pi, wires=0)
    qml.RY(x[1] * np.pi, wires=1)
    qml.CNOT(wires=[0, 1])

def variational_ansatz(weights):
    for i in range(n_qubits):
        qml.Rot(*weights[i], wires=i)
    qml.CNOT(wires=[0, 1])

@qml.qnode(dev)
def qml_classifier(weights, x):
    feature_map(x)
    variational_ansatz(weights)
    return qml.expval(qml.PauliZ(0))

def cost(weights, X_batch, Y_batch):
    predictions = np.array([qml_classifier(weights, x) for x in X_batch])
    return np.mean((predictions - Y_batch) ** 2)

np.random.seed(42)
weights = np.random.randn(2, 3) * 0.1
opt = qml.AdamOptimizer(stepsize=0.1)

epochs, batch_size = 30, 32
print("Début de l'optimisation QML...")
for it in range(epochs):
    batch_index = np.random.randint(0, len(X_train), (batch_size,))
    weights, _, _ = opt.step(cost, weights, X_train[batch_index], y_train[batch_index])
    if (it + 1) % 5 == 0:
        print(f"Étape {it + 1:3d}/{epochs} | Coût : {cost(weights, X_test, y_test):.4f}")

print("\nMatrice des poids optimisés (à copier dans app.py) :")
print(repr(weights))