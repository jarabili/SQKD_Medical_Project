import base64
import io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from fastapi import FastAPI
from pydantic import BaseModel
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

matplotlib.use("Agg")
app = FastAPI(title="SQKD-QML Detection Service")

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
def qml_classifier(x, weights):
    feature_map(x)
    variational_ansatz(weights)
    return qml.expval(qml.PauliZ(0))

# REMPLACER PAR VOS POIDS APRÈS ENTRAÎNEMENT
QML_WEIGHTS = np.array([
    [0.45, 1.21, -0.32],
    [1.10, -0.85, 0.42],
])

class SimulationRequest(BaseModel):
    n_bits: int = 500 
    noise_level: float = 0.03  
    eve_active: bool = False
    eve_interception_rate: float = 1.0 

def simulate_sqkd(n_bits, noise, eve, eve_rate):
    np.random.seed(None)
    alice_bits = np.random.randint(2, size=n_bits)
    alice_bases = np.random.randint(2, size=n_bits)
    bob_actions = np.random.randint(2, size=n_bits)

    noise_model = NoiseModel()
    if noise > 0:
        error = depolarizing_error(noise, 1)
        noise_model.add_all_qubit_quantum_error(error, ['id'])

    simulator = AerSimulator()
    circuits = []

    for i in range(n_bits):
        qc = QuantumCircuit(1, 2, name=f"qc_{i}") 
        if alice_bits[i] == 1: qc.x(0)
        if alice_bases[i] == 1: qc.h(0)
        qc.id(0)

        if eve and np.random.rand() < eve_rate:
            eve_base = np.random.randint(2)
            if eve_base == 1: qc.h(0)
            qc.measure(0, 0) 
            if eve_base == 1: qc.h(0) 

        if bob_actions[i] == 1: qc.measure(0, 0) 
        qc.id(0)

        if alice_bases[i] == 1: qc.h(0)
        qc.measure(0, 1)
        circuits.append(qc)

    job = simulator.run(circuits, shots=1, noise_model=noise_model)
    result = job.result()

    bob_results, alice_results = [], []
    for i in range(n_bits):
        counts = result.get_counts(f"qc_{i}")
        bitstring = list(counts.keys())[0].replace(' ', '')
        alice_results.append(int(bitstring[0]))
        bob_results.append(int(bitstring[1]))

    key_indices = (alice_bases == 0) & (bob_actions == 1)
    alice_key = alice_bits[key_indices]
    bob_key = np.array(bob_results)[key_indices]

    if len(alice_key) == 0: return 0.5, 0.0, 0

    qber = float(np.sum(alice_key != bob_key) / len(alice_key))
    gain_shift = float(abs((len(alice_key) / n_bits) - 0.25))
    return qber, gain_shift, len(alice_key)

@app.post("/simulate_sqkd")
def run_simulation(req: SimulationRequest):
    qber, gain_shift, key_len = simulate_sqkd(req.n_bits, req.noise_level, req.eve_active, req.eve_interception_rate)
    qml_features = np.array([qber, gain_shift])
    expval = float(qml_classifier(qml_features, QML_WEIGHTS))

    eve_detected = expval < 0.0 or qber > 0.11
    status = "COMPROMISED" if eve_detected else "SECURE"

    fig, ax = plt.subplots(figsize=(5, 3))
    categories = ["QBER Mesuré", "Seuil Critique (11%)", "QML Output"]
    values = [qber * 100, 11.0, expval * 10]
    ax.bar(categories, values, color=["#1f77b4", "#ff7f0e", "#2ca02c" if not eve_detected else "#d62728"])
    ax.set_ylabel("Valeurs relatives (%)")
    ax.set_title(f"Statut Canal SQKD : {status}")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "status": status,
        "protocol": "SQKD",
        "qber": round(qber, 4),
        "gain_shift": round(gain_shift, 4),
        "sifted_key_length": key_len,
        "qml_expectation_value": round(expval, 4),
        "eve_interception_predicted": eve_detected,
        "plot_base64": img_b64,
    }