import numpy as np
import pandas as pd
from tqdm import tqdm
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

def simulate_sqkd_data(n_bits: int, noise: float, eve: bool, eve_rate: float):
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
        
        qc.id(0) # Aller

        if eve and np.random.rand() < eve_rate:
            eve_base = np.random.randint(2)
            if eve_base == 1: qc.h(0)
            qc.measure(0, 0)
            if eve_base == 1: qc.h(0)

        if bob_actions[i] == 1: 
            qc.measure(0, 0)
        
        qc.id(0) # Retour

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

    if len(alice_key) == 0:
        return 0.5, 0.0

    qber = np.sum(alice_key != bob_key) / len(alice_key)
    gain_shift = abs((len(alice_key) / n_bits) - 0.25)
    return float(qber), float(gain_shift)

def generate_sqkd_dataset(num_samples=5000, bits_per_key=150):
    data = []
    print(f"Génération de {num_samples} échantillons SQKD...")
    for _ in tqdm(range(num_samples)):
        is_eve_active = np.random.choice([True, False])
        noise_level = np.random.uniform(0.0, 0.08)
        eve_rate = np.random.uniform(0.50, 1.0) if is_eve_active else 0.0
        
        qber, gain_shift = simulate_sqkd_data(bits_per_key, noise_level, is_eve_active, eve_rate)
        data.append({"noise_level": noise_level, "eve_rate": eve_rate, "QBER": qber, "Gain_Shift": gain_shift, "Label": 1 if is_eve_active else 0})
        
    pd.DataFrame(data).to_csv("sqkd_dataset.csv", index=False)
    print("Dataset SQKD généré : 'sqkd_dataset.csv'")

if __name__ == "__main__":
    generate_sqkd_dataset()