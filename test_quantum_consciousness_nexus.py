from planetary_optimization.components.quantum_consciousness_nexus import QuantumConsciousnessNexus

def test_quantum_consciousness_nexus():
    print("Testing Quantum Consciousness Nexus...")

    # Initialize nexus
    nexus = QuantumConsciousnessNexus(num_qubits=8, num_microtubules=1000)  # Reduced for testing

    # Run simulation
    emergence, amplified, integrated = nexus.run_full_simulation()

    print("Test completed successfully!")
    print(f"Consciousness measure: {emergence[0]:.6f}")
    print(f"Decoherence time: {emergence[1]:.2e} seconds")

    return emergence, amplified, integrated

if __name__ == "__main__":
    test_quantum_consciousness_nexus()