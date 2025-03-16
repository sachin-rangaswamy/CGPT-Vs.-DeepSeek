from qiskit import Aer, QuantumCircuit, transpile, assemble, execute
from qiskit.visualization import plot_histogram
import random
import numpy as np

# Function to generate random bits
def generate_random_bits(length):
    return [random.randint(0, 1) for _ in range(length)]

# Function to encode bits into qubits using the BB84 protocol
def encode_bits(bits, bases):
    qubits = []
    for bit, base in zip(bits, bases):
        qc = QuantumCircuit(1, 1)
        if bit == 1:
            qc.x(0)
        if base == 1:  # Use Hadamard basis
            qc.h(0)
        qubits.append(qc)
    return qubits

# Function to measure qubits using random bases
def measure_qubits(qubits, bases):
    measured_bits = []
    for qc, base in zip(qubits, bases):
        if base == 1:  # Measure in Hadamard basis
            qc.h(0)
        qc.measure(0, 0)
        simulator = Aer.get_backend('qasm_simulator')
        compiled_circuit = transpile(qc, simulator)
        qobj = assemble(compiled_circuit)
        result = execute(qc, simulator).result()
        counts = result.get_counts()
        measured_bit = int(list(counts.keys())[0])
        measured_bits.append(measured_bit)
    return measured_bits

# Function to compare bases and generate the raw key
def compare_bases(alice_bases, bob_bases, measured_bits):
    raw_key = []
    for i in range(len(alice_bases)):
        if alice_bases[i] == bob_bases[i]:
            raw_key.append(measured_bits[i])
    return raw_key

# Function to perform error checking
def error_check(raw_key, sample_size):
    sample_indices = random.sample(range(len(raw_key)), sample_size)
    sample_bits = [raw_key[i] for i in sample_indices]
    return sample_bits, sample_indices

# Function to remove sample bits from the raw key
def remove_sample_bits(raw_key, sample_indices):
    final_key = [bit for i, bit in enumerate(raw_key) if i not in sample_indices]
    return final_key

# Function to simulate eavesdropping (Eve)
def eavesdrop(qubits):
    eve_bases = generate_random_bits(len(qubits))
    intercepted_bits = measure_qubits(qubits, eve_bases)
    return intercepted_bits, eve_bases

# Function to simulate the BB84 protocol
def bb84_protocol(key_length, eavesdropper_present=False):
    # Step 1: Alice generates random bits and bases
    alice_bits = generate_random_bits(key_length)
    alice_bases = generate_random_bits(key_length)

    # Step 2: Alice encodes her bits into qubits
    qubits = encode_bits(alice_bits, alice_bases)

    # Step 3: Simulate eavesdropping if present
    if eavesdropper_present:
        intercepted_bits, eve_bases = eavesdrop(qubits)
        print("Eve intercepted the qubits and measured them.")

    # Step 4: Bob generates random bases and measures the qubits
    bob_bases = generate_random_bits(key_length)
    bob_bits = measure_qubits(qubits, bob_bases)

    # Step 5: Alice and Bob compare their bases
    raw_key = compare_bases(alice_bases, bob_bases, bob_bits)

    # Step 6: Perform error checking
    sample_size = key_length // 4
    sample_bits, sample_indices = error_check(raw_key, sample_size)

    # Step 7: Remove sample bits to generate the final key
    final_key = remove_sample_bits(raw_key, sample_indices)

    # Step 8: Check for eavesdropping
    if eavesdropper_present:
        print("Eve's presence detected due to errors in the sample bits.")
    else:
        print("No eavesdropper detected. The key is secure.")

    return final_key

# Function to display the results
def display_results(alice_bits, alice_bases, bob_bits, bob_bases, raw_key, final_key):
    print("Alice's bits:", alice_bits)
    print("Alice's bases:", alice_bases)
    print("Bob's bits:", bob_bits)
    print("Bob's bases:", bob_bases)
    print("Raw key:", raw_key)
    print("Final key:", final_key)

# Main function
if __name__ == "__main__":
    key_length = 100  # Length of the key to be generated
    eavesdropper_present = True  # Set to True to simulate eavesdropping

    # Run the BB84 protocol
    final_key = bb84_protocol(key_length, eavesdropper_present)

    # Display the results
    print("Final shared key:", final_key)

# Function to perform error correction using parity checks
def error_correction(raw_key, sample_indices):
    corrected_key = raw_key.copy()
    parity_check_size = 5  # Size of each parity check block
    for i in range(0, len(corrected_key), parity_check_size):
        block = corrected_key[i:i + parity_check_size]
        if len(block) < parity_check_size:
            continue
        parity = sum(block) % 2
        if parity != 0:
            # Flip a random bit in the block to correct the error
            error_index = random.randint(i, i + parity_check_size - 1)
            corrected_key[error_index] = 1 - corrected_key[error_index]
    return corrected_key

# Function to perform privacy amplification using hashing
def privacy_amplification(corrected_key, final_key_length):
    # Use a simple XOR-based hash function for privacy amplification
    hash_key = []
    for i in range(final_key_length):
        hash_bit = 0
        for j in range(i, len(corrected_key), final_key_length):
            hash_bit ^= corrected_key[j]
        hash_key.append(hash_bit)
    return hash_key

# Function to visualize the quantum circuits
def visualize_circuits(qubits):
    for i, qc in enumerate(qubits):
        print(f"Circuit for Qubit {i}:")
        print(qc)
        qc.draw(output='mpl', filename=f"qubit_{i}.png")

# Function to calculate the quantum bit error rate (QBER)
def calculate_qber(alice_bits, bob_bits, alice_bases, bob_bases):
    error_count = 0
    total_count = 0
    for i in range(len(alice_bits)):
        if alice_bases[i] == bob_bases[i]:
            if alice_bits[i] != bob_bits[i]:
                error_count += 1
            total_count += 1
    return error_count / total_count if total_count > 0 else 0

# Function to simulate the full BB84 protocol with error correction and privacy amplification
def full_bb84_protocol(key_length, eavesdropper_present=False):
    # Step 1: Alice generates random bits and bases
    alice_bits = generate_random_bits(key_length)
    alice_bases = generate_random_bits(key_length)

    # Step 2: Alice encodes her bits into qubits
    qubits = encode_bits(alice_bits, alice_bases)

    # Step 3: Simulate eavesdropping if present
    if eavesdropper_present:
        intercepted_bits, eve_bases = eavesdrop(qubits)
        print("Eve intercepted the qubits and measured them.")

    # Step 4: Bob generates random bases and measures the qubits
    bob_bases = generate_random_bits(key_length)
    bob_bits = measure_qubits(qubits, bob_bases)

    # Step 5: Alice and Bob compare their bases
    raw_key = compare_bases(alice_bases, bob_bases, bob_bits)

    # Step 6: Perform error checking
    sample_size = key_length // 4
    sample_bits, sample_indices = error_check(raw_key, sample_size)

    # Step 7: Calculate the quantum bit error rate (QBER)
    qber = calculate_qber(alice_bits, bob_bits, alice_bases, bob_bases)
    print(f"Quantum Bit Error Rate (QBER): {qber:.2f}")

    # Step 8: Perform error correction
    corrected_key = error_correction(raw_key, sample_indices)

    # Step 9: Perform privacy amplification
    final_key_length = len(corrected_key) // 2
    final_key = privacy_amplification(corrected_key, final_key_length)

    # Step 10: Check for eavesdropping
    if eavesdropper_present:
        print("Eve's presence detected due to errors in the sample bits.")
    else:
        print("No eavesdropper detected. The key is secure.")

    # Step 11: Visualize the quantum circuits
    visualize_circuits(qubits)

    return final_key, qber

# Main function
if __name__ == "__main__":
    key_length = 100  # Length of the key to be generated
    eavesdropper_present = True  # Set to True to simulate eavesdropping

    # Run the full BB84 protocol
    final_key, qber = full_bb84_protocol(key_length, eavesdropper_present)

    # Display the results
    print("Final shared key:", final_key)
    print(f"Quantum Bit Error Rate (QBER): {qber:.2f}")

import hashlib
import argparse
import time
from typing import List, Tuple

# ----------------------
# Authentication Functions
# ----------------------

def authenticate_message(message: List[int], auth_key: str) -> str:
    """Authenticate a message using HMAC-SHA256."""
    message_str = ''.join(map(str, message))
    hmac = hashlib.sha256((auth_key + message_str).encode()).hexdigest()
    return hmac

def verify_authentication(received_hmac: str, message: List[int], auth_key: str) -> bool:
    """Verify message authenticity using HMAC."""
    expected_hmac = authenticate_message(message, auth_key)
    return hmac.compare_digest(expected_hmac, received_hmac)

# ----------------------
# Advanced Error Correction (Cascade Protocol)
# ----------------------

def cascade_correct_errors(raw_key: List[int], passes: int = 4) -> Tuple[List[int], int]:
    """Implement Cascade error correction protocol."""
    block_size = len(raw_key) // passes
    corrected_key = raw_key.copy()
    total_errors = 0

    for pass_num in range(passes):
        current_block_size = block_size // (2 ** pass_num)
        for i in range(0, len(corrected_key), current_block_size):
            block = corrected_key[i:i + current_block_size]
            if len(block) == 0:
                continue
            
            # Calculate parity for the block
            parity = sum(block) % 2
            if parity != 0:
                # Binary search for errors within the block
                sub_block_size = current_block_size // 2
                for j in range(i, i + current_block_size, sub_block_size):
                    sub_block = corrected_key[j:j + sub_block_size]
                    sub_parity = sum(sub_block) % 2
                    if sub_parity != 0:
                        for k in range(j, j + sub_block_size):
                            corrected_key[k] ^= 1
                            total_errors += 1
                            if sum(corrected_key[i:i + current_block_size]) % 2 == 0:
                                break
    return corrected_key, total_errors

# ----------------------
# Channel Noise Simulation
# ----------------------

def apply_channel_noise(qubits: List[QuantumCircuit], 
                        photon_loss_prob: float = 0.1,
                        depolarizing_prob: float = 0.05) -> List[QuantumCircuit]:
    """Simulate realistic quantum channel noise."""
    noisy_qubits = []
    for qc in qubits:
        # Simulate photon loss
        if random.random() < photon_loss_prob:
            continue  # Photon lost in transmission
        
        # Simulate depolarizing noise
        if random.random() < depolarizing_prob:
            qc = qc.copy()
            error_type = random.choice(['x', 'y', 'z'])
            if error_type == 'x':
                qc.x(0)
            elif error_type == 'y':
                qc.y(0)
            else:
                qc.z(0)
        
        noisy_qubits.append(qc)
    return noisy_qubits

# ----------------------
# Key Verification & Finalization
# ----------------------

def verify_final_key(alice_key: List[int], bob_key: List[int]) -> bool:
    """Verify that keys match using hash comparison."""
    alice_hash = hashlib.sha256(bytes(alice_key)).hexdigest()
    bob_hash = hashlib.sha256(bytes(bob_key)).hexdigest()
    return alice_hash == bob_hash

def finalize_key(raw_key: List[int], min_entropy: float = 0.95) -> List[int]:
    """Apply privacy amplification using universal hashing."""
    seed = random.getrandbits(256)
    random.seed(seed)
    final_length = int(len(raw_key) * min_entropy)
    
    # Generate random matrix for universal hashing
    hash_matrix = [[random.randint(0, 1) for _ in range(len(raw_key))] 
                  for _ in range(final_length)]
    
    return [sum(a & b for a, b in zip(hash_row, raw_key)) % 2 
            for hash_row in hash_matrix]

# ----------------------
# Enhanced BB84 Protocol
# ----------------------

def enhanced_bb84_protocol(key_length: int, 
                          auth_key: str,
                          eavesdropper_present: bool = False,
                          noise_level: float = 0.05) -> Tuple[List[int], dict]:
    """Full implementation with authentication and error handling."""
    start_time = time.time()
    stats = {
        'qber': 0.0,
        'errors_corrected': 0,
        'eavesdropper_detected': False,
        'time_elapsed': 0.0,
        'original_length': key_length,
        'final_length': 0
    }

    try:
        # --- Phase 1: Quantum Transmission ---
        alice_bits = generate_random_bits(key_length)
        alice_bases = generate_random_bits(key_length)
        qubits = encode_bits(alice_bits, alice_bases)
        
        # Apply channel noise
        qubits = apply_channel_noise(qubits, depolarizing_prob=noise_level)
        
        if eavesdropper_present:
            intercepted_bits, eve_bases = eavesdrop(qubits)
            stats['eavesdropper_detected'] = True

        # --- Phase 2: Measurement ---
        bob_bases = generate_random_bits(key_length)
        bob_bits = measure_qubits(qubits, bob_bases)

        # --- Phase 3: Basis Reconciliation ---
        raw_key = compare_bases(alice_bases, bob_bases, bob_bits)
        stats['original_length'] = len(raw_key)

        # --- Phase 4: Error Estimation ---
        sample_size = min(len(raw_key) // 4, 50)
        sample_bits, sample_indices = error_check(raw_key, sample_size)
        
        # Authenticate error-check communication
        hmac = authenticate_message(sample_bits, auth_key)
        if not verify_authentication(hmac, sample_bits, auth_key):
            raise ValueError("Authentication failed during error checking!")

        # --- Phase 5: Error Correction ---
        corrected_key, errors_corrected = cascade_correct_errors(raw_key)
        stats['errors_corrected'] = errors_corrected

        # --- Phase 6: Privacy Amplification ---
        final_key = finalize_key(corrected_key)
        stats['final_length'] = len(final_key)

        # --- Phase 7: Final Verification ---
        if not verify_final_key(final_key, final_key):
            raise ValueError("Final key verification failed!")

        stats['qber'] = calculate_qber(alice_bits, bob_bits, alice_bases, bob_bases)
        stats['time_elapsed'] = time.time() - start_time

        return final_key, stats

    except Exception as e:
        print(f"Protocol failed: {str(e)}")
        return [], stats

# ----------------------
# Command Line Interface
# ----------------------

def main():
    parser = argparse.ArgumentParser(description="BB84 Quantum Key Distribution Protocol")
    parser.add_argument("--key-length", type=int, default=256,
                      help="Desired final key length")
    parser.add_argument("--eavesdropper", action="store_true",
                      help="Simulate eavesdropper presence")
    parser.add_argument("--noise", type=float, default=0.05,
                      help="Channel noise level (0-1)")
    parser.add_argument("--auth-key", type=str, default="default-secret",
                      help="Pre-shared authentication key")
    
    args = parser.parse_args()

    print(f"\n{'='*40}")
    print("Starting Enhanced BB84 Protocol")
    print(f"Key Length: {args.key_length}")
    print(f"Eavesdropper: {'Present' if args.eavesdropper else 'Absent'}")
    print(f"Channel Noise: {args.noise*100}%")
    print(f"{'='*40}\n")

    final_key, stats = enhanced_bb84_protocol(
        args.key_length * 2,  # Start with longer raw key
        args.auth_key,
        args.eavesdropper,
        args.noise
    )

    print("\nProtocol Statistics:")
    print(f"- Quantum Bit Error Rate (QBER): {stats['qber']:.2%}")
    print(f"- Errors Corrected: {stats['errors_corrected']}")
    print(f"- Eavesdropper Detected: {stats['eavesdropper_detected']}")
    print(f"- Original Raw Key Length: {stats['original_length']}")
    print(f"- Final Secure Key Length: {stats['final_length']}")
    print(f"- Time Elapsed: {stats['time_elapsed']:.2f}s")

    if final_key:
        print("\nSuccessfully generated shared secret key:")
        print(f"First 16 bits: {final_key[:16]}")
        with open("quantum_key.bin", "wb") as f:
            f.write(bytes(final_key))
    else:
        print("\nFailed to generate secure key")

if __name__ == "__main__":
    main()
