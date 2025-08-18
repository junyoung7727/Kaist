"""
Test script for the simple circuit generator
"""

import sys
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "quantumcommon"))

from inference.simple_circuit_generator import SimpleCircuitGenerator, SimpleGenerationConfig


def test_simple_generator():
    """Test the simple circuit generator"""
    print("🧪 Testing Simple Circuit Generator")
    
    # Test configuration
    config = SimpleGenerationConfig(
        max_gates=10,
        num_qubits=4,
        temperature=0.8,
        verbose=True
    )
    
    # Test 1: Random generation (no model)
    print("\n📋 Test 1: Random Circuit Generation")
    try:
        generator = SimpleCircuitGenerator(model_path=None, config=config)
        circuit = generator.generate_circuit_random()
        
        print(f"✅ Random circuit generated successfully")
        print(f"   Circuit ID: {circuit.circuit_id}")
        print(f"   Qubits: {circuit.num_qubits}")
        print(f"   Gates: {len(circuit.gates)}")
        
        # Show first few gates
        for i, gate in enumerate(circuit.gates[:3]):
            print(f"   Gate {i+1}: {gate.name} on qubits {gate.qubits}")
        
    except Exception as e:
        print(f"❌ Random generation failed: {e}")
        return False
    
    # Test 2: Model-based generation (if model exists)
    print("\n🤖 Test 2: Model-based Circuit Generation")
    model_path = Path("checkpoints/best_model.pt")
    
    try:
        generator_with_model = SimpleCircuitGenerator(str(model_path), config)
        
        if generator_with_model.model is not None:
            print("✅ Model loaded successfully")
            
            # Test model-based generation
            circuit = generator_with_model.generate_circuit_model()
            print(f"✅ Model-based circuit generated")
            print(f"   Circuit ID: {circuit.circuit_id}")
            print(f"   Qubits: {circuit.num_qubits}")
            print(f"   Gates: {len(circuit.gates)}")
            
        else:
            print("⚠️ No model found, using random generation")
            circuit = generator_with_model.generate_circuit()
            print(f"✅ Fallback circuit generated")
            
    except Exception as e:
        print(f"❌ Model-based generation failed: {e}")
        print(f"   Error details: {str(e)}")
        return False
    
    # Test 3: Multiple circuit generation
    print("\n📊 Test 3: Multiple Circuit Generation")
    try:
        circuits = generator.generate_multiple_circuits(num_circuits=3)
        print(f"✅ Generated {len(circuits)} circuits")
        
        for i, circuit in enumerate(circuits):
            print(f"   Circuit {i+1}: {len(circuit.gates)} gates")
            
    except Exception as e:
        print(f"❌ Multiple generation failed: {e}")
        return False
    
    # Test 4: Save circuits
    print("\n💾 Test 4: Save Circuits")
    try:
        output_path = "test_circuits.json"
        generator.save_circuits(circuits, output_path)
        
        # Check if file was created
        if Path(output_path).exists():
            print(f"✅ Circuits saved to {output_path}")
        else:
            print(f"❌ File {output_path} was not created")
            return False
            
    except Exception as e:
        print(f"❌ Save failed: {e}")
        return False
    
    print("\n🎉 All tests passed!")
    return True


if __name__ == "__main__":
    success = test_simple_generator()
    if not success:
        sys.exit(1)
