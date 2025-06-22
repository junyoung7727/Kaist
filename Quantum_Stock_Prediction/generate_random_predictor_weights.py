import os
import torch
from quantum_rl.predictor import CircuitPredictor
from quantum_rl.constants import PREDICTOR_MODEL_PATH

# Ensure output directory exists
output_dir = os.path.dirname(PREDICTOR_MODEL_PATH)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Instantiate predictor with fresh random weights
predictor = CircuitPredictor(model_path=None)

# Save random-initialized weights
torch.save(predictor.model.state_dict(), PREDICTOR_MODEL_PATH)
print(f"Saved random predictor weights to {PREDICTOR_MODEL_PATH}")
