import torch

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model/Training Hyperparameters
MAX_QUBITS = 16  # Maximum number of qubits in a circuit
MAX_GATES = 100  # Maximum number of gates in a circuit sequence
MAX_DEPTH = 30   # Maximum circuit depth

LATENT_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
NUM_GATE_TYPES = 15

# RL Agent & Training Parameters
RL_BATCH_SIZE = 128 # Batch size for PPO updates
RL_EPOCHS = 10 # Number of epochs for PPO updates per iteration
LEARNING_RATE = 3e-4
GAMMA = 0.99 # Discount factor for rewards
GAE_LAMBDA = 0.95 # Lambda for Generalized Advantage Estimation
CLIP_EPSILON = 0.2 # Epsilon for PPO clipping
ENTROPY_COEF = 0.01 # Coefficient for entropy bonus
CRITIC_COEF = 0.5 # Coefficient for critic loss
MAX_GRAD_NORM = 0.5 # Max gradient norm for clipping
ACTION_DIM = 3  # Dimension of the action space (e.g., complexity_factor, entanglement_factor, gate_density)

# Diffusion Model & Sampling Parameters
SAMPLING_BATCH_SIZE = 32 # Increased batch size for faster diffusion sampling using more compute resources
NUM_TIMESTEPS = 1000 # Number of timesteps for diffusion model

# Optimizer/Scheduler Parameters
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.9

# CPU/GPU Optimizations
if DEVICE.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# For CPU, set number of threads (adjust as needed for your system)
# torch.set_num_threads(16) # You mentioned 16 vCores
# torch.set_num_interop_threads(16)

print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"CUDA Matmul TF32: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"CuDNN TF32: {torch.backends.cudnn.allow_tf32}")
    print(f"CuDNN Benchmark: {torch.backends.cudnn.benchmark}")

# Gate mapping for feature extraction
GATE_MAPPING = {
    'h': 1, 'x': 2, 'y': 3, 'z': 4, 's': 5, 't': 6,
    'rx': 7, 'ry': 8, 'rz': 8, 'cx': 8, 'cz': 8 # Simplified for now, 'rz', 'cx', 'cz' often have distinct roles
}

# Action space for RL agent (example: modify complexity, entanglement, gate density)
ACTION_DIM = 3 
OBS_DIM = 33 + MAX_GATES * (3 + MAX_QUBITS) # Observation dimension: circuit features + flattened circuit tensor representation

# Path for saving models
PREDICTOR_MODEL_PATH = "./models/best_predictor_model.pth"
DIFFUSION_MODEL_PATH = "./models/best_diffusion_model.pth"
RL_AGENT_MODEL_PATH = "./models/best_rl_agent_model.pth"
