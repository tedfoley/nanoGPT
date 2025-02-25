# config/test_no_features.py
from config.train_gpt2_compact import *  # Import base config

# Override with specific configuration
wandb_log = True
wandb_project = 'nanoGPT-algo-test'
wandb_run_name = 'no_features'

# Feature toggles
use_layer_norm = False
use_rope = True
use_flash_attn = False
use_sparse_attn = False

# Use smaller model for quick testing
n_layer = 4
block_size = 256
batch_size = 8
max_iters = 100