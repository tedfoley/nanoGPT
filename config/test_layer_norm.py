# config/test_no_features.py
from config.train_gpt2_compact import *  # Import base config

# Override with specific configuration
wandb_log = True
wandb_project = 'nanoGPT-algo-test'
wandb_run_name = 'layer_norm'

# Feature toggles
use_layer_norm = True
use_rope = False
use_flash_attn = False
use_sparse_attn = False
