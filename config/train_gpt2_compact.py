# config for training compact GPT-2 (~50M) - optimized for faster training
# launch as: $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_compact.py

# wandb logging setup
wandb_log = True
wandb_project = 'owt'
wandb_run_name = 'gpt2-50M-compact'

# Model architecture changes
# Total params reduced from 124M to ~50M
n_layer = 8        # Down from 12
n_head = 8         # Down from 12
n_embd = 512       # Down from 768
block_size = 512   # Reduced context length
dropout = 0.2      # Increased from 0.0 for regularization

# Batch size configuration
# Original: 12 * 1024 * 40 = 491,520
# New: 64 * 512 * 8 = 262,144 (still substantial)
batch_size = 64
gradient_accumulation_steps = 8

# Training schedule
# Reduced from original 600k iterations due to smaller model
max_iters = 400000
warmup_iters = 2000
lr_decay_iters = 400000

# Learning rate configuration
# Scaled based on model size: 6e-4 * sqrt(512/768)
learning_rate = 6e-4
min_lr = 6e-5    # 10% of base lr
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Add cosine learning rate schedule
lr_decay_iters = 400000
decay_lr = True
lr_decay_type = 'cosine'

# Evaluation settings
eval_interval = 1000
eval_iters = 200
log_interval = 10

# Regularization
weight_decay = 1e-1
