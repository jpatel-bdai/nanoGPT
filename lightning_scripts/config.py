# Training hyperparameters

DATA_DIR = "data/openwebtext"
GRADIENT_ACCUMULATION_STEPS = 1  # used to simulate larger batch sizes
VOCAB_SIZE = 50304
# adamw optimizer
LEARNING_RATE = 6e-4  # max learning rate
MAX_ITERS = 600000  # total number of training iterations
WEIGHT_DECAY = 1e-2
BETA1 = 0.9
BETA2 = 0.95
GRAD_CLIP = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
DECAY_LR = True  # whether to decay the learning rate
WARMUP_ITERS = 2000  # how many steps to warm up for
LR_DECAY_ITERS = 600000  # should be ~= max_iters per Chinchilla
MIN_LR = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# Model parameters
BATCH_SIZE = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
BLOCK_SIZE = 1024
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768
DROPOUT = 0.0  # for pretraining 0 is good, for finetuning try 0.1+

# Checkpointing parameters
OUT_DIR = "out"  # For resuming training from a checkpoint
ALWAYS_SAVE_CHECKPOINT = True  # if True, always save a checkpoint after each eval
INTI_FROM = "scratch"  # 'scratch' or 'resume' or 'gpt2*'

# Evaluation hyperparameters
EVAL_INTERVAL = 2000  # check_val_every_n_epoch in trainer
LOG_INTERVAL = 1  # define in training_step
EVAL_ITERS = 200  # Number of iterations to eval on
EVAL_ONLY = False  # if True, script exits right after the first eval

# Trainer parameters
ACCELERATOR = "gpu"
DEVICES = 8

NUM_WORKERS = 4
# DDP settings - Not sure if these will be needed
BACKEND = "nccl"  # 'nccl', 'gloo', etc.
# system
DEVICE = (
    "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)
DTYPE = "bfloat16"  # 'float32' or 'bfloat16'
COMPILE = True  # use PyTorch 2.0 to compile the model to be faster
