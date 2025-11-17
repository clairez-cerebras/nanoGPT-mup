import torch

class Config:
    """
    Configuration class for managing hyperparameters and settings.
    """

    def __init__(self):
        self._load_defaults()
    
    def _load_defaults(self):
        """
        Load default hyperparameters.
        """

        # I/O
        self.out_dir = 'out'
        self.eval_interval = 2000
        self.log_interval = 1
        self.eval_iters = 200
        self.eval_only = False # if True, script exits right after the first eval
        self.skip_val_loss = False # If True, will only measure train loss
        self.always_save_checkpoint = True # if True, always save a checkpoint after each eval
        self.never_save_checkpoint = False # if True, never save a checkpoint
        self.init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

        # wandb logging
        self.wandb_log = False # disabled by default
        self.wandb_project = 'owt'
        self.wandb_run_name = 'gpt2' # 'run' + str(time.time())

        # csv logging
        self.csv_log = True # If enabled, logs stats to a csv file

        # data
        self.dataset = 'openwebtext'
        self.gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
        self.batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
        self.block_size = 1024

        # model
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
        self.bias = False # do we use bias inside LayerNorm and Linear layers?
        self.init_std = 0.02 # Initialization standard deviation for weights

        # adamw optimizer
        self.learning_rate = 6e-4 # max learning rate
        self.max_iters = 600000 # total number of training iterations
        self.weight_decay = 1e-1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
        self.adam_eps = 1e-12

        # learning rate decay settings
        self.decay_lr = True # whether to decay the learning rate
        self.warmup_iters = 2000 # how many steps to warm up for
        self.lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
        self.min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

        # mup settings
        self.mup_enabled = False # Whether to use muP. If False then all other mup variables are ignored
        self.mup_disable_attention_scaling = False # Uses 1/sqrt(d_head) attn scaling instead of 1/d_head (Only needed for the step-by-step coord check in the blog)
        self.mup_disable_hidden_lr_scaling = False # Disables muP hidden LR adjustment (Only needed for the step-by-step coord check in the blog)
        self.mup_width_multiplier = 1.0 # mup_width_multiplier = width / base_width where base_width is typically 256
        self.mup_input_alpha = 1.0 # Optional tunable multiplier applied to input embedding forward pass output
        self.mup_output_alpha = 1.0 # Optional tunable multiplier applied to output unembedding forward pass output
        self.mup_enable_coord_check_logging = False # If True will track the output.abs().mean() of various layers throughout training

        # Depth scaling settings
        self.depth_alpha_enabled = False 
        self.depth_multiplier = 1.0
        self.depth_alpha_exp = 1.0

        # seed
        self.seed = 1337

        # DDP settings
        self.backend = 'nccl' # 'nccl', 'gloo', etc.

        # system
        self.device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
        self.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
        self.compile = False # use PyTorch 2.0 to compile the model to be faster
    

    def _update_from_dict(self, config_dict):
        """
        Update configuration parameters from a dictionary.
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                default_value = getattr(self, key)
                default_type = type(default_value)
                setattr(self, key, default_type(value))
            else:
                raise KeyError(f"Invalid configuration key: {key}")
    