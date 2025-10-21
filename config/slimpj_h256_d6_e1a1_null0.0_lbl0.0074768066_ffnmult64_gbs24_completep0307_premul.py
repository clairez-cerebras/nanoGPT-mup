outdir = "out/slimpj_h256_d6_e1a1_null0.0_lbl0.0074768066_ffnmult64_gbs24_completep0307_premul"
batch_size = 8 # 24 
gradient_accumulation_steps = 8 # 24
block_size = 2048 # 8192
n_layer = 6
n_head = 4
n_embd = 256
bias = True
learning_rate = 0.00390625
weight_decay = 0.34394313666861864
adam_eps = 1.0e-16
warmup_iters = 529
lr_decay_iters = 529 + 4762
min_lr = 0.0
max_iters = 529 + 4762
eval_iters = 1881

mup_enabled = True
depth_alpha_enabled = True

dtype = 'float32'
compile = False
