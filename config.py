# config.py

batch_size = 32
block_size = 128
max_iters = 5000
eval_interval = 200
learning_rate = 3e-4
device = "cuda" if __import__('torch').cuda.is_available() else "cpu" 

n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2