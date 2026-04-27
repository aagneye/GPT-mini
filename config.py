# config.py

batch_size = 32
block_size = 128
max_iters = 50000  # Increased for larger OpenWebText dataset
eval_interval = 200
learning_rate = 3e-4
device = "cuda" if __import__('torch').cuda.is_available() else "cpu" 

n_embd = 256
n_head = 8
n_layer = 6
dropout = 0.2