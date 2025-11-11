import torch
import random
import numpy as np

# シード値の設定
def set_seed(seed: int):
    random.seed(seed) # Pythonのランダムシード
    np.random.seed(seed) # NumPyのランダムシード
    torch.manual_seed(seed) # PyTorchのランダムシード（CPU用）
    torch.cuda.manual_seed(seed) # PyTorchのランダムシード（GPU用）