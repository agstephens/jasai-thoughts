import torch
assert torch.cuda.is_available()

import tensorflow as tf
assert len(tf.config.list_physical_devices('GPU')) > 0

