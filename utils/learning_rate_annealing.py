import numpy as np
import torch.optim as optim

learning_rate_annealer = lambda epoch: max(np.exp(-epoch / 8 - 2), 0.00025)
scheduler = optim.LambdaLR(optimizer, 
                     lr_lambda=learning_rate_annealer)
scheduler.step()
