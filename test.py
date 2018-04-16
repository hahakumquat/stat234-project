#!/usr/bin/env python3.5

import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
print(use_cuda)
