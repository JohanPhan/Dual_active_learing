import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
class Embedded_distance:
    def __init__(self, memory_len):
        self.embeded_tensor_Fifo = []
        self.memory_len = memory_len
    def get_variance(self):
        temp_var = np.var((np.array(self.embeded_tensor_Fifo)),axis = 0)
        return temp_var
    def get_variance_sum(self):
        temp_var = np.var((np.array(self.embeded_tensor_Fifo)))
        return temp_var
    def get_distance(self):
        temp_distance = pdist(np.array(self.embeded_tensor_Fifo))
        return temp_distance
    def get_var_distance(self):
        temp_distance = np.var(pdist(np.array(self.embeded_tensor_Fifo)))
        return temp_distance
    def get_distance_min(self):
        temp_distance = pdist(np.array(self.embeded_tensor_Fifo))
        return np.min(temp_distance[np.nonzero(temp_distance)])
    def get_distance_sum(self):
        temp_distance = pdist(np.array(self.embeded_tensor_Fifo))
        return temp_distance.sum()
    def push(self, embedded_target):
        if type(embedded_target) == torch.Tensor:
            embedded_target = embedded_target.numpy()
            self.embeded_tensor_Fifo.append(embedded_target)
        else:
            self.embeded_tensor_Fifo.append(embedded_target)
        if (len(self.embeded_tensor_Fifo) > (self.memory_len-1)):
            self.embeded_tensor_Fifo.pop(0)
    def temp_push(self, embedded_target):
        if type(embedded_target) == torch.Tensor:
            embedded_target = embedded_target.numpy()
            self.embeded_tensor_Fifo.append(embedded_target)
        else:
            self.embeded_tensor_Fifo.append(embedded_target)
    def temp_clear(self):
        self.embeded_tensor_Fifo.pop()
    def size(self):
        return len(self.embeded_tensor_Fifo)
    