import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, input_shape, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.l1 = nn.Linear(input_shape[1], 256)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(256, 32)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs):
        logits = self.relu1(self.l1(inputs))
        logits = self.relu2(self.l2(logits))
        logits = self.sigmoid(self.l3(logits))
        return logits

if __name__ == '__main__':
    pass