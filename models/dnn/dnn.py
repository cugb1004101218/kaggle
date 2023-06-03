import torch
import torch.nn as nn
import torch.optim as optim

class DNN(nn.Module):
    def __init__(self, input_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.l1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(256, 32)
        self.l3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs):
        logits = self.relu1(self.l1(inputs))
        logits = self.l2(logits)
        logits = self.sigmoid(self.l3(logits))
        return logits

if __name__ == '__main__':
    test = torch.rand(2, 1000)
    label = torch.tensor([[1.0], [0.0]])

    model = DNN(1000)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(),lr= 0.01 , momentum= 0.5)

    data,target = test, label
    for i in range(1000):
        optimizer.zero_grad()
        output = model(data)
        print(target.shape)
        print(output)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        print(loss.item())