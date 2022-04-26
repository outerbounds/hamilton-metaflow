from torch import nn
import torch.nn.functional as F


class SkorchModule(nn.Module):
    def __init__(self, num_input_feats=59, num_units=10, nonlin=F.relu, num_classes=3):
        super(SkorchModule, self).__init__()

        self.dense0 = nn.Linear(num_input_feats, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, num_classes)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X))
        return X