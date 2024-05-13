import torch
import torch.nn as nn


class FCNN(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_1 = 2 * 512
        self.hidden_2 = 2 * 256
        self.hidden_3 = 2 * 128
        self.hidden_4 = 2 * 64

        self.seq = nn.Sequential(nn.Linear(self.input_size, self.hidden_1, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_1, self.hidden_2, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_2, self.hidden_3, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_3, self.hidden_4, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_4, self.output_size, bias=True),
                                 )

    def forward(self, x):
        out = self.seq(x)
        return out


class SharedNN(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = 1024

        self.shared = nn.Sequential(nn.Linear(self.input_size, self.hidden_size, bias=True), nn.ReLU())

        self.private1 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size, bias=True), nn.ReLU())
        self.private2 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size, bias=True), nn.ReLU())
        self.private3 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size, bias=True), nn.ReLU())
        self.private4 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size, bias=True), nn.ReLU())
        self.private5 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size, bias=True), nn.ReLU())

        self.private1_1 = nn.Linear(self.hidden_size, 1, bias=True)
        self.private2_1 = nn.Linear(self.hidden_size, 1, bias=True)
        self.private3_1 = nn.Linear(self.hidden_size, 1, bias=True)
        self.private4_1 = nn.Linear(self.hidden_size, 1, bias=True)
        self.private5_1 = nn.Linear(self.hidden_size, 1, bias=True)

    def forward(self, x):
        out = self.shared(x)

        out1 = self.private1(out)
        out2 = self.private2(out)
        out3 = self.private3(out)
        out4 = self.private4(out)
        out5 = self.private5(out)

        out1 = self.private1_1(out1)
        out2 = self.private2_1(out2)
        out3 = self.private3_1(out3)
        out4 = self.private4_1(out4)
        out5 = self.private5_1(out5)

        out = torch.stack([out1, out2, out3, out4, out5], 0)
        out = torch.transpose(out, 0, 1)
        dim1 = out.size(dim=0)
        dim2 = out.size(dim=1)
        out = torch.reshape(out, (dim1, dim2))

        return out