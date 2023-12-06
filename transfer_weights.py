import torch
import torch.nn as nn
import numpy as np

print('\n-------------------------Create and Save Backbone --------------------------------\n')
class Net1(nn.Module):
    def __init__(self, pretrained=None, extend=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1,1,3,1)

        if pretrained:
            self.load_state_dict(torch.load(pretrained))

        if extend:
            self.extend()
            self.forward = self.extended_forward

    def forward(self, x):
        out = self.conv1(x)
        return [out]

    def extend(self):
        self.conv2 = nn.Conv2d(1,1,3,1)
    
    def extended_forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        return [out1, out2]

net1 = Net1().cuda()

torch.save(net1.state_dict(), 'Net1.pth')

print('\n------------------------- Extend Model and Print Outputs--------------------------------\n')

net2 = Net1('Net1.pth', True).cuda()
print(net1.state_dict())
print(net2.state_dict())

print('\n------------------------- Perform Training Cycle and Print State Dict --------------------------------\n')

test_input = torch.tensor(np.ones((1,1,5,5), dtype=np.float32)).cuda()
loss_func = nn.MSELoss().cuda()
optimizer = torch.optim.SGD(net2.conv2.parameters(), lr=0.001, momentum=0.9)
optimizer.zero_grad()
out = net2(test_input)
gt1 = torch.rand(out[0].shape).cuda()
gt2 = torch.rand(out[1].shape).cuda()
loss1 = loss_func(out[0], gt1)
loss2 = loss_func(out[1], gt2)
(loss1 + loss2).backward()
optimizer.step()

print(net2.state_dict())


