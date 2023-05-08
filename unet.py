import torch

class UNet(torch.nn.Module):
  """ Pytorch implementation of multi-headed UNet like model """
  def __init__(self):
    super(UNet, self).__init__()

    self.relu = torch.nn.ReLU(inplace=True)
    self.elu  = torch.nn.ELU(inplace=True)
    self.sigmoid = torch.nn.Sigmoid()
    self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    self.activation = self.relu

    input_channels = 1
    stage_1_features    = 16
    stage_2_features    = 2 * stage_1_features
    stage_3_features    = 4 * stage_1_features
    stage_4_features    = 8 * stage_1_features

    # Backbone
    self.conv1_1 = torch.nn.Conv2d(  input_channels, stage_1_features, kernel_size=3, stride=1, padding=1)
    self.conv1_2 = torch.nn.Conv2d(stage_1_features, stage_1_features, kernel_size=3, stride=1, padding=1)

    self.conv2_1 = torch.nn.Conv2d(stage_1_features, stage_2_features, kernel_size=3, stride=1, padding=1)
    self.conv2_2 = torch.nn.Conv2d(stage_2_features, stage_2_features, kernel_size=3, stride=1, padding=1)

    self.conv3_1 = torch.nn.Conv2d(stage_2_features, stage_3_features, kernel_size=3, stride=1, padding=1)
    self.conv3_2 = torch.nn.Conv2d(stage_3_features, stage_3_features, kernel_size=3, stride=1, padding=1)

    self.conv4_1 = torch.nn.Conv2d(stage_3_features, stage_4_features, kernel_size=3, stride=1, padding=1)
    self.conv4_2 = torch.nn.Conv2d(stage_4_features, stage_4_features, kernel_size=3, stride=1, padding=1)
    
    # Head 1
    self.deconv1_1 = torch.nn.ConvTranspose2d(    stage_4_features, stage_3_features, kernel_size=2, stride=2)
    self.deconv1_2 = torch.nn.ConvTranspose2d(2 * stage_3_features, stage_2_features, kernel_size=2, stride=2)
    self.deconv1_3 = torch.nn.ConvTranspose2d(2 * stage_2_features, stage_1_features, kernel_size=2, stride=2)
    self.deconv1_4 = torch.nn.ConvTranspose2d(2 * stage_1_features,                1, kernel_size=2, stride=2)

    # Head 2
    self.deconv2_1 = torch.nn.ConvTranspose2d(    stage_4_features, stage_3_features, kernel_size=2, stride=2)
    self.deconv2_2 = torch.nn.ConvTranspose2d(2 * stage_3_features, stage_2_features, kernel_size=2, stride=2)
    self.deconv2_3 = torch.nn.ConvTranspose2d(2 * stage_2_features, stage_1_features, kernel_size=2, stride=2)
    self.deconv2_4 = torch.nn.ConvTranspose2d(2 * stage_1_features,                2, kernel_size=2, stride=2)


  def forward(self, x):
    """ Forward pass that computes output for both heads

    Input
      x: Image pytorch tensor shaped N x 1 x H x W

    Output
      head 1: Output pytorch tensor shaped N x num_output_layers x H x W
      head 2: Output pytorch tensor shaped N x num_output_layers x H x W
    """

    input_size = x.shape[2:]

    # Backbone
    x = self.activation(self.conv1_1(x))
    x = self.activation(self.conv1_2(x))
    stage_1 = self.max_pool(x)

    x = self.activation(self.conv2_1(stage_1))
    x = self.activation(self.conv2_2(x))
    stage_2 = self.max_pool(x)

    x = self.activation(self.conv3_1(stage_2))
    x = self.activation(self.conv3_2(x))
    stage_3 = self.max_pool(x)

    x = self.activation(self.conv4_1(stage_3))
    x = self.activation(self.conv4_2(x))
    stage_4 = self.max_pool(x)

    # Head 1
    head1 = self.activation(self.deconv1_1(stage_4))
    head1 = torch.cat((stage_3, head1), 1)
    head1 = self.activation(self.deconv1_2(head1))
    head1 = torch.cat((stage_2, head1), 1)
    head1 = self.activation(self.deconv1_3(head1))
    head1 = torch.cat((stage_1, head1), 1)
    head1 = self.sigmoid(self.deconv1_4(head1, output_size=input_size))

    # Head 2
    head2 = self.activation(self.deconv2_1(stage_4))
    head2 = torch.cat((stage_3, head2), 1)
    head2 = self.activation(self.deconv2_2(head2))
    head2 = torch.cat((stage_2, head2), 1)
    head2 = self.activation(self.deconv2_3(head2))
    head2 = torch.cat((stage_1, head2), 1)
    head2 = self.sigmoid(self.deconv2_4(head2, output_size=input_size))

    return (head1, head2)

def main():
    model = UNet()
    output = model(torch.rand(2, 1, 256, 256))
    print(output)

if __name__ == '__main__':
    main()
