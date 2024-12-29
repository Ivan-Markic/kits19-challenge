import torch
import torch.nn as nn

class SimpleUNet(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch=32):  # Kept base_ch=32
        super(SimpleUNet, self).__init__()
        # Encoder path - First layer handles arbitrary input channels
        self.inc = self._Conv(in_ch, base_ch)  # This will now handle any number of input channels
        self.down1 = self._Down(base_ch, base_ch * 2)
        self.down2 = self._Down(base_ch * 2, base_ch * 4)
        self.down3 = self._Down(base_ch * 4, base_ch * 8)
        
        # Decoder path remains the same
        self.up1 = self._Up(base_ch * 8, base_ch * 4)
        self.up2 = self._Up(base_ch * 4, base_ch * 2)
        self.up3 = self._Up(base_ch * 2, base_ch)
        self.outc = nn.Conv2d(base_ch, out_ch, kernel_size=1)
    
    def forward(self, x):
        # Forward pass remains the same
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return {'output': x}  # Return dict to match DenseUNet format
    
    class _Down(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(SimpleUNet._Down, self).__init__()
            self.mp = nn.MaxPool2d(kernel_size=2)
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
        
        def forward(self, x):
            x = self.mp(x)
            x = self.conv(x)
            return x
    
    class _Up(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(SimpleUNet._Up, self).__init__()
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # Adjust input channels to account for concatenation
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch + out_ch, out_ch, kernel_size=3, padding=1),  # in_ch + out_ch due to skip connection
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
        
        def forward(self, x1, x2):
            x1 = self.up(x1)
            x = torch.cat([x2, x1], dim=1)
            x = self.conv(x)
            return x
    
    class _Conv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(SimpleUNet._Conv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
        
        def forward(self, x):
            return self.conv(x)


if __name__ == '__main__':
    from torchsummary import summary
    
    # Test with 3 stacked slices (3 channels)
    net = SimpleUNet(in_ch=3, out_ch=3).cuda()
    summary(net, (3, 320, 320))