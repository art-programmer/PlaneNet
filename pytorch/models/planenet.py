from models.drn import drn_d_54
from models.modules import *

class PlaneNet(nn.Module):
    def __init__(self, options):
        super(PlaneNet, self).__init__()
        
        self.options = options        
        self.drn = drn_d_54(pretrained=True, out_map=32, num_classes=-1, out_middle=False)
        self.pool = torch.nn.AvgPool2d((32 * options.height / options.width, 32))
        self.plane_pred = nn.Linear(512, options.numOutputPlanes * 3)
        self.pyramid = PyramidModule(options, 512, 128)
        self.feature_conv = ConvBlock(1024, 512)
        self.segmentation_pred = nn.Conv2d(512, options.numOutputPlanes + 1, kernel_size=1)
        self.depth_pred = nn.Conv2d(512, 1, kernel_size=1)
        self.upsample = torch.nn.Upsample(size=(options.outputHeight, options.outputWidth), mode='bilinear')
        return

    def forward(self, inp):
        features = self.drn(inp)
        planes = self.plane_pred(self.pool(features).view((-1, 512))).view((-1, self.options.numOutputPlanes, 3))
        features = self.pyramid(features)
        features = self.feature_conv(features)
        segmentation = self.upsample(self.segmentation_pred(features))
        depth = self.upsample(self.depth_pred(features))
        return planes, segmentation, depth
