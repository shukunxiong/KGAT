#--------------------------------------------#
#   This part of the code is used to view the network structure
#--------------------------------------------#
import torch
from thop import clever_format, profile

from nets.yolo import YoloBody

if __name__ == "__main__":
    input_shape     = [640, 640]
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes     = 80
    phi             = 's'
    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m       = YoloBody(input_shape, num_classes, phi, False).to(device)
    for i in m.children():
        print(i)
        print('==============================')
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(m.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   Multiply flops by 2 because the profile does not count convolution as two operations
    #   Some papers count convolution as two operations: multiplication and addition. In such cases, multiply by 2.
    #   Some papers only consider the number of multiplication operations and ignore addition. In such cases, do not multiply by 2.
    #   This code chooses to multiply by 2, referring to YOLOX.
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
