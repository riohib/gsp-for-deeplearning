import networks.resnet as resnet
import networks.vgg as vgg


def model(arch):
    if 'resnet' in arch:
        network = resnet.__dict__[arch]()
    if 'vgg' in arch:
        network = vgg.__dict__[arch]()
    
    return network