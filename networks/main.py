from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder

def build_autoencoder():
    """Builds the corresponding autoencoder network."""


    ae_net = None
    ae_net = CIFAR10_LeNet_Autoencoder()
    return ae_net
