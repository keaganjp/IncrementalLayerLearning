## Incremental Layer Learning

### Instructions for running the backpropagation benchmarking code:
`python trainBP.py -dataset <name_of_dataset> -net <name_of_net>`

Datasets available:
- MNIST 
- FashionMNIST
- CIFAR10
- CIFAR100
- CUB 
- Food101

Networks available: 
- vgg11
- vgg16 
- resnet18 

To change the number of epochs, edit the EPOCH variable in conf/global_settings.py

Optional parameters: 
- `-gpu True` : to use GPU (default set to false) 
- `-b` : specify the batch-size (delault is 128) 

### Instructions for running the ILL benchmarking code:
`python trainILL.py -dataset <name_of_dataset> -net <name_of_net>` (this can train a simple conv, vgg11 and vgg16 networks)

`python resnet_ILL.py -dataset <name_of_dataset>` (this is for resnet18 exclusively)

Datasets available:
- MNIST 
- FashionMNIST
- CIFAR10
- CIFAR100
- CUB 
- Food101

Networks available: 
- simple (mostly for debugging) 
- vgg11
- vgg16 

Optional parameters: 
- `-gpu cuda:0` : to use GPU (default set to 'cpu') 
- `-b` : specify the batch-size (delault is 128) 

To change the number of epochs, you need to edit this in the ILLLayer.py file. Edit the num_epoch variable in the class definition of ILLConv2D and ILLLinearLayer. The learning rate can be changed in a similar manner. 

