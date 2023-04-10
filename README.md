## Incremental Layer Learning

### Running the code:
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

Optional parameters: 
- `-gpu True` : to use GPU (default set to false) 
- `-b` : specify the batch-size (delault is 128) 
