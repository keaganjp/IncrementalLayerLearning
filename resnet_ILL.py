"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import os
import sys
import argparse
import time
from datetime import datetime
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from conf import settings
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import get_network, get_training_dataloader, get_test_dataloader
from ILL_layers import * 
    
class ILLResnet18(nn.Module):
    def __init__(self, args, input_size, linear_size=32768, num_classes=100):
        super().__init__()
        self.device = args.gpu
        sample_input_tensor = torch.rand(1, input_size[0], input_size[1], input_size[2]).to(self.device)

        self.avgpool = nn.AvgPool2d(1, 1)
        self.initial_conv = ILLConv2D(input_size[0], 64, 3, sample_input_tensor, bias=False, device=self.device)
        intermediate_output, _ = self.initial_conv(sample_input_tensor)

        # Block 1
        self.b1_c1 = ILLConv2D(64, 64, 3, intermediate_output, bias=False, device=self.device)
        intermediate_output, _ = self.b1_c1.forward(intermediate_output)
        self.b1_n1 = nn.BatchNorm2d(64, device=self.device)
        intermediate_output = self.b1_n1(intermediate_output)

        self.b1_c2 = ILLConv2D(64, 64, 3, intermediate_output, padding=1, bias=False, device=self.device)
        intermediate_output, _ = self.b1_c2.forward(intermediate_output)
        self.b1_n2 = nn.BatchNorm2d(64, device=self.device)
        intermediate_output = self.b1_n2(intermediate_output)

        self.b1_c3 = ILLConv2D(64, 64, 3, intermediate_output, padding=1, bias=False, device=self.device)
        intermediate_output, _ = self.b1_c3.forward(intermediate_output)
        self.b1_n3 = nn.BatchNorm2d(64, device=self.device)
        intermediate_output = self.b1_n3(intermediate_output)

        self.b1_c4 = ILLConv2D(64, 64, 3, intermediate_output, padding=1, bias=False, device=self.device)
        intermediate_output, _ = self.b1_c4.forward(intermediate_output)
        self.b1_n4 = nn.BatchNorm2d(64, device=self.device)
        intermediate_output = self.b1_n4(intermediate_output)

        # Block 2 
        self.b2_c1 = ILLConv2D(64, 128, 3, intermediate_output, padding=1, bias=False, device=self.device)
        intermediate_output, _ = self.b2_c1.forward(intermediate_output)
        self.b2_n1 = nn.BatchNorm2d(128, device=self.device)
        intermediate_output = self.b2_n1(intermediate_output)

        self.b2_c2 = ILLConv2D(128, 128, 3, intermediate_output, padding=1, bias=False, device=self.device)
        intermediate_output, _ = self.b2_c2.forward(intermediate_output)
        self.b2_n2 = nn.BatchNorm2d(128, device=self.device)
        intermediate_output = self.b2_n2(intermediate_output)

        self.b2_c3 = ILLConv2D(128, 128, 3, intermediate_output, padding=1, bias=False, device=self.device)
        intermediate_output, _ = self.b2_c3.forward(intermediate_output)
        self.b2_n3 = nn.BatchNorm2d(128, device=self.device)
        intermediate_output = self.b2_n3(intermediate_output)

        self.b2_c4 = ILLConv2D(128, 128, 3, intermediate_output, padding=1, bias=False, device=self.device)
        intermediate_output, _ = self.b2_c4.forward(intermediate_output)
        self.b2_n4 = nn.BatchNorm2d(128, device=self.device)
        intermediate_output = self.b2_n4(intermediate_output)

        # Block 3 
        self.b3_c1 = ILLConv2D(128, 256, 3, intermediate_output, padding=1, bias=False, device=self.device)
        intermediate_output, _ = self.b3_c1.forward(intermediate_output)
        self.b3_n1 = nn.BatchNorm2d(256, device=self.device)
        intermediate_output = self.b3_n1(intermediate_output)

        self.b3_c2 = ILLConv2D(256, 256, 3, intermediate_output, padding=1, bias=False, device=self.device)
        intermediate_output, _ = self.b3_c2.forward(intermediate_output)
        self.b3_n2 = nn.BatchNorm2d(256, device=self.device)
        intermediate_output = self.b3_n2(intermediate_output)

        self.b3_c3 = ILLConv2D(256, 256, 3, intermediate_output, padding=1, bias=False, device=self.device)
        intermediate_output, _ = self.b3_c3.forward(intermediate_output)
        self.b3_n3 = nn.BatchNorm2d(256, device=self.device)
        intermediate_output = self.b3_n3(intermediate_output)

        self.b3_c4 = ILLConv2D(256, 256, 3, intermediate_output, padding=1, bias=False, device=self.device)
        intermediate_output, _ = self.b3_c4.forward(intermediate_output)
        self.b3_n4 = nn.BatchNorm2d(256, device=self.device)
        intermediate_output = self.b3_n4(intermediate_output)
     
        # Block 4 
        self.b4_c1 = ILLConv2D(256, 512, 3, intermediate_output, bias=False, device=self.device)
        intermediate_output, _ = self.b4_c1.forward(intermediate_output)
        self.b4_n1 = nn.BatchNorm2d(512, device=self.device)
        intermediate_output = self.b4_n1(intermediate_output)

        self.b4_c2 = ILLConv2D(512, 512, 3, intermediate_output, bias=False, device=self.device)
        intermediate_output, _ = self.b4_c2.forward(intermediate_output)
        self.b4_n2 = nn.BatchNorm2d(512, device=self.device)
        intermediate_output = self.b4_n2(intermediate_output)

        self.b4_c3 = ILLConv2D(512, 512, 3, intermediate_output, bias=False, device=self.device)
        intermediate_output, _ = self.b4_c3.forward(intermediate_output)
        self.b4_n3 = nn.BatchNorm2d(512, device=self.device)
        intermediate_output = self.b4_n3(intermediate_output)

        self.b4_c4 = ILLConv2D(512, 512, 3, intermediate_output, bias=False, device=self.device)
        intermediate_output, _ = self.b4_c4.forward(intermediate_output)
        self.b4_n4 = nn.BatchNorm2d(512, device=self.device)
        intermediate_output = self.b4_n4(intermediate_output)

        pooledOut = self.avgpool(intermediate_output)
        self.flat = nn.Flatten()

        # Classifier at the end 
        self.fc = ILLLinearLayer(pooledOut.flatten(start_dim=1).shape[1], out_features=num_classes, device=self.device)

        self.layers = [self.initial_conv,
                       self.b1_c1,
                       self.b1_n1,
                       self.b1_c2,
                       self.b1_n2,
                       self.b1_c3,
                       self.b1_n3,
                       self.b1_c4,
                       self.b1_n4,
                       self.b2_c1,
                       self.b2_n1,
                       self.b2_c2,
                       self.b2_n2,
                       self.b2_c3,
                       self.b2_n3,
                       self.b2_c4,
                       self.b2_n4,
                       self.b3_c1,
                       self.b3_n1,
                       self.b3_c2,
                       self.b3_n2,
                       self.b3_c3,
                       self.b3_n3,
                       self.b3_c4,
                       self.b3_n4,
                       self.b4_c1,
                       self.b4_n1,
                       self.b4_c2,
                       self.b4_n2,
                       self.b4_c3,
                       self.b4_n3,
                       self.b4_c4,
                       self.b4_n4,
                       self.avgpool,
                       self.flat, 
                       self.fc
        ]

        self.trainableList = [self.initial_conv,
                       self.b1_c1,
                       self.b1_c2,
                       self.b1_c3,
                       self.b1_c4,
                       self.b2_c1,
                       self.b2_c2,
                       self.b2_c3,
                       self.b2_c4,
                       self.b3_c1,
                       self.b3_c2,
                       self.b3_c3,
                       self.b3_c4,
                       self.b4_c1,
                       self.b4_c2,
                       self.b4_c3,
                       self.b4_c4,
                       self.fc
        ]

        self.skip_list = {0:5,
                          4:9,
                          8:13,
                          12:17,
                          16:21,
                          20:25,
                          24:29,
                          28:33}

    def train(self, dataloader, writer, filename):
        device = self.device
        for layer in range(len(self.layers)):
            if self.layers[layer] not in self.trainableList:
                print("Skipping untrainable layer ", layer + 1)
            else:
                print("Training Layer", layer + 1)
                previousLayers = self.layers[:layer]
                self.layers[layer].trainLayer(dataloader, previousLayers, device, writer, filename, self.skip_list)
            acc, acc_final = self.eval_training(training_loader, 'true')
            acc_test, acc_test_final = self.eval_training(test_loader, 'false')
            print("Writing epoch data")
            with open(filename, 'a') as f:
                f.write("Training accuracy is {:.3f} after training layer {} \n".format(acc*100.0, layer))
                f.write("Training accuracy is {:.3f} after training layer {} \n while considering only the final predictor layer".format(acc_final*100.0, layer))
                f.write("Testing accuracy is {:.3f} after training layer {} \n".format(acc_test*100.0, layer))
                f.write("Testing accuracy is {:.3f} after training layer {} \n while considering only the final predictor layer".format(acc_test_final*100.0, layer))
    
    # Predict on a batch
    def predict(self, x):
        # get per layer logits
        layerPreds = []
        layerOutputs = []
        inputs = x
        
        layer_counter = 0
        skip_input = None
        skip_target = None
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.Flatten)  or isinstance(layer, nn.AvgPool2d) or isinstance(layer, nn.BatchNorm2d):
                if skip_target is not None and skip_target == layer_counter:
                    inputs =  layer.forward(inputs+skip_input)
                else:
                    inputs = layer.forward(inputs)
            else:
                if skip_target is not None and skip_target == layer_counter:
                    inputs, predictor_output =  layer.forward(inputs+skip_input)
                else:
                    inputs, predictor_output = layer.forward(inputs)
                layerOutputs.append(predictor_output)
                layerPreds.append(F.softmax(predictor_output, dim=1))
            if self.skip_list is not None:
                if layer_counter in self.skip_list: # the current layer output is to be saved
                    skip_input = inputs
            layer_counter +=1
        finallayerpred = layerPreds[-1]
        #print(layerPreds)
        layerPreds = torch.stack(layerPreds)
        # Add up per layer softmax
        combinedPred = torch.sum(layerPreds, dim=0)
        finalPred = F.softmax(combinedPred, dim=1)
        return inputs, finalPred, finallayerpred
    
    @torch.no_grad()        
    def eval_training(self, test_loader, check_train, tb=True):

        start = time.time()

        test_loss = 0.0 # cost function error
        correct = 0.0
        correct_finallayer = 0.0
        criterion = nn.CrossEntropyLoss()
        for  i, data in enumerate(test_loader):
            images, labels = data
            if self.device == 'cuda:0':
                images = images.cuda()
                labels = labels.cuda()

            outputs, preds, finallayerpreds = self.predict(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            #_, preds = self.predict(images)
            _, predicted = torch.max(preds, 1)
            _, predicted_final = torch.max(finallayerpreds, 1)
            correct += (predicted == labels).sum().item()
            correct_finallayer += (predicted_final == labels).sum().item()
            if (i == 0): 
                break
        finish = time.time()
        if self.device == 'cuda:0':
            print('GPU INFO.....')
            print(torch.cuda.memory_summary(), end='')
        print('Evaluating Network.....')
        print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s (combined prediction)'.format(
            test_loss / len(test_loader.dataset),
            correct / len(test_loader.dataset),
            finish - start
        ))   
        print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s (final layer prediction)'.format(
            test_loss / len(test_loader.dataset),
            correct_finallayer / len(test_loader.dataset),
            finish - start
        ))      

        #add informations to tensorboard
        if check_train == 'false':
            writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset))
            writer.add_scalar('Test/Accuracy', correct / len(test_loader.dataset))
        else: 
            writer.add_scalar('Train/Accuracy', correct/ len(test_loader.dataset))
            writer.add_scalar('Train/Accuracy_with_finalPred', correct_finallayer/ len(test_loader.dataset))
        return correct / len(test_loader.dataset), correct_finallayer / len(test_loader.dataset)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, required=True, help='select dataset')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-gpu', default='cpu', help='use gpu or not')
    args = parser.parse_args()

    torch.manual_seed(1234)
    epochs = 2 #This is more of a dummy variable. You need to adjust this according to the layer epochs in ILLConv2D and ILLLinearlayer 
    linear_size = 512 
    input_size = [3, 256, 256]
    linear_size = 32768

    if args.dataset == 'FashionMNIST':
        num_classes = 10
        linear_size = 512
        input_size = [3, 32, 32]
        training_loader = get_training_dataloader(args,
        settings.FASHIONMNIST_TRAIN_MEAN,
        settings.FASHIONMNIST_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
        )
        test_loader = get_test_dataloader(args,
        settings.FASHIONMNIST_TRAIN_MEAN,
        settings.FASHIONMNIST_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
        )
    
    elif args.dataset == 'MNIST':
        num_classes = 10
        linear_size = 512 
        input_size = [3, 32, 32]
        training_loader = get_training_dataloader(args,
        settings.MNIST_TRAIN_MEAN,
        settings.MNIST_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
        )
        test_loader = get_test_dataloader(args,
        settings.MNIST_TRAIN_MEAN,
        settings.MNIST_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
        )

    elif args.dataset == 'CIFAR10':
        num_classes = 10
        training_loader = get_training_dataloader(args,
        settings.CIFAR10_TRAIN_MEAN,
        settings.CIFAR10_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
        )
        test_loader = get_test_dataloader(args,
        settings.CIFAR10_TRAIN_MEAN,
        settings.CIFAR10_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
        )

    elif args.dataset == 'CIFAR100':
        training_loader = get_training_dataloader(args,
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
        )
        test_loader = get_test_dataloader(args,
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
        )
    
    elif args.dataset == 'Food101':
        num_classes = 101
        training_loader = get_training_dataloader(args,
        settings.CALTECH256_TRAIN_MEAN,
        settings.CALTECH256_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
        )
        test_loader = get_test_dataloader(args,
        settings.CALTECH256_TRAIN_MEAN,
        settings.CALTECH256_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
        )
    
    elif args.dataset == 'CUB':
        num_classes = 200
        training_loader = get_training_dataloader(args,
        settings.CUB_TRAIN_MEAN,
        settings.CUB_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
        )
        test_loader = get_test_dataloader(args,
        settings.CUB_TRAIN_MEAN,
        settings.CUB_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
        )
    else: 
        raise Exception('Dataset not valid')
    
    net = ILLResnet18(args, input_size=input_size, linear_size=linear_size, num_classes=num_classes)

    filename = 'ILL_' + args.dataset + '_' + 'resnet18' + '.txt'
    with open(filename, "w") as f:
        f.write("Testing\n")
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, settings.TIME_NOW))
    '''input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu == 'gpu':
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)'''

    net.train(training_loader, writer, filename)

    train_acc = net.eval_training(training_loader, 'true')
    train_accuracy = round(train_acc*100, 3)
    print("Train Accuracy:", train_accuracy, '%')
    print("Train Error   :", round(100 - train_accuracy,3), '%')
    
    with open(filename, "w") as f:
        f.write("The final training accuracy after {} epochs is: {:.3f}\n".format(epochs, train_accuracy))

    test_acc = net.eval_training(test_loader, 'false')
    test_accuracy = round(test_acc*100, 3)
    print("Test Accuracy:", test_accuracy, '%')
    print("Test Error   :", round(100 - test_accuracy,3), '%')

    pickle_file = 'ILL_'+ args.dataset + '_' + 'resnet18' + '.pickle'
    with open(pickle_file, 'wb') as p: 
        pickle.dump(net, p)
    writer.close()




