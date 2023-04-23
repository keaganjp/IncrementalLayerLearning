import os
import sys
import argparse
import time
from datetime import datetime
import pickle 
from tqdm import tqdm 

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

# mfw security violation
import requests
requests.packages.urllib3.disable_warnings()

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# The overall FF Network

class ILLNet(torch.nn.Module):
    def __init__(self, args, input_size, linear_size=32768, num_classes=100):
        super().__init__()
        self.layers = []
        self.pool = nn.MaxPool2d(2, 2)
        device = args.gpu
        sample_input = torch.rand(1, input_size[0], input_size[1], input_size[2]).to(device)

        if args.net == 'simple': 
            self.conv1 = ILLConv2D(3, 64, 3, sample_input, device=device)
            conv1_output_eg, eg_preds = self.conv1.forward(sample_input)
            pooledEg = self.pool.forward(conv1_output_eg)
            self.conv2 = ILLConv2D(64, 128, 3, pooledEg, device=device)
            conv2_output_eg, eg_preds = self.conv2.forward(pooledEg)
            pooledEg = self.pool.forward(conv2_output_eg)
            self.fc1 = ILLLinearLayer(pooledEg.flatten(start_dim=1).shape[1], num_classes, device=device) # was 16*5*5
            self.flat = nn.Flatten()
            self.layers = [self.conv1, 
                       self.pool,
                       self.conv2,
                       self.pool,
                       self.flat,
                       self.fc1]
            self.trainableList = [self.conv1, 
                              self.conv2, 
                              self.fc1]    
            print(self.layers)
            print(self.trainableList) 
                  
        elif args.net == 'vgg11':
            self.conv1 = ILLConv2D(3, 64, 3, sample_input, device=device,num_classes=num_classes)
            conv1_output_eg, eg_preds = self.conv1.forward(sample_input)
            pooledEg = self.pool.forward(conv1_output_eg)
            self.conv2 = ILLConv2D(64, 128, 3, pooledEg, device=device,num_classes=num_classes)
            conv2_output_eg, eg_preds = self.conv2.forward(pooledEg)
            pooledEg = self.pool.forward(conv2_output_eg)
            self.conv3 = ILLConv2D(128, 256, 3, pooledEg, device=device,num_classes=num_classes)
            conv3_output_eg, eg_preds = self.conv3.forward(pooledEg)
            self.conv4 = ILLConv2D(256, 256, 3, conv3_output_eg, device=device,num_classes=num_classes)
            conv4_output_eg, eg_preds = self.conv4.forward(conv3_output_eg)
            pooledEg = self.pool.forward(conv4_output_eg)
            self.conv5 = ILLConv2D(256, 512, 3, pooledEg, device=device,num_classes=num_classes)
            conv5_output_eg, eg_preds = self.conv5.forward(pooledEg)
            self.conv6 = ILLConv2D(512, 512, 3, conv5_output_eg, device=device,num_classes=num_classes)
            conv6_output_eg, eg_preds = self.conv6.forward(conv5_output_eg)
            pooledEg = self.pool.forward(conv6_output_eg)
            self.conv7 = ILLConv2D(512, 512, 3, pooledEg, device=device,num_classes=num_classes)
            conv7_output_eg, eg_preds = self.conv7.forward(pooledEg)
            self.conv8 = ILLConv2D(512, 512, 3, conv7_output_eg, device=device,num_classes=num_classes)
            conv8_output_eg, eg_preds = self.conv7.forward(conv7_output_eg)
            pooledEg = self.pool.forward(conv8_output_eg)
            self.fc1 = ILLLinearLayer(pooledEg.flatten(start_dim=1).shape[1], 4096, device=device,num_classes=num_classes) # was 16*5*5
            self.fc2 = ILLLinearLayer(4096,4096, device=device,num_classes=num_classes)
            self.fc3 = ILLLinearLayer(4096,num_classes, device=device,num_classes=num_classes)
            self.flat = nn.Flatten()
            self.layers = [self.conv1, 
                       self.pool,
                       self.conv2,
                       self.pool,
                       self.conv3,
                       self.conv4,
                       self.pool,
                       self.conv5,
                       self.conv6,
                       self.pool,
                       self.conv7,
                       self.conv8,
                       self.pool,
                       self.flat, 
                       self.fc1, 
                       self.fc2, 
                       self.fc3]
            self.trainableList = [self.conv1, 
                              self.conv2, 
                              self.conv3,
                              self.conv4,
                              self.conv5,
                              self.conv6,
                              self.conv7,
                              self.conv8,
                              self.fc1, 
                              self.fc2, 
                              self.fc3]
            self.skip_targets = None
            
        elif args.net == 'vgg16': 
            self.conv1 = ILLConv2D(3, 64, 3, sample_input, device=device,num_classes=num_classes)
            conv1_output_eg, eg_preds = self.conv1.forward(sample_input)
            self.conv2 = ILLConv2D(64, 64, 3, conv1_output_eg, device=device,num_classes=num_classes)
            conv2_output_eg, eg_preds = self.conv2.forward(conv1_output_eg)
            pooledEg = self.pool.forward(conv2_output_eg)
            self.conv3 = ILLConv2D(64, 128, 3, pooledEg, device=device,num_classes=num_classes)
            conv3_output_eg, eg_preds = self.conv3.forward(pooledEg)
            self.conv4 = ILLConv2D(128, 128, 3, conv3_output_eg, device=device,num_classes=num_classes)
            conv4_output_eg, eg_preds = self.conv4.forward(conv3_output_eg)
            pooledEg = self.pool.forward(conv4_output_eg)
            self.conv5 = ILLConv2D(128, 256, 3, pooledEg, device=device,num_classes=num_classes)
            conv5_output_eg, eg_preds = self.conv5.forward(pooledEg)
            self.conv6 = ILLConv2D(256, 256, 3, conv5_output_eg, device=device,num_classes=num_classes)
            conv6_output_eg, eg_preds = self.conv6.forward(conv5_output_eg)
            self.conv7 = ILLConv2D(256, 256, 3, conv6_output_eg, device=device,num_classes=num_classes)
            conv7_output_eg, eg_preds = self.conv7.forward(conv6_output_eg)
            pooledEg = self.pool.forward(conv7_output_eg)
            self.conv8 = ILLConv2D(256, 512, 3, pooledEg, device=device,num_classes=num_classes)
            conv8_output_eg, eg_preds = self.conv8.forward(pooledEg)
            self.conv9 = ILLConv2D(512, 512, 3, conv8_output_eg, device=device,num_classes=num_classes)
            conv9_output_eg, eg_preds = self.conv9.forward(conv8_output_eg)
            self.conv10 = ILLConv2D(512, 512, 3, conv9_output_eg, device=device,num_classes=num_classes)
            conv10_output_eg, eg_preds = self.conv10.forward(conv9_output_eg)
            pooledEg = self.pool.forward(conv10_output_eg)
            self.conv11 = ILLConv2D(512, 512, 3, pooledEg, device=device,num_classes=num_classes)
            conv11_output_eg, eg_preds = self.conv11.forward(pooledEg)
            self.conv12 = ILLConv2D(512, 512, 3, conv11_output_eg, device=device,num_classes=num_classes)
            conv12_output_eg, eg_preds = self.conv12.forward(conv11_output_eg)
            self.conv13 = ILLConv2D(512, 512, 3, conv12_output_eg, device=device,num_classes=num_classes)
            conv13_output_eg, eg_preds = self.conv13.forward(conv12_output_eg)
            pooledEg = self.pool.forward(conv13_output_eg)
            self.batchnorm64 = nn.BatchNorm2d(64,device=device)
            self.batchnorm128 = nn.BatchNorm2d(128,device=device)
            self.batchnorm256 = nn.BatchNorm2d(256,device=device)
            self.batchnorm512 = nn.BatchNorm2d(512, device=device)
            self.dropout = nn.Dropout(0.5)
            self.fc1 = ILLLinearLayer(pooledEg.flatten(start_dim=1).shape[1], 4096, device=device,num_classes=num_classes) # was 16*5*5
            self.fc2 = ILLLinearLayer(4096,4096, device=device,num_classes=num_classes)
            self.fc3 = ILLLinearLayer(4096,num_classes, device=device,num_classes=num_classes)
            self.flat = nn.Flatten()
            self.layers = [self.conv1, 
                       self.batchnorm64,
                       self.conv2,
                       self.batchnorm64,
                       self.pool,
                       self.conv3,
                       self.batchnorm128,
                       self.conv4,
                       self.batchnorm128,
                       self.pool,
                       self.conv5,
                       self.batchnorm256,
                       self.conv6,
                       self.batchnorm256,
                       self.conv7,
                       self.batchnorm256,
                       self.pool,
                       self.conv8,
                       self.batchnorm512,
                       self.conv9,
                       self.batchnorm512,
                       self.conv10,
                       self.batchnorm512,
                       self.pool,
                       self.conv11,
                       self.batchnorm512,
                       self.conv12,
                       self.batchnorm512,
                       self.conv13,
                       self.batchnorm512,
                       self.pool,
                       self.flat, 
                       self.fc1, 
                       self.dropout,
                       self.fc2, 
                       self.dropout,
                       self.fc3]
            self.trainableList = [self.conv1, 
                              self.conv2, 
                              self.conv3,
                              self.conv4,
                              self.conv5,
                              self.conv6,
                              self.conv7,
                              self.conv8,
                              self.conv9,
                              self.conv10,
                              self.conv11,
                              self.conv12,
                              self.conv13,
                              self.fc1, 
                              self.fc2, 
                              self.fc3]

    
    def train(self, dataloader, writer, filename):
        device = args.gpu
        for layer in range(len(self.layers)):
            if self.layers[layer] not in self.trainableList:
                print("Skipping untrainable layer ", layer + 1)
            else:
                print("Training Layer", layer + 1, self.layers[layer])
                previousLayers = self.layers[:layer]
                self.layers[layer].trainLayer(dataloader, previousLayers, device, writer, filename)
                current_trained_layers = self.layers[:layer+1]
                #print(current_trained_layers)
                #acc, acc_final = self.eval_training(training_loader, current_trained_layers, True)
                #acc_test, acc_test_final = self.eval_training(test_loader, current_trained_layers, False)
                #with open(filename, "a") as f:
                #        f.write("Training accuracy is {:.3f} after training layer {} \n".format(acc*100.0, layer))
                #        f.write("Training accuracy is {:.3f} after training layer {} while considering only the final predictor layer\n".format(acc_final*100.0, layer))
                #        f.write("Testing accuracy is {:.3f} after training layer {} \n".format(acc_test*100.0, layer))
                #        f.write("Testing accuracy is {:.3f} after training layer {} while considering only the final predictor layer\n".format(acc_test_final*100.0, layer))
                if (self.layers[layer] in self.trainableList) and (layer != len(self.layers)):
                    self.layers[layer].removePredictor()


    # Predict on a batch
    def predict(self, x, current_trained_layers):
        # get per layer logits
        layerPreds = []
        currentlayerPreds = []
        layerOutputs = []
        layerInput = x
        for layer in current_trained_layers:
            if layer not in self.trainableList:
                layerInput = layer.forward(layerInput)
            else:
                layerOutput, layerPred = layer.forward(layerInput)
                # Get per layer softmax
                currentlayerPreds.append(layerPred)
                layerOutputs.append(layerOutput)
                layerPreds.append(F.softmax(layerPred, dim=1))
                layerInput = layerOutput
        finallayerpred = layerPreds[-1]
        layerPreds = torch.stack(layerPreds)
         # Add up per layer softmax
        combinedPred = torch.sum(layerPreds, dim=0)
        finalPred = F.softmax(combinedPred, dim=1)
        return layerOutput, finalPred, finallayerpred
    
    @torch.no_grad()        
    def eval_training(self, test_loader, current_trained_layers, check_train=True):

        start = time.time()
        if current_trained_layers == []: 
          current_trained_layers = self.layers
        test_loss = 0.0 # cost function error
        correct = 0.0
        correct_finallayer = 0.0
        criterion = nn.CrossEntropyLoss()
        for (images, labels) in test_loader:

            if args.gpu == 'cuda:0':
                images = images.cuda()
                labels = labels.cuda().squeeze()

            outputs, preds, finallayerpreds = self.predict(images, current_trained_layers)
            loss = criterion(preds, labels)
            test_loss += loss.item()
            #_, preds = self.predict(images)
            _, predicted = torch.max(preds, 1)
            _, predicted_final = torch.max(finallayerpreds, 1)
            correct += (predicted == labels).sum().item()
            correct_finallayer += (predicted_final == labels).sum().item()

        finish = time.time()
        if check_train:
            print('Evaluating Network on train dataset.....')
            writer.add_scalar('Train/Accuracy', correct/ len(test_loader.dataset))
            writer.add_scalar('Train/Accuracy_with_finalPred', correct_finallayer/ len(test_loader.dataset))
            print('Training set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s (combined prediction)\n'.format(
            test_loss/len(test_loader.dataset), correct / len(test_loader.dataset), finish - start))
            print('Training set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s (final layer prediction)\n'.format(
            test_loss / len(test_loader.dataset),
            correct_finallayer / len(test_loader.dataset),
            finish - start
        )) 
        else: 
            print('Evaluating Network on test dataset.....')
            writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset))
            writer.add_scalar('Test/Accuracy', correct / len(test_loader.dataset))
            print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s (combined prediction) \n'.format(test_loss / len(test_loader.dataset),correct / len(test_loader.dataset),finish - start))
            print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s (final layer prediction)     \n'.format(test_loss / len(test_loader.dataset),correct_finallayer / len(test_loader.dataset),finish - start))    
    
        return correct / len(test_loader.dataset), correct_finallayer / len(test_loader.dataset)   
    
    # Evaluate on loader
    '''def evaluate(self, loader):
        correct = 0
        total = 0
        device = args.gpu
        with torch.no_grad():
            for data in loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = self.predict(inputs)
                _, predicted = torch.max(preds, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct/total'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, required=True, help='select dataset')
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-gpu', default='cpu', help='use gpu or not')
    args = parser.parse_args()
    input_size = [3, 256, 256]
    num_classes = 10

    if args.dataset == 'FashionMNIST':
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
        input_size = [3, 224, 224]
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
    elif args.dataset == 'EuroSAT':
        input_size = [3, 224, 224]
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
    elif args.dataset == 'GTSRB':
        input_size = [3, 128, 128]
        num_classes = 43
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
    elif args.dataset == 'PCAM':
        input_size = [3, 94, 94]
        num_classes = 2
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
        input_size = [3, 224, 224]
        num_classes = 100
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
    elif args.dataset == 'Pets':
        input_size = [3, 224, 224]
        num_classes = 37
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
    elif args.dataset == 'Blood':
        input_size = [3, 224, 224]
        num_classes = 8
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
    elif args.dataset == 'CUB':
        num_classes = 200
        input_size = [3, 128, 128]
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

    elif args.dataset == 'Food101':
        num_classes = 101
        input_size = [3, 224, 224]
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

    else: 
        raise Exception('Dataset not valid')

    net = ILLNet(args, input_size=input_size, num_classes=num_classes)

    epochs = 2

    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, args.dataset, settings.TIME_NOW))

    filename = 'ILL_memtext_'+ args.dataset + '_' + args.net + '.txt'
    with open(filename, "w") as f:
        f.write("Testing\n")

    torch.cuda.reset_peak_memory_stats()
    net.train(training_loader, writer, filename)
    memory_usage = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    print("Mem stat")
    print(memory_usage)


