import time
import json
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from PIL import Image
from collections import OrderedDict
import torch
from torch import nn, optim
#from classifier import create_classifier,pretrained,train_model,get_optimizer
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from argparser import get_input_args
#loading the different data_directory
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#using the pipelines to transform the data
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
}
dirs = {'train': train_dir, 
        'valid': valid_dir, 
        'test' : test_dir}
image_datasets = {x: datasets.ImageFolder(dirs[x],   transform=data_transforms[x]) for x in ['train', 'valid', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
class_names = image_datasets['train'].classes

in_args=get_input_args()
#criterion=nn.NLLLoss()
if in_args.arch == 'VGG':
        model = models.vgg16(pretrained=True)
        # Only train the classifier parameters, feature parameters are frozen
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 4096)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(4096, in_args.hidden_units)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.5)),
            ('fc3', nn.Linear(in_args.hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier

elif in_args.arch == 'Densenet':
    model = models.densenet121(pretrained=True)
        # Only train the classifier parameters, feature parameters are frozen
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(1024,in_args.hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout()),
            ('fc2', nn.Linear(in_args.hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    model.classifier = classifier

else:
    raise Exception("Architecture not accepted")

print("Training model...")

#lets train the model:

criterion=nn.NLLLoss()
optimizer=optim.Adam(model.classifier.parameters(),in_args.lr)



if in_args.gpu and torch.cuda.is_available():
    model.cuda()

    steps = 0
    running_loss = 0
    print_every=10

    for e in range(in_args.epochs):
        # Model in training mode, dropout is on
        model.train()
        for inputs, labels in iter(dataloaders['train']):
            steps += 1

            optimizer.zero_grad()

            if in_args.gpu and torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            # Model in inference mode, dropout is off
            if steps % print_every == 0:
                model.eval()

                accuracy = 0
                test_loss = 0
                for ii, (inputs, labels) in enumerate(dataloaders['valid']):
                    if in_args.gpu and torch.cuda.is_available():
                        inputs, labels = inputs.cuda(), labels.cuda()
                    output = model.forward(inputs)
                    test_loss += criterion(output, labels).data[0]

                    ## Calculating the accuracy
                    # Model's output is log-softmax, take exponential to get the probabilities
                    ps = torch.exp(output).data
                    # Class with highest probability is our predicted class, compare with true label
                    equality = (labels.data == ps.max(1)[1])
                    # Accuracy is number of correct predictions divided by all predictions, just take the mean
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}/{}.. ".format(e + 1, in_args.epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss / len(dataloaders['valid'])),
                      "Validation Accuracy: {:.3f}".format(accuracy / len(dataloaders['valid'])))

                running_loss = 0

                # Make sure dropout is on for training
                model.train()
    
if __name__=='main':
    main()
    
    