import numpy as np
import torchvision
import torch

from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from collections import OrderedDict
from torch.utils.data import DataLoader as DL
from torch.autograd import Variable

import argparse
from time import time

def main():
    start_time = time()
    in_arg = input_args()
    
    train_datasets, train_loader, validation_loader = load_data(in_arg.data_dir)
    
    model, input_size = load_arch(in_arg.arch)
    criterion, optimizer = build_classifier(in_arg.hidden_units, in_arg.learning_rate, model, input_size)
    
    validation(model, criterion, validation_loader)
    training(model, in_arg.epochs, in_arg.learning_rate, criterion, optimizer, train_loader, validation_loader )
    check_prediction_accuracy(validation_loader, model)
    
    model.class_to_idx = train_datasets.class_to_idx
    torch.save(model, 'check_point.pth')
    
    end_time = time()
    tot_time = end_time - start_time
    
    # print runtime in format hh:mm:ss
    print("\noverall elapsed runtime:", str( int( (tot_time / 3600) ) ) + ":" + 
          str( int(  ( (tot_time % 3600) / 60 )  ) ) + ":" + 
          str( int( ( (tot_time % 3600) % 60 ) ) ) )

def input_args():
    """
     Retrieves and parses the command line arguments created and defined using the argparse module. This function returns these arguments as an ArgumentParser object.
     Parameters:
        None - simply using argparse module to create & store command line arguments
     Returns:
        parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser(description='Image Classifier')
    
    parser.add_argument('--data_dir', type=str, default='flowers', help='Path to image directory with 3 subdirectories, "train", "valid", and "test"')
    parser.add_argument('--arch', type=str, default='vgg16', help='CNN model for image classification; choose either "vgg16" or "alexnet" only')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units')
    parser.add_argument('--learning_rate', type=float, default= 0.00001, help='Learning rate for the CNN model')
    parser.add_argument('--epochs', type=int, default=12, help='Number of epochs to run')
    parser.add_argument('--gpu', type=bool, default=True, help='Train classifier on GPU?')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")


    return parser.parse_args()

def load_data(data_dir):
    '''
    Load data set with torchvision's ImageFolder
    Parameters:
        data_dir: path to the image folder. Required subdirectories are "train", "valid", and "test"
    Returns:
        parse_args() - data structure that stores the CLA object
    '''
    
    # Path
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        'training': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'testing': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        }
    
    #Image datasets
    image_datasets = {
        'training': datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'validation': datasets.ImageFolder(valid_dir, transform=data_transforms['validation']),
        'testing': datasets.ImageFolder(test_dir, transform=data_transforms['testing']),
    }

    #Dataloaders
    dataloaders = {
        'training': DL(image_datasets['training'], batch_size=32, shuffle=True),
        'validation': DL(image_datasets['validation'], batch_size=16, shuffle=False),
        'testing': DL(image_datasets['testing'], batch_size=16, shuffle=False)
    }
    
    return image_datasets['training'], dataloaders['training'], dataloaders['validation']

def load_arch(arch):
    '''
    Use only vgg16 or alexnet pretrained networks
    '''
    
    if arch=='vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch=='alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216
    else:
        raise ValueError('Architecture Not Found: try vgg16 or alexnet')
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model, input_size

def build_classifier(hidden_units, learning_rate, model, input_size):
    gpu = False
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 6272)),
        ('relu1', nn.ReLU()),
        ('drop', nn.Dropout(p=0.75)),
        ('fc2', nn.Linear(6272, 3136)),
        ('relu2', nn.ReLU()),
        ('drop', nn.Dropout(p=0.75)),
        ('fc3', nn.Linear(3136, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    if torch.cuda.is_available():
        gpu = True
        model.cuda()
    else:
        model.cpu()

    model.classifier = classifier    

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)        

    return criterion, optimizer
    
def validation(model, criterion, d_loader):
    testing_loss = 0
    accuracy = 0
    gpu = False
    
    if torch.cuda.is_available():
        gpu = True
        model.cuda()
    else:
        model.cpu()
        
    with torch.no_grad():
        for images, labels in iter(d_loader):
            if gpu:
                images = Variable(images.float().cuda())
                labels = Variable(labels.long().cuda())

            output = model.forward(images.float())
            testing_loss += criterion(output, labels).item()
            ps = torch.exp(output).data
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()

        return testing_loss/len(d_loader), accuracy/len(d_loader)

def training(model, epochs, lr, criterion, optimizer, t_loader, v_loader):
    model.train()
    print_every_steps = 40
    steps = 0
    gpu = False
    
    if torch.cuda.is_available():
        gpu = True
        model.cuda()
    else:
        model.cpu()
    
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in iter(t_loader):
            steps += 1
            
            if gpu:
                images, labels = Variable(images.float().cuda()), Variable(labels.long().cuda())
                
            optimizer.zero_grad()
            output = model.forward(images.float())
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every_steps == 0:
                model.eval()
                valid_loss, accuracy = validation(model, criterion, v_loader)
                
                print("Epochs {}/{}".format(epoch+1, epochs),
                     "training loss {:.4f}".format(running_loss/print_every_steps),
                     "validing loss {:.4f}".format(valid_loss),
                     "accuracy {:.3f}".format(accuracy)
                     )

# Get prediction accuracy
def check_prediction_accuracy(validation_loader, model):
    correct = 0
    total = 0
    gpu = False
    
    if torch.cuda.is_available():
        gpu = True
        model.cuda()
    else:
        model.cpu()
    
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            images, labels = Variable(images.float().cuda()), Variable(labels.long().cuda())

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Prediction accuracy: %d %%' % (100 * correct / total))
    
if __name__ == '__main__':
    main()