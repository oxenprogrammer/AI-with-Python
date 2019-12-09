import numpy as np
import pandas as pd
import torch
import argparse

from torch import nn, optim
from torchvision import transforms
from torchvision import models

import json
from time import time
from PIL import Image


def main():
    start_time = time()
    in_arg = input_args()
    
    # Load checkpoint
    model = torch.load('check_point.pth')
    model.eval()
    
    # Predict class
    pbs, classes = predict(in_arg.image_path, model, in_arg.k)
    f_names = flower_names(classes, model.class_to_idx, in_arg.class_name)
    
    # Display prediction
    data = pd.DataFrame({ 'Flower': f_names, 'Probability': pbs })
    data = data.sort_values('Probability', ascending=True)
    print('The item identified in the image file is:')
    print(data)
    
    # Computes overall runtime in seconds
    end_time = time()
    tot_time = end_time - start_time
    
    # Prints overall runtime in format hh:mm:ss
    print("\nTotal Elapsed Runtime:", str( int( (tot_time / 3600) ) ) + ":" + 
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
    
    parser.add_argument('--gpu', type=bool, default=True, help='Train classifier on GPU?')
    parser.add_argument('--image_path', type=str, default='flowers/test/28/image_05230.jpg', help='image path of the image that the model is predicting')
    parser.add_argument('--k', type=int, default='5', help='top K predicted classes of image')
    parser.add_argument('--class_name', type=str, default='cat_to_name.json', help='File containing mapping from category label to category name')
    parser.add_argument('--arch', type=str, default='vgg16', help='CNN model for image classification; choose either "vgg16" or "alexnet" only')

    return parser.parse_args()

def load_arch(arch):
    '''
    Load a pretrained CNN network for image classification; only "vgg16" and "alexnet" can be used
    '''
    
    if arch=='vgg16':
        model = models.vgg16(pretrained=True)
    elif arch=='alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError('Please choose either "vgg16" or "alexnet"')
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    scale = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    
    return scale(img)

def predict(image_path, model,k):
    
    img_processed = process_image(image_path)
    
    if torch.cuda.is_available():
        img_input = img_processed.unsqueeze(0).float().cuda()
    else:
        img_input = img_processed.unsqueeze(0).float()

    # Run image through model to compute probabilities
    prob = torch.exp(model.forward(img_input))

    # Get top-K probabilities and labels
    top_probs, top_labels = prob.topk(k)

    top_probs = top_probs.detach().cpu().numpy().tolist()[0]
    top_labels = top_labels.detach().cpu().numpy().tolist()[0]

    return top_probs, top_labels

def flower_names(classes, class_to_idx, class_name):
    names = {}
    
    with open(class_name, 'r') as f:
        cat_to_name = json.load(f)
        
    for k in class_to_idx:
        names[class_to_idx[k]] = cat_to_name[k]
        
    return [names[c] for c in classes]

# Call to main function to run the program
if __name__ == '__main__':
    main()