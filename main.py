import os
import torch.nn as nn
import argparse
import deepcore.nets as nets
import deepcore.datasets as datasets
import deepcore.methods as methods
from torchvision import transforms
from utils import *
from datetime import datetime
from time import sleep


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # Basic arguments
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ResNet18', help='model')
    parser.add_argument('--selection', type=str, default="uniform", help="selection method")
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--gpu', default=None, nargs="+", type=int, help='GPU id to use')
    parser.add_argument('--print_freq', '-p', default=20, type=int, help='print frequency (default: 20)')
    parser.add_argument('--fraction', default=0.1, type=float, help='fraction of data to be selected (default: 0.1)')
    parser.add_argument('--seed', default=int(time.time() * 1000) % 100000, type=int, help="random seed")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    
    # Optimizer and scheduler
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for updating network parameters')
         

    # Selecting
    parser.add_argument("--selection_epochs", "-se", default=40, type=int,
                        help="number of epochs whiling performing selection on full dataset")
     
    # Algorithm
    parser.add_argument('--submodular', default="GraphCut", help="specifiy submodular function to use")
    parser.add_argument('--submodular_greedy', default="LazyGreedy", help="specifiy greedy algorithm for submodular optimization")
    parser.add_argument('--uncertainty', default="Entropy", help="specifiy uncertanty score to use")
    

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Build model
    model = args.model
    models = [model]
            
    network = nets.__dict__[model](channel, num_classes, im_size)
    if args.device == "cpu":
        print("Using CPU.")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu[0])
        network = nets.nets_utils.MyDataParallel(network, device_ids=args.gpu)
    elif torch.cuda.device_count() > 1:
        network = nets.nets_utils.MyDataParallel(network).cuda()
    else:
        # Una sola GPU visible o modo CPU
        network = network.to(args.device)
        
    print(f"Model device: {next(network.parameters()).device}")  
    criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
    
    # Optimizer       
    optimizer = torch.optim.Adam(network.parameters(), args.lr)      
       
       
    for epoch in range(1, args.epochs):
        # train for one epoch
        train(train_loader, network, criterion, optimizer, epoch, args, rec)
        # evaluate on validation set        
   

if __name__ == '__main__':
    main()
