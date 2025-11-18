import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import TensorDataset
import deepcore.nets as nets
import deepcore.datasets as datasets
import deepcore.methods as methods
from utils import *


def load_custom_npz(data_path, dataset_name='custom'):
    """Load custom .npz dataset"""
    train_path = os.path.join(data_path, f"{dataset_name}-train.npz")
    val_path = os.path.join(data_path, f"{dataset_name}-val.npz")
    test_path = os.path.join(data_path, f"{dataset_name}-test.npz")
    
    # Load train
    train_data = np.load(train_path)
    X_train = torch.from_numpy(train_data['images']).float()
    y_train = torch.from_numpy(train_data['labels']).long()
    
    # Load val
    val_data = np.load(val_path)
    X_val = torch.from_numpy(val_data['images']).float()
    y_val = torch.from_numpy(val_data['labels']).long()
    
    # Load test
    test_data = np.load(test_path)
    X_test = torch.from_numpy(test_data['images']).float()
    y_test = torch.from_numpy(test_data['labels']).long()
    
    # Infer dataset properties
    if len(X_train.shape) == 3:  # (N, H, W) -> add channel
        X_train = X_train.unsqueeze(1)
        X_val = X_val.unsqueeze(1)
        X_test = X_test.unsqueeze(1)
    
    channel = X_train.shape[1]
    im_size = (X_train.shape[2], X_train.shape[3])
    num_classes = len(torch.unique(y_train))
    class_names = [str(i) for i in range(num_classes)]
    
    # Compute mean and std
    mean = [X_train.mean().item()]
    std = [X_train.std().item()]
    
    # Create datasets
    dst_train = TensorDataset(X_train, y_train)
    dst_test = TensorDataset(X_test, y_test)
    
    print(f"Loaded custom dataset: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test


def main():
    parser = argparse.ArgumentParser(description='DeepCore Coreset Selection')

    # Basic arguments
    parser.add_argument('--dataset', type=str, default='MNIST', 
                        help='dataset: CIFAR10, CIFAR100, MNIST, ImageNet, or custom')
    parser.add_argument('--input_data', type=str, default=None, 
                        help='path to custom .npz data (overrides --dataset)')
    parser.add_argument('--model', type=str, default='ResNet18', help='model architecture')
    parser.add_argument('--selection', type=str, default='Uniform', 
                        help='selection method: Uniform, Random, Herding, KCenterGreedy, '
                             'ContextualDiversity, Forgetting, GraNd, Cal, DeepFool, '
                             'Craig, GradMatch, Glister, Submodular, Uncertainty, EarlyTrain, Full')
    parser.add_argument('--num_exp', type=int, default=10, 
                        help='number of experiments with different seeds')
    parser.add_argument('--fraction', default=0.1, type=float, 
                        help='fraction of data to select (default: 0.1)')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='results', 
                        help='path to save selection results')
    parser.add_argument('--seed', default=42, type=int, help="base random seed")
    
    # GPU and workers
    parser.add_argument('--gpu', default=None, nargs="+", type=int, help='GPU id to use')
    parser.add_argument('-j', '--workers', default=4, type=int, 
                        help='number of data loading workers')
    parser.add_argument('--print_freq', '-p', default=20, type=int, 
                        help='print frequency during selection')
    
    # Selection parameters (for methods that need training)
    parser.add_argument('--selection_epochs', '-se', default=40, type=int,
                        help='epochs for training during selection (for proxy-based methods)')
    parser.add_argument('--selection_lr', '-slr', type=float, default=0.1, 
                        help='learning rate for selection')
    parser.add_argument('--selection_momentum', '-sm', default=0.9, type=float,
                        help='momentum for selection')
    parser.add_argument('--selection_weight_decay', '-swd', default=5e-4, type=float,
                        help='weight decay for selection')
    parser.add_argument('--selection_optimizer', '-so', default='SGD',
                        help='optimizer for selection: SGD, Adam')
    parser.add_argument('--selection_nesterov', '-sn', default=True, type=str_to_bool,
                        help='use nesterov momentum during selection')
    parser.add_argument('--selection_batch', '-sb', default=128, type=int,
                        help='batch size during selection')
    
    # Algorithm-specific parameters
    parser.add_argument('--balance', default=False, type=str_to_bool,
                        help='balance selection per class (for Submodular methods)')
    parser.add_argument('--submodular', default='GraphCut', 
                        help='submodular function: GraphCut, FacilityLocation, LogDet')
    parser.add_argument('--submodular_greedy', default='LazyGreedy',
                        help='greedy algorithm: LazyGreedy, NaiveGreedy, StochasticGreedy')
    parser.add_argument('--uncertainty', default='Entropy',
                        help='uncertainty measure: Entropy, Margin, LeastConfident')

    args = parser.parse_args()
    
    # Set device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu[0])
        args.device = f'cuda:{args.gpu[0]}'
    
    print(f"\n{'='*80}")
    print(f"DeepCore Coreset Selection")
    print(f"{'='*80}")
    print(f"Selection Method: {args.selection}")
    print(f"Dataset: {args.input_data if args.input_data else args.dataset}")
    print(f"Model: {args.model}")
    print(f"Fraction: {args.fraction}")
    print(f"Num Experiments: {args.num_exp}")
    print(f"Device: {args.device}")
    print(f"{'='*80}\n")
    
    # Load dataset
    if args.input_data is not None:
        # Load custom .npz
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = \
            load_custom_npz(args.input_data, dataset_name='mnist')  # adjust name as needed
        args.dataset = 'custom'
    else:
        # Load torchvision dataset
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = \
            datasets.__dict__[args.dataset](args.data_path)
    
    args.channel = channel
    args.im_size = im_size
    args.num_classes = num_classes
    args.class_names = class_names
    
    print(f"Dataset loaded: {len(dst_train)} train samples, {len(dst_test)} test samples")
    print(f"Image size: {im_size}, Channels: {channel}, Classes: {num_classes}\n")
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Run multiple experiments with different seeds
    for exp in range(args.num_exp):
        exp_seed = args.seed + exp
        torch.manual_seed(exp_seed)
        np.random.seed(exp_seed)
        
        print(f"\n{'='*80}")
        print(f"Experiment {exp+1}/{args.num_exp} (seed={exp_seed})")
        print(f"{'='*80}\n")
        
        # Create experiment directory
        exp_name = f"{args.selection}_{args.dataset}_frac{args.fraction}_seed{exp_seed}"
        exp_dir = os.path.join(args.save_path, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Prepare selection arguments
        selection_args = dict(
            epochs=args.selection_epochs,
            selection_method=args.uncertainty,
            balance=args.balance if args.selection == 'Submodular' else False,
            greedy=args.submodular_greedy,
            function=args.submodular
        )
        
        # Instantiate selection method
        print(f"Initializing selection method: {args.selection}...")
        start_time = time.time()
        
        try:
            method = methods.__dict__[args.selection](
                dst_train, args, args.fraction, exp_seed, **selection_args
            )
            
            # Perform selection
            print(f"Performing selection...")
            subset = method.select()
            
            selection_time = time.time() - start_time
            
            # Extract indices and labels
            indices = subset['indices']
            selected_labels = [dst_train[i][1] for i in indices]
            
            print(f"\nSelection completed in {selection_time:.2f}s")
            print(f"Selected {len(indices)} samples ({len(indices)/len(dst_train)*100:.2f}%)")
            print(f"Class distribution: {np.bincount(selected_labels)}")
            
            # Save subset indices
            indices_path = os.path.join(exp_dir, 'subset_indices.npz')
            np.savez(indices_path, 
                     indices=np.array(indices),
                     labels=np.array(selected_labels))
            print(f"Saved indices to: {indices_path}")
            
            # Save metadata
            metadata = {
                'method': args.selection,
                'dataset': args.dataset,
                'model': args.model,
                'fraction': args.fraction,
                'seed': exp_seed,
                'num_selected': len(indices),
                'total_samples': len(dst_train),
                'selection_time_seconds': selection_time,
                'class_distribution': np.bincount(selected_labels).tolist(),
                'selection_args': selection_args,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            metadata_path = os.path.join(exp_dir, 'selection_info.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"Saved metadata to: {metadata_path}")
            
        except Exception as e:
            print(f"Error during selection: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Results saved to: {args.save_path}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()