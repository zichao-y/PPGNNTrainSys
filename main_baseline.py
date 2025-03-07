import sys
import os, random
import json
from pathlib import Path
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from logger import Logger, save_result
from eval import eval_acc_gpu, eval_rocauc, eval_f1 , evaluate_batch_preload, evaluate_batch_gds
from parse import parse_method, load_args
import time
from plot_curve import plot_curve
from torch.profiler import profile, ProfilerActivity
from contextlib import contextmanager
from data_utils import load_data, load_meta_data
import warnings
warnings.filterwarnings('ignore')
import resource

class NoOpContextManager:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=100)
    # print(output)
    p.export_chrome_trace(f"./data/profiling.json")

@contextmanager
def conditional_profile(enable_profiling):
    if enable_profiling:
        # Assuming `profile` is the profiling context manager you're using,
        # like `torch.profiler.profile`
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],on_trace_ready=trace_handler,record_shapes=True,profile_memory=True,with_stack=True) as p:
            yield p
    else:
        # Use the no-op context manager when profiling is not enabled
        with NoOpContextManager() as p:
            yield p

def update_epoch_times(file_path, settings_dict, metric):
    # Convert settings_dict to a string key to use in JSON
    settings_key = json.dumps(settings_dict, sort_keys=True)
    
    # Check if the file exists
    if Path(file_path).exists():
        # Read the existing data
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
    else:
        data = {}
    
    # Initialize the list for the settings_key if it doesn't exist
    if settings_key not in data:
        data[settings_key] = []
    
    # Append the new epoch time
    data[settings_key].append(metric)
    
    # Write the updated data back to the file
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def save_profile_to_json(json_path="system_par.json", number_nodes=0, in_dim=0, num_train_nodes=0):
    # Measure peak memory usage (in GB, for example)
    peak_host_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    peak_gpu_mem = torch.cuda.max_memory_allocated(device=device) 
    # Prepare the new entry
    profile_data = {
        "peak_host_mem": peak_host_mem,
        "peak_gpu_mem": peak_gpu_mem,
        "number_nodes": number_nodes,
        "in_dim": in_dim,
        "num_train_nodes": num_train_nodes
    }
    # Write directly as dictionary (not list)
    with open(json_path, 'w') as f:
        json.dump(profile_data, f, indent=4)
    print(f'Write profiling results to {json_path}')
    
# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    wall_start = time.time()
    command_string = ' '.join(sys.argv)

    ### Parse args ###
    args = load_args()
    print(args)

    settings_dict = {
        'method': args.method,
        'dataset': args.dataset,
        'input_type': args.input_type,
        'num_layers': args.num_layers,
        'training_hops': args.training_hops,
        'hidden_channels': args.hidden_channels,
        'batch_size': args.batch_size,
    }
    fix_seed(args.seed)

    if args.mode=='cpu':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### Load preprocessed data ###
    labels, split_idx, num_classes, in_dim, dim_i, num_nodes, data_path = load_meta_data(args)
    
    train_data = load_data(in_dim, args, 'train')
    valid_data = load_data(in_dim, args, 'val')
    test_data = load_data(in_dim, args, 'test')

    ### Loss function (Single-class, Multi-class) ###
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins', 'questions'):
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.NLLLoss()

    ### Performance metric (Acc, AUC, F1) ###
    if args.metric == 'rocauc':
        eval_func = eval_rocauc
    elif args.metric == 'f1':
        eval_func = eval_f1
    else:
        eval_func = eval_acc_gpu

    logger = Logger(args.runs, args)
    # Get the current time
    now = datetime.now()
    # load the model
    model = parse_method(args, num_classes, dim_i, device)
    ### Training loop ###
    for run in range(args.runs):
        experiment_name = f'{args.method}_{args.dataset}_{args.training_hops}hop_chunksize{args.chunk_size}'
        tensorboard_dir = os.path.join("./tensorboard_log/", args.dataset, args.method, experiment_name)
        print(">>>>> Write TensorBoard logs to {}".format(tensorboard_dir))
        # Initialize TensorBoard writer
        writer = SummaryWriter(tensorboard_dir)

        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']

        train_dataset = TensorDataset(train_data, labels[train_idx])
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
        best_val = float('-inf')
        total_time = 0
        total_memory = 0
        best_val_acc = 0
        epochs_no_improve = 0
        loss = 0
        # with profiler.profile(use_cuda=True) as prof:
        with conditional_profile(args.enable_profiling) as p:
            for epoch in range(args.epochs):
                model.train()
                if args.cpu:
                    start = time.time()
                else:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)  
                    start.record()
                collect_time = 0
                
                for data, label in train_dataloader:
                    data = data.squeeze(0)
                    label = label.squeeze(0)
                    data = data.to(device)
                    label = label.to(device)
                    optimizer.zero_grad()
                    out = model(data)
                    out = F.log_softmax(out, dim=1)
                    loss = criterion(out, label.squeeze(1))
                    loss.backward()
                    optimizer.step()
                  
                if args.enable_profiling:
                    p.step()                      
                if args.cpu:
                    train_time = time.time() - start
                else:
                    end.record()
                    torch.cuda.synchronize()
                    train_time = start.elapsed_time(end)
                if epoch > 4:
                    total_time += train_time
                    total_memory += torch.cuda.max_memory_allocated(device=device) if not args.mode=='cpu' else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if args.mode == 'cpu':
                    print(f'Epoch: {epoch}, Epoch time: {train_time:.4f} sec')
                else:
                    print(f'Epoch: {epoch}, Epoch time: {train_time/1000:.4f} sec, Train Mem:{torch.cuda.max_memory_allocated(device=device)/1.07e9:.0f} GB')
                
                if epoch > args.test_start_epoch and (epoch+1) % args.eval_step == 0:                                        
                    result = evaluate_batch_preload(model, split_idx, labels, train_data, valid_data, test_data, eval_func, criterion, device, num_classes,args)
                    logger.add_result(run, result[:-1])
                    if result[1] > best_val:
                        best_val = result[1]
                        if args.save_model:
                            torch.save(model.state_dict(), args.model_dir + f'{args.dataset}-{args.method}-{args.training_hops}.pkl')

                    writer.add_scalar("Loss/train", loss, epoch)
                    writer.add_scalar("Acc/train", result[0], epoch)
                    writer.add_scalar("Acc/validation", result[1], epoch)
                    writer.add_scalar("Acc/test", result[2], epoch)
                    print(f'Epoch: {epoch:02d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * result[0]:.2f}%, '
                        f'Valid: {100 * result[1]:.2f}%, '
                        f'Test: {100 * result[2]:.2f}%')
                    
                    if best_val > best_val_acc:
                        best_val_acc = best_val
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    if epochs_no_improve == args.patience:
                        print("Early stopping!")
                        break
        if args.trail_profile:
            #Peak Host Mem: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1.07e6:.0f} GB'
            sys_json_path = args.sys_json
            save_profile_to_json(sys_json_path, num_nodes, in_dim, len(train_idx))
        # release pinned CPU memory for the current run
        train_data_pinned = None
        torch.cuda.empty_cache()
        gc.collect()
                    
        end_to_end_time = time.time() - wall_start
        print(f'Average epoch time: {total_time / (args.epochs -5)/1000 :.4f} sec')
        print(f'Average memory: {total_memory / (args.epochs -5)/1.07e9 :.4f} GB')
        if args.save_json:
            json_file_path = args.json_file_path + f'{args.dataset}-{args.method}-Baseline-time-memory.json'
            metrics = {
                'epoch_time': total_time / (args.epochs -5)/1000,
                'memory': total_memory / (args.epochs -5)/1.07e9,
                'end_to_end_time': end_to_end_time
            }
            update_epoch_times(json_file_path, settings_dict, metrics)
            print(f'Write to {json_file_path}')
        if args.epochs-1 > args.test_start_epoch:
            logger.print_statistics(run)
        # Close the TensorBoard writer
        writer.close()
        if args.plot_curve:
            plot_curve(tensorboard_dir)
            print(">>>>> Plot training curve to {}".format(tensorboard_dir))
        print(">>>>> Write TensorBoard logs to {}".format(tensorboard_dir))
    if args.epochs-1 > args.test_start_epoch:
        results = logger.print_statistics()
        print("results: \n", results)

        if args.save_result:
            save_result(args, results, command_string)
