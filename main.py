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
from logger import Logger, save_result
from eval import eval_acc_gpu, eval_rocauc, eval_f1 , evaluate_batch_preload, evaluate_batch_gds
from parse import parse_method, load_args
import time
from plot_curve import plot_curve
from torch.profiler import profile, ProfilerActivity
from contextlib import contextmanager
from data_utils import load_data, load_meta_data, ping_pang_loader
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
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
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
    if args.mode == 'gpu' or args.mode == 'uvm' or args.mode == 'cpu':
        train_data = load_data(in_dim, args, 'train')
    if args.load_all:
        valid_data = load_data(in_dim, args, 'val').to(device)
        test_data = load_data(in_dim, args, 'test').to(device)
        gc.collect()
    elif args.eval_load_host:
        valid_data = load_data(in_dim, args, 'val')
        test_data = load_data(in_dim, args, 'test')
    

    load_A_stream = torch.cuda.Stream()
    load_B_stream = torch.cuda.Stream()
    compute_A_stream = torch.cuda.Stream()
    compute_B_stream = torch.cuda.Stream()

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

        train_idx_run = split_idx['train']
        valid_idx_run = split_idx['valid']
        test_idx_run = split_idx['test']

        ### PIN Train data in CPU ###
        # train data should be padded so that it can be divided by batch size
        if args.mode == 'uvm' or args.mode == 'gpu' or args.mode=='cpu': # do not shuffle data since read directly from storage
            train_data_len = train_idx_run.size(0)
            tail_len = args.batch_size - train_data_len % args.batch_size
            if tail_len == args.batch_size:
                tail_len = 0
            train_idx_run = torch.cat([train_idx_run, train_idx_run[:tail_len]])
            # permute the train_idx_run at the beginning
            train_idx_local=torch.cat((torch.arange(0, train_data_len), torch.arange(0, tail_len)))
            train_idx_perm=torch.randperm(train_idx_local.size(0))
            train_idx_local=train_idx_local[train_idx_perm]
            train_idx_run = train_idx_run[train_idx_perm]       
        
        if args.mode == 'uvm' or args.mode == 'cpu':
            train_data_pinned = train_data[train_idx_local].contiguous()
            train_labels = labels[train_idx_run].contiguous().to(device)
            if args.pin_memory:
                train_data_pinned = train_data_pinned.pin_memory()
            # train_data_pinned.copy_(train_data_reorder)
        elif args.mode == 'gpu':
            train_data_pinned = train_data[train_idx_local].to(device)
            train_labels = labels[train_idx_run].to(device)
        elif args.mode == 'gds':
            train_labels = labels[train_idx_run].to(device) 
            train_data_pinned = None
        else:
            raise ValueError('Invalid mode')
        
        # assume batch_size is multiple of chunk_size
        num_chunks = len(train_idx_run) // args.chunk_size
        num_batch = len(train_idx_run) // args.batch_size
        chunks_per_batch = args.batch_size // args.chunk_size
        
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
                #permutate the chunks
                if args.mode == 'gpu':
                    train_chunk_idx = torch.randperm(num_chunks, device=device)
                elif args.mode == 'uvm' or args.mode == 'cpu':
                    train_chunk_idx = torch.randperm(num_chunks)
                elif args.mode == 'gds':
                    train_chunk_idx = torch.randperm(num_batch) # in our gds mode always use batch as the unit
                else:
                    raise ValueError('Invalid mode')
                model.train()
                if args.cpu:
                    start = time.time()
                else:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)  
                    start.record()
                collect_time = 0
                total_items=0
                
                with torch.cuda.stream(load_A_stream):
                    buffer_A, labels_A = ping_pang_loader(load_A_stream, compute_A_stream, train_chunk_idx, chunks_per_batch, -1, train_data_pinned, train_labels, device, args, in_dim, 4, data_path)
                for batch in range(num_batch):             
                    if batch % 2 == 0:
                        if batch != num_batch-1:    
                            with torch.cuda.stream(load_B_stream):  
                                buffer_B, labels_B = ping_pang_loader(load_B_stream, compute_B_stream, train_chunk_idx, chunks_per_batch, batch, train_data_pinned, train_labels, device, args, in_dim, 4, data_path)
                        # work on buffer_A
                        with torch.cuda.stream(compute_A_stream):
                            compute_A_stream.wait_stream(load_A_stream)
                            compute_A_stream.wait_stream(compute_B_stream)
                            optimizer.zero_grad()                                 
                            out = model(buffer_A)
                            if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins', 'questions'):
                                if labels_A.shape[1] == 1:
                                    true_label = F.one_hot(labels_A, num_classes).squeeze(1)
                                else:
                                    true_label = labels_A
                                loss = criterion(out, true_label.squeeze(1).to(torch.float))
                            else:
                                out = F.log_softmax(out, dim=1)
                                loss = criterion(out, labels_A.squeeze(1))                            
                            loss.backward()
                            optimizer.step()
                        total_items += buffer_A.size(0)
                    else:
                        if batch != num_batch-1:    
                            with torch.cuda.stream(load_A_stream):    
                                buffer_A, labels_A = ping_pang_loader(load_A_stream, compute_A_stream, train_chunk_idx, chunks_per_batch, batch, train_data_pinned, train_labels, device, args, in_dim, 4, data_path)                                                        
                        # work on buffer_B
                        with torch.cuda.stream(compute_B_stream):
                            compute_B_stream.wait_stream(load_B_stream)
                            compute_B_stream.wait_stream(compute_A_stream)                           
                            optimizer.zero_grad()                            
                            out = model(buffer_B)                        
                            if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins', 'questions'):
                                if labels_B.shape[1] == 1:
                                    true_label = F.one_hot(labels_B, num_classes).squeeze(1)
                                else:
                                    true_label = labels_B
                                loss = criterion(out, true_label.squeeze(1).to(torch.float))
                            else:
                                out = F.log_softmax(out, dim=1)
                                loss = criterion(out, labels_B.squeeze(1))
                            # compute_B_stream.wait_stream(compute_A_stream)
                            loss.backward()
                            optimizer.step()
                        total_items += buffer_B.size(0)    
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
                    if args.load_all or args.eval_load_host:
                        result = evaluate_batch_preload(model, split_idx, labels, train_data, valid_data, test_data, eval_func, criterion, device, num_classes,args)
                    else:
                        result = evaluate_batch_gds(model, split_idx, data_path, labels, in_dim, 4, eval_func, criterion, device, num_classes,args)
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
            save_profile_to_json(sys_json_path, num_nodes, in_dim, len(train_idx_run))
        # release pinned CPU memory for the current run
        train_data_pinned = None
        torch.cuda.empty_cache()
        gc.collect()
                    
        end_to_end_time = time.time() - wall_start
        print(f'Average epoch time: {total_time / (args.epochs -5)/1000 :.4f} sec')
        print(f'Average memory: {total_memory / (args.epochs -5)/1.07e9 :.4f} GB')
        if args.save_json:
            use_rnn = args.num_layers > 0
            json_file_path = args.json_file_path + f'{args.dataset}-{args.method}-SingleProcess-mode_{args.mode}-chunksize_{args.chunk_size}-time-memory.json'
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
