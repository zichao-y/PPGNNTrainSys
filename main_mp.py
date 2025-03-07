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
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.algorithms.join import Join
from collections import OrderedDict
import resource

def remove_module_prefix(state_dict):
    """Remove the 'module.' prefix from the keys in the state dictionary."""
    return OrderedDict((key.replace('module.', ''), value) for key, value in state_dict.items())

def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def split_tensor_round_robin(data, n_parts, idx):
    # Number of elements in the data
    n_elements = data.size(0)
    # Create an index tensor that reflects the round-robin assignment
    indices = torch.arange(n_elements).long() % n_parts
    # filter indices to only include the ones that match the current rank
    split_indices = torch.where(indices == idx)
    # Use sorted_indices to reorder data in a round-robin fashion
    parts = data[split_indices]
    return parts

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

def expand_chunk_indices(chunk_idx, chunk_size):
    if chunk_size == 1:
        return chunk_idx
    else:
        start_idx = chunk_idx * chunk_size
        item_idx = torch.cat([torch.arange(start, start + chunk_size) for start in start_idx])
        return item_idx
    
# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_shared_tensor(tensor):
    shared_tensor = tensor.share_memory_()
    return shared_tensor

def main(rank, world_size, train_data_shared, valid_data_shared, test_data_shared, labels, split_idx, num_classes, in_dim, dim_i, data_path, args):
    wall_start = time.time()
    command_string = ' '.join(sys.argv)
    ddp_setup(rank, world_size)
    if rank == 0:
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

    device = torch.device(f"cuda:{rank}")
    
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
    ### Training loop ###
    for run in range(args.runs):
        # load the model
        model = parse_method(args, num_classes, dim_i, device)
        model_unwrapped = model
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        if rank == 0:
            experiment_name = f'{args.method}_{args.dataset}_{args.training_hops}hop_chunksize{args.chunk_size}'
            tensorboard_dir = os.path.join("./tensorboard_log/", args.dataset, args.method, experiment_name)
            print(">>>>> Write TensorBoard logs to {}".format(tensorboard_dir))
            # Initialize TensorBoard writer
            writer = SummaryWriter(tensorboard_dir)

        train_idx_run = split_idx['train']
        
        if args.mode == 'gpu':
            train_labels = labels[train_idx_run]
            train_data_pinned = split_tensor_round_robin(train_data_shared, world_size, rank).to(device)
            train_labels = split_tensor_round_robin(train_labels, world_size, rank).to(device)
            gc.collect()
        elif args.mode == 'uvm':
            train_labels = labels[train_idx_run]
            train_data_pinned = train_data_shared
            train_labels = train_labels.to(device)
            gc.collect()
        
        # assume batch_size is multiple of chunk_size
        num_chunks = train_data_pinned.size(0) // args.chunk_size
        chunks_per_batch = args.batch_size // args.chunk_size
        
        optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
        best_val = float('-inf')
        total_time = 0
        total_memory = 0
        best_val_acc = 0
        epochs_no_improve = 0
        loss = 0
        
        for epoch in range(args.epochs):
            with Join([model]):
                #permutate the chunks
                if args.mode == 'gpu':
                    train_chunk_idx = torch.randperm(num_chunks, device=device)
                    num_batch = train_data_pinned.size(0) // args.batch_size
                elif args.mode == 'uvm':
                    #permutate the chunks                 
                    g = torch.Generator(device=device)
                    g.manual_seed(epoch)
                    train_chunk_idx_all = torch.randperm(num_chunks, generator=g, device=device)
                    idx_filter = train_chunk_idx_all % world_size
                    train_chunk_idx = train_chunk_idx_all[idx_filter == rank]
                    num_batch = train_chunk_idx.size(0) // chunks_per_batch
                else:
                    raise NotImplementedError('Only support GPU and UVM mode for multi-threading')
                model.train()
                if args.cpu:
                    start = time.time()
                else:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)  
                    start.record()
                
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
            torch.distributed.barrier()  
            if rank == 0:                    
                if args.cpu:
                    train_time = time.time() - start
                else:
                    end.record()
                    torch.cuda.synchronize()
                    train_time = start.elapsed_time(end)
                if epoch > 4:
                    total_time += train_time
                    total_memory += torch.cuda.max_memory_allocated(device=device)
                print(f'Epoch: {epoch}, Epoch time: {train_time/1000:.4f} sec, Train Mem:{torch.cuda.max_memory_allocated(device=device)/1.07e9:.0f} GB')
                
                if epoch > args.test_start_epoch and (epoch+1) % args.eval_step == 0:  
                    adjusted_state_dict = remove_module_prefix(model.state_dict())
                    model_unwrapped.load_state_dict(adjusted_state_dict)                  
                    if args.load_all or args.eval_load_host:
                        result = evaluate_batch_preload(model_unwrapped, split_idx, labels, train_data_shared, valid_data_shared, test_data_shared, eval_func, criterion, device, num_classes, args)
                    else:
                        result = evaluate_batch_gds(model_unwrapped, split_idx, data_path, labels, in_dim, 4, eval_func, criterion, device, num_classes,args)
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
            # 1. Measure peak CPU memory usage for this rank
            peak_mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # On Linux, ru_maxrss is in KB; on macOS, it's in bytes, so adjust if needed
            peak_mem_tensor = torch.tensor([peak_mem_kb], dtype=torch.float64, device=device)
            # 2. Sum across all ranks using all_reduce
            torch.distributed.all_reduce(peak_mem_tensor, op=torch.distributed.ReduceOp.SUM)        
        # release pinned CPU memory for the current run
        train_data_pinned = None
        torch.cuda.empty_cache()
        gc.collect()

        if rank == 0:            
            end_to_end_time = time.time() - wall_start
            print(f'Average epoch time: {total_time / (args.epochs -5)/1000 :.4f} sec')
            print(f'Average memory: {total_memory / (args.epochs -5)/1.07e9 :.4f} GB')
            if args.save_json:
                use_rnn = args.num_layers > 0
                json_file_path = args.json_file_path + f'{args.dataset}-{args.method}-MultiProcess-mode_{args.mode}-chunksize_{args.chunk_size}-time-memory.json'
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
    if rank == 0:
        if args.epochs-1 > args.test_start_epoch:
            results = logger.print_statistics()
            print("results: \n", results)
            if args.save_result:
                save_result(args, results, command_string)
        if args.trail_profile:
            print(f"Peak CPU memory usage: {peak_mem_tensor.item() / 1e6:.2f} GB")

    destroy_process_group()
    return

if __name__ == '__main__':
    args = load_args()
    print(args)
    visible_devices = ",".join(str(x) for x in args.gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    world_size = len(args.gpu_ids)
    # world_size = 1
    print(world_size)
    ### Load preprocessed data ###
    labels, split_idx, num_classes, in_dim, dim_i, num_nodes, data_path = load_meta_data(args)
    
    if args.mode == 'gpu' or args.mode == 'uvm':
        train_data = load_data(in_dim, args, 'train') 
    else:
        raise NotImplementedError('Only support GPU and UVM mode for multi-threading')
    if args.load_all or args.eval_load_host:
        valid_data = load_data(in_dim, args, 'val')
        test_data = load_data(in_dim, args, 'test')
        valid_data_shared = create_shared_tensor(valid_data)
        test_data_shared = create_shared_tensor(test_data)
    else:
        valid_data_shared = None
        test_data_shared = None
    # if args.pin_memory:
    #     train_data = train_data.pin_memory()
    #     gc.collect()
    train_data_shared = create_shared_tensor(train_data)
    mp.spawn(main, args=(world_size, train_data_shared, valid_data_shared, test_data_shared, labels, split_idx, num_classes, in_dim, dim_i, data_path, args), nprocs=world_size, join=True)
