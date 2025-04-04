import argparse
import sys
import os, random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import dgl
import dgl.nn as dglnn
import torchmetrics.functional as MF
import tqdm
from collections import OrderedDict
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
    LaborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from logger import Logger, save_result
from contextlib import contextmanager
from eval import eval_acc, eval_rocauc, eval_f1
import time
from plot_curve import plot_curve
from pathlib import Path
from torch.profiler import profile, ProfilerActivity
import gc
from ogb.nodeproppred import DglNodePropPredDataset
import warnings
from igb.dataloader import IGB260MDGLDataset
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.algorithms.join import Join
warnings.filterwarnings('ignore')

class NoOpContextManager:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def get_trace_handler(filename, filepath='../data/'):
    def trace_handler(prof):
        output = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=100)
        output_file = filepath + filename + '.json'
        prof.export_chrome_trace(output_file)
    
    return trace_handler


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=100)
    # print(output)
    output_file = '../data/profiling_dgl-mp.json'
    p.export_chrome_trace(output_file)

@contextmanager
def conditional_profile(enable_profiling, filename):
    if enable_profiling:
        # Assuming `profile` is the profiling context manager you're using,
        # like `torch.profiler.profile`
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],on_trace_ready=get_trace_handler(filename)) as p:
            yield p
    else:
        # Use the no-op context manager when profiling is not enabled
        with NoOpContextManager() as p:
            yield p

# Function to print detailed memory usage
def print_detailed_memory_usage(snapshot, description):
    print(f"Memory usage after {description}:")
    top_stats = snapshot.statistics('traceback')
    for stat in top_stats[:10]:
        frame = stat.traceback[0]
        print(f"File {frame.filename}, line {frame.lineno}")
        print(f"   Size: {stat.size / (1024 ** 2):.2f} MB")
        print(f"   Count: {stat.count}")
        print("   Traceback (most recent call last):")
        for line in stat.traceback.format():
            print(line.strip())
        print("")

# Function to save detailed memory usage to a file
def save_detailed_memory_usage(snapshot, description, filename):
    with open(filename, 'a') as f:
        f.write(f"Memory usage after {description}:\n")
        top_stats = snapshot.statistics('traceback')
        for stat in top_stats[:10]:
            frame = stat.traceback[0]
            f.write(f"File {frame.filename}, line {frame.lineno}\n")
            f.write(f"   Size: {stat.size / (1024 ** 2):.2f} MB\n")
            f.write(f"   Count: {stat.count}\n")
            f.write("   Traceback (most recent call last):\n")
            for line in stat.traceback.format():
                f.write(f"{line.strip()}\n")
            f.write("\n")

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(dglnn.SAGEConv(in_size, out_size, "mean", activation=F.relu))
        else:
            self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
            for i in range(num_layers - 2):               
                self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
            self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.hid_size = hid_size
        self.out_size = out_size
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # print("Number of destination nodes in the block:", block.number_of_dst_nodes())
            h = layer(block, h)
            # print(f'Layer {l} output shape: {h.shape}')
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        # if device == 'cpu':
        #     start = time.time()
        # else:
        #     start = torch.cuda.Event(enable_timing=True)
        #     end = torch.cuda.Event(enable_timing=True)  
        #     start.record()
        feat = g.ndata["feat"]
        # print(f'g is on device: {g.device}')
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        use_uva = g.device.type == 'cpu' and device.type != 'cpu'
        # print(f'Use UVA: {use_uva}, device:{device}, g.device:{g.device}')
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            use_uva=use_uva
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            # feat = feat.to(device)
            for input_nodes, output_nodes, blocks in dataloader:
                # print(f'feat is on device: {feat.device}')
                # print(f'blocks is on device: {blocks[0].device}')
                input_nodes = input_nodes.to(feat.device)
                x = feat[input_nodes].to(device)
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
                # clean up
                del x
                del h
                torch.cuda.empty_cache()
                # print(f'GPU memory allocated: {torch.cuda.max_memory_allocated(device=device)/1e6:.0f} MB'}')
            feat = y
            # print(f'GPU memory allocated: {torch.cuda.max_memory_allocated(device=device)/1e6:.0f} MB after layer {l} calculated')
        return y

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_heads, num_layers, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        # input layer
        # print(f'Number of heads: {num_heads}, Number of layers: {num_layers}')
        if num_layers == 1:
            self.layers.append(dglnn.GATConv(in_size, out_size, num_heads, residual=True, activation=F.elu, allow_zero_in_degree=True))
        else:
            self.layers.append(dglnn.GATConv(in_size, hid_size, num_heads, residual=True, allow_zero_in_degree=True))
            # hidden layers
            for l in range(num_layers - 2):
                self.layers.append(dglnn.GATConv(hid_size * num_heads, hid_size, num_heads, residual=True, allow_zero_in_degree=True))
            # output layer
            self.layers.append(dglnn.GATConv(hid_size * num_heads, out_size, num_heads, residual=True, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hid_size = hid_size
        self.out_size = out_size

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # print("Number of destination nodes in the block:", block.number_of_dst_nodes())
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = h.flatten(1)
                h = F.elu(h)
                h = self.dropout(h)
            else:
                h = h.mean(1)
            # print(f'Layer {l} output shape: {h.shape}')
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        use_uva = g.device.type == 'cpu' and device.type != 'cpu'
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            use_uva=use_uva
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size * self.num_heads if l != len(self.layers) - 1 else self.out_size,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            
            # feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                input_nodes = input_nodes.to(feat.device)
                x = feat[input_nodes].to(device)
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = h.flatten(1)
                    h = F.elu(h)
                    h = self.dropout(h)
                else:
                    h = h.mean(1)
                # by design, our output nodes are contiguous
                
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
                del x
                del h
                torch.cuda.empty_cache()
            feat = y
        return y
    
    


def evaluate(model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


@torch.no_grad()
def evaluate_batch(model, graph, split_idx, eval_func, criterion, device, args):
    
    model.eval()
    if device == 'cpu':
        start = time.time()
    else:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)  
        start.record()
    out = model.inference(graph, device, batch_size=args.eval_batch_size)
    if device == 'cpu':
        inference_time = time.time() - start
    else:
        end.record()
        torch.cuda.synchronize()
        inference_time = start.elapsed_time(end)
    print(f'Inference time: {inference_time:.4f}')

    labels = graph.ndata['label']
    if args.dataset in ('igb'):
        labels = labels.unsqueeze(1)
    train_acc = eval_func(labels[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(labels[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(labels[split_idx['test']], out[split_idx['test']])
    # print(f' out is on device: {out.device}')
    # print(f' labels is on device: {labels.device}')
    if args.mode == 'puregpu':
        out = out.to(device)
        labels = labels.to(device)
    
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins', 'questions'):
        if labels.shape[1] == 1:
            true_label = F.one_hot(labels[split_idx['valid']], labels.max() + 1).squeeze(1)
        else:
            true_label = labels[split_idx['valid']]
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1).to(torch.float))
    else:
        out_val = F.log_softmax(out[split_idx['valid']], dim=1)
        valid_loss = criterion(out_val, labels[split_idx['valid']].squeeze(1))
    return train_acc, valid_acc, test_acc, valid_loss, out, inference_time

@torch.no_grad()
def evaluate_subgraph(model, graph, split_idx, eval_func, criterion, device, out_size, args):
    
    model.eval()
    if device == 'cpu':
        start = time.time()
    else:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)  
        start.record()

    labeled_nodes = torch.cat([split_idx['train'], split_idx['valid'], split_idx['test']], dim=0)
    sampler = MultiLayerFullNeighborSampler(args.num_layers, prefetch_node_feats=["feat"])
    use_uva = (graph.device.type == 'cpu') and (device.type != 'cpu')
    print(f'Use UVA: {use_uva}, device:{device}, graph.device:{graph.device}')

    dataloader = DataLoader(
        graph,
        labeled_nodes,              # Seeds = labeled nodes
        sampler,
        device=device,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva
    )

    model.eval()
    feat = graph.ndata["feat"]

    # We'll store the model outputs (logits) here:
    out = torch.empty(
        graph.num_nodes(),
        out_size,             # or (model.hid_size) if you'd like intermediate
        dtype=feat.dtype,
        device='cpu',               # or device if you prefer
        pin_memory=(device != 'cpu')
    )

    for input_nodes, output_nodes, blocks in dataloader:
        # Grab the input features for these input nodes
        input_nodes = input_nodes.to(feat.device)
        for layer, block in enumerate(blocks):
            print(f"Layer {layer} block:")
            print("  Number of source nodes:", block.number_of_src_nodes())
            print("  Number of destination nodes:", block.number_of_dst_nodes())
            print("  Total number of nodes:", block.num_nodes())
            print("  Number of edges:", block.num_edges())
        x = feat[input_nodes].to(device)
        # Forward pass through all layers in one shot
        h = model(blocks, x)  # model(...) runs all L layers at once
        # Store the output embedding/logits for the output_nodes
        output_nodes = output_nodes.to('cpu')
        out[output_nodes] = h.cpu()  # or keep on GPU if you prefer


    if device == 'cpu':
        inference_time = time.time() - start
    else:
        end.record()
        torch.cuda.synchronize()
        inference_time = start.elapsed_time(end)
    print(f'Inference time: {inference_time:.4f}')

    labels = graph.ndata['label']
    if args.dataset in ('igb'):
        labels = labels.unsqueeze(1)
    train_acc = eval_func(labels[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(labels[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(labels[split_idx['test']], out[split_idx['test']])
    # print(f' out is on device: {out.device}')
    # print(f' labels is on device: {labels.device}')
    if args.mode == 'puregpu':
        out = out.to(device)
        labels = labels.to(device)
    
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins', 'questions'):
        if labels.shape[1] == 1:
            true_label = F.one_hot(labels[split_idx['valid']], labels.max() + 1).squeeze(1)
        else:
            true_label = labels[split_idx['valid']]
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1).to(torch.float))
    else:
        out_val = F.log_softmax(out[split_idx['valid']], dim=1)
        valid_loss = criterion(out_val, labels[split_idx['valid']].squeeze(1))
    return train_acc, valid_acc, test_acc, valid_loss, out, inference_time

@torch.no_grad()
def evaluate_sample(model, graph, split_idx, eval_func, criterion, device, out_size, args):
    
    model.eval()
    if device == 'cpu':
        start = time.time()
    else:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)  
        start.record()

    if args.sampler == 'neighbor':
        sampler = NeighborSampler(
            args.sample_sizes_eval,  # fanout for [layer-0, layer-1, layer-2]
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
        )
    elif args.sampler == 'LABOR':
        sampler = LaborSampler(
            args.sample_sizes_eval,  # fanout for [layer-0, layer-1, layer-2]
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
        )

    labeled_nodes = torch.cat([split_idx['train'], split_idx['valid'], split_idx['test']], dim=0)
    use_uva = (graph.device.type == 'cpu') and (device.type != 'cpu')
    print(f'Use UVA: {use_uva}, device:{device}, graph.device:{graph.device}')

    dataloader = DataLoader(
        graph,
        labeled_nodes,              # Seeds = labeled nodes
        sampler,
        device=device,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva
    )

    model.eval()
    feat = graph.ndata["feat"]

    # We'll store the model outputs (logits) here:
    out = torch.empty(
        graph.num_nodes(),
        out_size,             # or (model.hid_size) if you'd like intermediate
        dtype=feat.dtype,
        device='cpu',               # or device if you prefer
        pin_memory=(device != 'cpu')
    )

    for input_nodes, output_nodes, blocks in dataloader:
        # Grab the input features for these input nodes
        input_nodes = input_nodes.to(feat.device)
        x = feat[input_nodes].to(device)
        # Forward pass through all layers in one shot
        h = model(blocks, x)  # model(...) runs all L layers at once
        # Store the output embedding/logits for the output_nodes
        output_nodes = output_nodes.to('cpu')
        out[output_nodes] = h.cpu()  # or keep on GPU if you prefer


    if device == 'cpu':
        inference_time = time.time() - start
    else:
        end.record()
        torch.cuda.synchronize()
        inference_time = start.elapsed_time(end)
    print(f'Inference time: {inference_time:.4f}')

    labels = graph.ndata['label']
    if args.dataset in ('igb'):
        labels = labels.unsqueeze(1)
    train_acc = eval_func(labels[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(labels[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(labels[split_idx['test']], out[split_idx['test']])
    # print(f' out is on device: {out.device}')
    # print(f' labels is on device: {labels.device}')
    if args.mode == 'puregpu':
        out = out.to(device)
        labels = labels.to(device)
    
    out_val = F.log_softmax(out[split_idx['valid']], dim=1)
    valid_loss = criterion(out_val, labels[split_idx['valid']].squeeze(1))
    return train_acc, valid_acc, test_acc, valid_loss, out, inference_time


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=100)
    # print(output)
    p.export_chrome_trace(f"../data/profiling_gat_preload.json")

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

def read_file(file_path,shape):
    return torch.from_numpy(np.fromfile(file_path, dtype=np.float32).reshape(shape))

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def place_back_in_original(tensor, indices, original_tensor):
    for i, idx in enumerate(indices):
        original_tensor[idx] = tensor[i]

def remove_module_prefix(state_dict):
    """Remove the 'module.' prefix from the keys in the state dictionary."""
    return OrderedDict((key.replace('module.', ''), value) for key, value in state_dict.items())

def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def main(rank, world_size, g,split_idx, d,c, args):
    ddp_setup(rank, world_size)
    if rank == 0:
        print(args)

    settings_dict = {
        'method': args.method,
        'dataset': args.dataset,
        'sample_sizes': args.sample_sizes,
        'hidden_channels': args.hidden_channels,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'num_heads': args.num_heads,
        
    }

    fix_seed(args.seed)
    torch.cuda.set_device(rank)
    device = torch.device("cpu" if args.mode == "cpu" else f"cuda:{rank}")
    if args.mode == 'puregpu':
            g = g.to(device)
    criterion = nn.NLLLoss()

    ### Performance metric (Acc, AUC, F1) ###
    if args.metric == 'rocauc':
        eval_func = eval_rocauc
    elif args.metric == 'f1':
        eval_func = eval_f1
    else:
        eval_func = eval_acc

    if rank == 0:
        logger = Logger(args.runs, args)
        # Get the current time
        now = datetime.now()

    ### Training loop ###
    for run in range(args.runs):
        ### Load method ###
        if args.method == 'sage':
            model = SAGE(d, args.hidden_channels, c, args.num_layers, args.dropout).to(device)
        elif args.method == 'gat':
            model = GAT(d, args.hidden_channels, c, args.num_heads, args.num_layers, args.dropout).to(device)
        
        model_unwrapped = model
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        if rank == 0:
            experiment_name = f'{args.method}_{args.dataset}_{args.num_layers}_{args.sample_sizes}_{args.hidden_channels}_{args.sampler}'
            tensorboard_dir = os.path.join("../tensorboard_log/mpgnn", args.dataset, args.method, experiment_name)
            print(">>>>> Write TensorBoard logs to {}".format(tensorboard_dir))
            # Initialize TensorBoard writer
            writer = SummaryWriter(tensorboard_dir)
        
        if args.dataset == 'igb':
            train_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
            val_idx = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
            test_idx = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]
            split_idx = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        
        #create dataloader for training and test
        train_idx = split_idx['train']
        
        if args.mode == 'puregpu' or args.mode == 'mixed':
            train_idx = train_idx.to(device)
        val_idx = split_idx['valid'].to(device)

        if args.sampler == 'neighbor':
            sampler = NeighborSampler(
                args.sample_sizes,  # fanout for [layer-0, layer-1, layer-2]
                prefetch_node_feats=["feat"],
                prefetch_labels=["label"],
            )
        elif args.sampler == 'LABOR':
            sampler = LaborSampler(
                args.sample_sizes,  # fanout for [layer-0, layer-1, layer-2]
                prefetch_node_feats=["feat"],
                prefetch_labels=["label"],
            )
        use_uva = args.mode == "mixed"
        train_dataloader = DataLoader(
            g,
            train_idx,
            sampler,
            device=device if args.mode == "puregpu" or args.mode == "mixed" else "cpu",
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            use_uva=use_uva,
            use_ddp=True,
        )
        
        optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
        best_val = float('-inf')
        total_time = 0
        total_memory = 0
        best_val_acc = 0
        epochs_no_improve = 0
        total_inference_time = 0
        eval_cnt = 0
        
        total_epochs = args.epochs
        with conditional_profile(args.enable_profiling, 'DGL_MP') as p:
            for epoch in range(args.epochs):
                model.train()
                optimizer.zero_grad()
                torch.cuda.synchronize()
                dist.barrier()
                if rank == 0:
                    if args.mode == "cpu":
                        start = time.time()
                    else:
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)  
                        start.record()
                
                with Join([model]):                              
                    for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
                        optimizer.zero_grad()                   
                        x = blocks[0].srcdata["feat"]
                        label = blocks[-1].dstdata["label"]                                
                        if args.mode == 'cpusample':
                            blocks = [block.to(device) for block in blocks]
                            x = x.to(device)
                            label = label.to(device)
                        out = model(blocks, x)                                                
                        out = F.log_softmax(out, dim=1)                         
                        if label.dim() > 1 and label.size(1) == 1:
                            label = label.squeeze(1)
                        loss = criterion(out, label)
                        loss.backward()                        
                        optimizer.step()
                dist.barrier()                                                                 
                if args.enable_profiling:
                    p.step()
                if rank == 0:
                    if args.mode == "cpu":
                        train_time = time.time() - start
                    else:
                        end.record()
                        torch.cuda.synchronize()
                        train_time = start.elapsed_time(end)
                    if epoch > -1:
                        total_time += train_time
                        if args.mode == 'puregpu' or args.mode == 'mixed':
                            total_memory += torch.cuda.max_memory_allocated(device=device)
                            print(f'Epoch: {epoch}, Epoch time: {train_time:.4f}, Train Mem:{torch.cuda.max_memory_allocated(device=device)/1e6:.0f} MB')
                        else:
                            print(f'Epoch: {epoch}, Epoch time: {train_time:.4f}')
                    if epoch > args.test_start_epoch and epoch % args.eval_step == 0:
                        adjusted_state_dict = remove_module_prefix(model.state_dict())
                        model_unwrapped.load_state_dict(adjusted_state_dict)
                        eval_device = torch.device('cpu') if args.dataset in ('igb') else device
                        model_unwrapped = model_unwrapped.to(eval_device)
                        print("eval device is", eval_device)
                        if args.dataset in ('ogbn-papers100M'):
                            result = evaluate_sample(model_unwrapped, g, split_idx, eval_func, criterion, eval_device, c, args)
                        else:
                            result = evaluate_batch(model_unwrapped, g, split_idx, eval_func, criterion, eval_device, args)
                        
                        # print(f'GPU memory allocated: {torch.cuda.max_memory_allocated(device=device)/1e6:.0f} MB after evaluation')
                        total_inference_time += result[-1]
                        result = result[:-1]
                        eval_cnt += 1
                        logger.add_result(run, result[:-1])

                        if result[1] > best_val:
                            best_val = result[1]
                            if args.save_model:
                                torch.save(model.state_dict(), args.model_dir + f'{args.dataset}-{args.method}.pkl')

                        writer.add_scalar("Loss/train", loss, epoch)
                        writer.add_scalar("Acc/train", result[0], epoch)
                        writer.add_scalar("Acc/validation", result[1], epoch)
                        writer.add_scalar("Acc/test", result[2], epoch)
                        
                        print(f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Train: {100 * result[0]:.2f}%, '
                            f'Valid: {100 * result[1]:.2f}%, '
                            f'Test: {100 * result[2]:.2f}%')
                        if result[1] > best_val_acc:
                            best_val_acc = result[1]
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1
                        if epochs_no_improve > args.patience:
                            print(f'Early stopping at epoch {epoch}')
                            total_epochs = epoch
                            break
        if rank == 0:
            print(f'Average epoch time: {total_time / (args.epochs):.4f}')
            # print(f'Average inference time: {total_inference_time / eval_cnt:.4f}')
            if args.save_json:
                json_file_path = args.json_file_path + f'{args.dataset}-{args.method}-{args.sampler}-DGL-{args.mode}-time-memory.json'
                metrics = {
                    'epoch_time': total_time / (args.epochs)/1000,
                    'memory': total_memory / (args.epochs)/1.07e9,
                    'total_epochs': total_epochs,
                }
                update_epoch_times(json_file_path, settings_dict, metrics)
                print(f'Write to {json_file_path}')
            if args.save_infer_time:
                json_file_path = args.json_file_path + f'{args.dataset}-{args.method}-DGL-{args.mode}-infer-time.json'
                metrics = {
                    'infer_time': total_inference_time / eval_cnt,
                }
                update_epoch_times(json_file_path, settings_dict, metrics)
                print(f'Write to {json_file_path}')
            # logger.print_statistics(run)
            # Close the TensorBoard writer
            writer.close()
            if args.plot_curve:
                plot_curve(tensorboard_dir)
                print(">>>>> Plot training curve to {}".format(tensorboard_dir))
            print(">>>>> Write TensorBoard logs to {}".format(tensorboard_dir))
    if rank == 0 and args.epochs>args.test_start_epoch:
        results = logger.print_statistics()
        print("results: \n", results)

        if args.save_result:
            save_result(args, results, command_string)
    destroy_process_group()

if __name__ == '__main__':
    mp.set_start_method("fork")
    command_string = ' '.join(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu", "cpusample"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="ogbn-products",
        help="Dataset name",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/",
        help="Dataset directory",
    )
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=512,
        help="Hidden layer size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Batch size",
    )
    parser.add_argument(
        "--eval_step",
        type=int,
        default=1,
        help="Evaluate every n epochs",
    )
    parser.add_argument(
        "--test_start_epoch",
        type=int,
        default=1,
        help="Start testing after n epochs",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs",
    )
    
    parser.add_argument('--sample_sizes', type=int, nargs='+', default=[15, 10, 5])
    parser.add_argument('--sample_sizes_eval', type=int, nargs='+', default=[20, 20, 20])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--json_file_path', type=str, default='../data/')
    parser.add_argument('--full_batch', action='store_true')
    parser.add_argument('--save_json', action='store_true')
    parser.add_argument('--method', '-m', type=str, default='nodeformer')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--metric', type=str, default='acc')
    parser.add_argument('--discrete', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_result', action='store_true')
    parser.add_argument('--plot_curve', action='store_true')
    parser.add_argument('--model_dir', type=str, default='../model/')
    parser.add_argument('--protocol', type=str, default='semi')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--eval_batch_size', type=int, default=8000)
    parser.add_argument('--rb_order', type=int, default=0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save_infer_time', action='store_true')
    parser.add_argument('--mix_bf16', action='store_true')
    parser.add_argument('--enable_profiling', action='store_true')
    parser.add_argument('--training_hops', type=int, default=1)
    parser.add_argument('--sampler', type=str, default='neighbor')
    parser.add_argument('--sub_dataset', type=str, default='random')
    #for loading IGB dataset
    parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M/', 
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='tiny',
        choices=['tiny', 'small', 'medium', 'large', 'full'], 
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19, 
        choices=[19, 2983], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=1, 
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    parser.add_argument('--col_split', type=int, default=1, 
        help='number of columns to split the features into')
    

    # for purpose of correct recording, set args.training_hops to the same as num_layers
    args = parser.parse_args()
    args.training_hops = args.num_layers
    print(args)
    world_size = torch.cuda.device_count()
    print(f"world size: {world_size}")

    ### Load preprocessed data ###
    if args.dataset == 'ogbn-papers100M':
        graph_save_path=os.path.join(args.data_dir, 'ogbn-papers100M_undirection.bin')
        split_save_path=os.path.join(args.data_dir, 'ogbn-papers100M_split.pt')
        if os.path.exists(graph_save_path):
            graphs, datadict = dgl.load_graphs(graph_save_path)
            g = graphs[0]
            labels = datadict['labels']
            split_idx = torch.load(split_save_path)
            
        else:
            dataset = DglNodePropPredDataset(name=args.dataset, root=f'{args.data_dir}/ogb')
            g, labels = dataset[0]
            # Convert the graph to an undirected (bidirected) graph
            g_sym = dgl.to_bidirected(g)
            g_sym.ndata.update(g.ndata)
            g_sym.edata.update(g.edata)
            g=g_sym
            del g_sym
            g.create_formats_()
            # Obtain the split indices
            split_idx = dataset.get_idx_split()
            dgl.save_graphs(graph_save_path, [g], {'labels': labels}) 
            torch.save(split_idx, split_save_path)
        
        # split_idx = dataset.get_idx_split()
        # split_idx_lst = [split_idx]
        c = 172
        d = g.ndata['feat'].shape[1]
        n = g.num_nodes()
        g.ndata['label'] = labels.long()
        # print(f'labels length: {len(labels)}')
    elif args.dataset == 'igb':
        dataset = IGB260MDGLDataset(args)
        g = dataset[0]
        c = args.num_classes
        d = 1024
        n = g.num_nodes()
        split_idx=None
    else:
        data_path = args.data_dir + 'preprocessed/' + str(args.dataset) + '/'
        data_dict = torch.load(data_path + f'{args.dataset}_aux.pt')

        split_idx = data_dict['splits']
        train_idx = split_idx['train']
        val_idx = split_idx['valid']
        test_idx = split_idx['test']
        in_dim = data_dict['in_dim']
        num_nodes = data_dict['num_nodes']
        
        
        feat_tmp=[]
        for col_idx in range(args.col_split):
            hop0_train_path = f"{data_path}{args.dataset}train_hop0_{col_idx}.bin"
            hop0_val_path = f"{data_path}{args.dataset}val_hop0_{col_idx}.bin"
            hop0_test_path = f"{data_path}{args.dataset}test_hop0_{col_idx}.bin"
            train_data = read_file(hop0_train_path, (split_idx['train'].shape[0], in_dim//args.col_split))
            val_data = read_file(hop0_val_path, (split_idx['valid'].shape[0], in_dim//args.col_split))
            test_data = read_file(hop0_test_path, (split_idx['test'].shape[0], in_dim//args.col_split))
            feat_tmp.append(torch.cat([train_data, val_data, test_data], dim=0))
        feat_tmp = torch.cat(feat_tmp, dim=1)
        feat=torch.empty(num_nodes, feat_tmp.shape[1])
        feat[train_idx] = feat_tmp[train_idx]
        feat[val_idx] = feat_tmp[val_idx]
        feat[test_idx] = feat_tmp[test_idx]
        del feat_tmp
                 
        edge_index = data_dict['edge_index']
        # adj_ori = data_dict['adj_ori']
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
        g.ndata['feat'] = feat.contiguous()
        g.ndata['label'] = data_dict['labels'].long().contiguous()
        g = dgl.add_self_loop(g)
        
        labels = data_dict['labels'].long()
        c = data_dict['labels'].max().item() + 1
        if args.dataset == 'ogbn-proteins':
            c = 112
        elif args.dataset == 'ogbn-papers100M':
            c = 172
        d = train_data.shape[1]
        
        n = train_data.shape[0]
        del train_data
        del edge_index
        del data_dict
        torch.cuda.empty_cache()
        gc.collect()

    print(f"dataset {args.dataset} | num node feats {d} | num classes {c}")
    mp.spawn(main, args=(world_size, g, split_idx, d, c, args), nprocs=world_size, join=True)

    
