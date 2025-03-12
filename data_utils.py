import torch
import numpy as np
import gc
from concurrent.futures import ThreadPoolExecutor
import async_fetch
import gds_read

def expand_chunk_indices(chunk_idx, chunk_size):
    if chunk_size == 1:
        return chunk_idx
    else:
        start_idx = chunk_idx * chunk_size
        item_idx = torch.cat([torch.arange(start, start + chunk_size) for start in start_idx])
        return item_idx

def ping_pang_loader(load_stream, compute_stream,train_chunk_idx, chunks_per_batch, batch,train_data_pinned, train_labels, device, args, in_dim=1024, element_size=4, data_path=None):
    if args.mode == 'gds':
        file_path_da = data_path + str(args.dataset) + 'train_da'
        file_path_dad = data_path + str(args.dataset) + 'train_dad'
        file_path_ori = data_path + str(args.dataset) + 'train_hop0'
        item_idx_batch = train_chunk_idx[batch+1]  
        if batch >= 0:                                                   
            load_stream.wait_stream(compute_stream)
        labels = train_labels[item_idx_batch*args.batch_size:(item_idx_batch+1)*args.batch_size]
        # load train data from file system to GPU using cufile
        data_offset = item_idx_batch * args.batch_size * (in_dim // args.col_split) * element_size
        data_length = args.batch_size * (in_dim // args.col_split) * element_size
        futures = []
        with ThreadPoolExecutor() as executor:
            buffer = []
            if args.method == 'SGC':
                for split_idx in range(args.col_split):
                    if args.input_type == 'da':
                        file_path = file_path_da + f'_{args.training_hops-1}_{split_idx}.bin'
                    elif args.input_type == 'dad':
                        file_path = file_path_dad + f'_{args.training_hops-1}_{split_idx}.bin'
                    futures.append(executor.submit(gds_read.readfile_async, file_path, data_offset, data_length))
            else:
                for split_idx in range(args.col_split):
                    file_path= file_path_ori + f'_{split_idx}.bin'
                    futures.append(executor.submit(gds_read.readfile_async, file_path, data_offset, data_length))
                
                for i in range(0, args.training_hops):
                    for split_idx in range(args.col_split):
                        if args.input_type == 'da':
                            file_path = file_path_da + f'_{i}_{split_idx}.bin'
                        elif args.input_type == 'dad':
                            file_path = file_path_dad + f'_{i}_{split_idx}.bin'
                        futures.append(executor.submit(gds_read.readfile_async, file_path, data_offset, data_length))
        # Collect results from futures
        results = [future.result() for future in futures]
        if args.method == 'SGC':
            for i in range(args.col_split):
                reshaped_data = results[i].view(torch.float32).view(-1, in_dim//args.col_split)
                buffer.append(reshaped_data)
            buffer = torch.cat(buffer, dim=1)
        else:
            buffer_elements = []
            for i in range(args.col_split):
                reshaped_data = results[i].view(torch.float32).view(-1, in_dim//args.col_split)
                buffer_elements.append(reshaped_data)
            buffer_elements = torch.cat(buffer_elements, dim=1)
            buffer.append(buffer_elements)
            
            for i in range(0, args.training_hops):
                buffer_elements = []
                for split_idx in range(args.col_split):
                    reshaped_data = results[args.col_split * i + split_idx + args.col_split].view(torch.float32).view(-1, in_dim//args.col_split)
                    buffer_elements.append(reshaped_data)
                buffer_elements = torch.cat(buffer_elements, dim=1)
                buffer.append(buffer_elements)

            if args.cat_input:
                buffer = torch.cat(buffer, dim=1)
            else:
                buffer = torch.stack(buffer, dim=1)    
    else:    
        if batch >= 0:
            load_stream.wait_stream(compute_stream)                            
        if args.mode == 'gpu':
            if args.chunk_size == 1:
                chunk_idx_batch = train_chunk_idx[(batch+1)*chunks_per_batch:(batch+2)*chunks_per_batch]
                item_idx_batch = expand_chunk_indices(chunk_idx_batch, args.chunk_size)
                buffer = train_data_pinned[item_idx_batch]
                labels = train_labels[item_idx_batch] 
            else:
                labels = torch.cat([train_labels[train_chunk_idx[i]*args.chunk_size:(train_chunk_idx[i]+1)*args.chunk_size] for i in range((batch+1)*chunks_per_batch, (batch+2)*chunks_per_batch)], dim=0)
                buffer = []
                for i in range((batch+1)*chunks_per_batch, (batch+2)*chunks_per_batch):
                    buffer.append(train_data_pinned[train_chunk_idx[i]*args.chunk_size:(train_chunk_idx[i]+1)*args.chunk_size])
                buffer = torch.cat(buffer, dim=0)                                   
        else:
            if args.chunk_size == 1:
                chunk_idx_batch = train_chunk_idx[(batch+1)*chunks_per_batch:(batch+2)*chunks_per_batch]
                item_idx_batch = expand_chunk_indices(chunk_idx_batch, args.chunk_size)
                item_idx_batch = item_idx_batch.to('cpu')
                buffer = fetch_data_process(item_idx_batch, train_data_pinned, device)
                labels = train_labels[item_idx_batch]
            else:
                labels = torch.cat([train_labels[train_chunk_idx[i]*args.chunk_size:(train_chunk_idx[i]+1)*args.chunk_size] for i in range((batch+1)*chunks_per_batch, (batch+2)*chunks_per_batch)], dim=0)
                buffer = []
                for i in range((batch+1)*chunks_per_batch, (batch+2)*chunks_per_batch):
                    buffer.append(train_data_pinned[train_chunk_idx[i]*args.chunk_size:(train_chunk_idx[i]+1)*args.chunk_size].to(device, non_blocking=True))
                buffer = torch.cat(buffer, dim=0)
    
    return buffer, labels

def read_file(file_path, shape):
    return torch.from_numpy(np.fromfile(file_path, dtype=np.float32).reshape(shape))

def fetch_data_process(item_idx_batch, train_data_pinned,device):
    fetcher = async_fetch.AsyncFetcher(item_idx_batch, train_data_pinned, device)
    buffer = fetcher.get()  # This waits for the async operation to complete
    return buffer

def load_meta_data(args):
    if args.full_path:
        data_path = args.data_dir
    else:
        data_path = args.data_dir + 'preprocessed/' + str(args.dataset) + '/'
    if args.dataset == 'igb':
        data_path = data_path + '/' + args.dataset_size + '/'
    data_dict = torch.load(data_path + f'{args.dataset}_aux.pt')
    labels = data_dict['labels']
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)
    split_idx = data_dict['splits']
    #reshape labels for ogbn-papers100M since not all nodes have labels and only data with labels will be used
    if args.dataset == 'ogbn-papers100M':
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']
        index_all = torch.cat([train_idx, valid_idx, test_idx], dim=0)
        labels = labels[index_all]
        #assert no nan in labels
        assert torch.isnan(labels).sum() == 0
        labels = labels.long()
        # for i in range(args.runs):
        split_idx['train'] = torch.arange(0, len(train_idx))
        split_idx['valid'] = torch.arange(len(train_idx), len(train_idx) + len(valid_idx))
        split_idx['test'] = torch.arange(len(train_idx) + len(valid_idx), len(train_idx) + len(valid_idx) + len(test_idx))
    num_classes = data_dict['labels'].max().item() + 1
    if args.dataset == 'ogbn-proteins':
        num_classes = 112
    elif args.dataset == 'ogbn-papers100M':
        num_classes = 172
    in_dim = data_dict['in_dim']
    if args.input_type == 'da' or args.input_type == 'dad':
        dim_i = in_dim
    elif args.input_type == 'dad_da':
        dim_i = in_dim * 2
    if args.cat_input:
        dim_i = dim_i * (args.training_hops +1)
    num_nodes = data_dict['num_nodes']
    return labels, split_idx, num_classes, in_dim, dim_i, num_nodes, data_path

def load_data(in_dim, args, dtype='val'):
    if args.full_path:
        data_path = args.data_dir
    else:
        data_path = args.data_dir + 'preprocessed/' + str(args.dataset) + '/'
    if args.dataset == 'igb':
        data_path = data_path + '/' + args.dataset_size + '/'
    if args.method == 'SGC':
        with ThreadPoolExecutor() as executor:
            futures = []
            for col_idx in range(args.col_split):
                if args.input_type == 'da':
                    file_path = f"{data_path}{args.dataset}{dtype}_da_{args.training_hops-1}_{col_idx}.bin"
                elif args.input_type == 'dad':
                    file_path = f"{data_path}{args.dataset}{dtype}_dad_{args.training_hops-1}_{col_idx}.bin"
                shape = (-1, in_dim // args.col_split)
                futures.append(executor.submit(read_file, file_path, shape))
            hop_data = [future.result() for future in futures]
        out_data = torch.cat(hop_data, dim=1)
        del hop_data
        gc.collect()
    else:
        out_data = []
        # Load hop0 data in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            for col_idx in range(args.col_split):
                hop0_path = f"{data_path}{args.dataset}{dtype}_hop0_{col_idx}.bin"
                shape = (-1, in_dim // args.col_split)
                futures.append(executor.submit(read_file, hop0_path, shape))
            hop0_data = [future.result() for future in futures]
        hop0_data = torch.cat(hop0_data, dim=1)           
        out_data.append(hop0_data)
        del hop0_data
        gc.collect()
        # Load training hops data in parallel
        for i in range(0, args.training_hops):
            with ThreadPoolExecutor() as executor:
                futures = []
                for col_idx in range(args.col_split):
                    if args.input_type == 'da':
                        file_path = f"{data_path}{args.dataset}{dtype}_da_{i}_{col_idx}.bin"
                    elif args.input_type == 'dad':
                        file_path = f"{data_path}{args.dataset}{dtype}_dad_{i}_{col_idx}.bin"
                    shape = (-1, in_dim // args.col_split)
                    futures.append(executor.submit(read_file, file_path, shape))
                hop_data = [future.result() for future in futures]
            hop_data = torch.cat(hop_data, dim=1)
            out_data.append(hop_data)
            del hop_data
            gc.collect()
        
        if args.cat_input:
            out_data = torch.cat(out_data, dim=1)
        else:
            out_data = torch.stack(out_data, dim=1)
    gc.collect()
    return out_data