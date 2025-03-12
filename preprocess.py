import torch
import argparse
# from torch_sparse import SparseTensor
import numpy as np
from parse import parser_add_main_args
from dataset import load_dataset, load_fixed_splits
import os
from tqdm import tqdm
import time
import gc
import dgl

def create_coo(values, row, col, row_size, col_size):
        indices = torch.stack([row, col], dim=0)
        return torch.sparse_coo_tensor(indices, values, (row_size, col_size)).coalesce()

def graph2adj_coo_new(edge_index, nnodes, output_type='dual'):
    row, col = edge_index
    # Check validity using tensor operations
    if torch.isnan(row).any() or torch.isnan(col).any() or \
       torch.isinf(row).any() or torch.isinf(col).any():
        raise ValueError('row or col contains nan or inf')
    if (row < 0).any() or (col < 0).any():
        raise ValueError('row or col contains negative values')
    if (row >= nnodes).any() or (col >= nnodes).any():
        raise ValueError('row or col contains values greater than nnodes')

    degree_vec = torch.bincount(row, minlength=nnodes)
    with torch.no_grad():  # Prevent gradient tracking
        d_inv_sqrt = torch.pow(degree_vec, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt) | torch.isnan(d_inv_sqrt)] = 0
        
    if output_type == 'dual':
        dad_data = d_inv_sqrt[row] * d_inv_sqrt[col]
        da_data = d_inv_sqrt[row] * d_inv_sqrt[row]
        DAD = create_coo(dad_data, row, col, nnodes, nnodes)
        DA = create_coo(da_data, row, col, nnodes, nnodes)
        return DAD, DA
    elif output_type == 'dad':
        dad_data = d_inv_sqrt[row] * d_inv_sqrt[col]
        return create_coo(dad_data, row, col, nnodes, nnodes)
    else:
        da_data = d_inv_sqrt[row] * d_inv_sqrt[row]
        return create_coo(da_data, row, col, nnodes, nnodes)
    
def graph2adj_coo(edge_index, nnodes, output_type='dual'):
    row, col = edge_index
    # check row and col has no nan and inf
    if np.isnan(row).any() or np.isnan(col).any() or np.isinf(row).any() or np.isinf(col).any():
        raise ValueError('row or col contains nan or inf')
    # check row and col has no negative values
    if (row < 0).any() or (col < 0).any():
        raise ValueError('row or col contains negative values')
    # check row and col has no values greater than nnodes 
    if (row >= nnodes).any() or (col >= nnodes).any():
        raise ValueError('row or col contains values greater than nnodes')

    degree_vec = np.bincount(row, minlength=nnodes)
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(degree_vec, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)|np.isnan(d_inv_sqrt)] = 0

    if output_type == 'dual':
        dad_data = d_inv_sqrt[row] * d_inv_sqrt[col]
        dad_data_tensor = torch.from_numpy(dad_data).float()
        DAD = SparseTensor(row=row, col=col, value=dad_data_tensor, sparse_sizes=(nnodes, nnodes))
    
        da_data = d_inv_sqrt[row] * d_inv_sqrt[row]   
        da_data_tensor = torch.from_numpy(da_data).float()
        DA = SparseTensor(row=row, col=col, value=da_data_tensor, sparse_sizes=(nnodes, nnodes))

        return DAD, DA

    elif output_type == 'dad':
        dad_data = d_inv_sqrt[row] * d_inv_sqrt[col]
        dad_data_tensor = torch.from_numpy(dad_data).float()
        DAD = SparseTensor(row=row, col=col, value=dad_data_tensor, sparse_sizes=(nnodes, nnodes))
        return DAD
    
    else:
        da_data = d_inv_sqrt[row] * d_inv_sqrt[row]   
        da_data_tensor = torch.from_numpy(da_data).float()
        DA = SparseTensor(row=row, col=col, value=da_data_tensor, sparse_sizes=(nnodes, nnodes))
        return DA

def spmm_blocks_sptensor_new(adj, features, device, block_size=1000000, col_size=8):
    if not adj.is_sparse or adj.layout != torch.sparse_coo:
        raise ValueError("adj must be a coalesced sparse COO tensor")
    adj = adj.coalesce()
    indices = adj.indices()
    values = adj.values()

    if not isinstance(features, torch.Tensor):
        raise ValueError("features must be a torch.Tensor")
    # Ensure features has no NaNs or Infs
    if torch.isnan(features).any() or torch.isinf(features).any():
        raise ValueError("features contains NaNs or Infs")

    results = []
    pbar_outer = tqdm(total=len(range(0, features.size(1), col_size)), desc="Feature Blocks")
    for k in range(0, features.size(1), col_size):
        end_idx = min(k + col_size, features.size(1))
        feature_block = features[:, k:end_idx].to(device)
        result_block = []
        n_rows = adj.size(0)
        pbar_inner = tqdm(total=len(range(0, n_rows, block_size)), desc="Processing", leave=False)
        
        for i in range(0, n_rows, block_size):
            start_row = i
            num_rows = min(block_size, n_rows - start_row)
            end_row = start_row + num_rows

            # Efficiently find the indices in the current block using binary search
            start_idx = torch.searchsorted(indices[0], start_row)
            end_idx = torch.searchsorted(indices[0], end_row, right=True)
            block_indices = indices[:, start_idx:end_idx]
            block_values = values[start_idx:end_idx]

            if block_indices.size(1) == 0:
                result = torch.zeros(num_rows, feature_block.size(1), device=device)
            else:
                # Adjust row indices for the block
                block_indices_row = block_indices[0] - start_row
                block_indices_adj = torch.stack([block_indices_row, block_indices[1]])
                
                # Create block sparse tensor on the device
                block = torch.sparse_coo_tensor(
                    block_indices_adj.to(device),
                    block_values.to(device),
                    (num_rows, adj.size(1)),
                    device=device
                ).coalesce()

                # Perform sparse matrix multiplication
                result = torch.sparse.mm(block, feature_block)
            
            result_block.append(result.cpu())
            del block, result
            pbar_inner.update(1)
        pbar_inner.close()
        result_block = torch.cat(result_block, dim=0)
        results.append(result_block)
        pbar_outer.update(1)
    pbar_outer.close()
    return torch.cat(results, dim=1)

def spmm_blocks_sptensor(adj, features, device, block_size=1000000, col_size=8):
    # Ensure adj is a SparseTensor
    if not isinstance(adj, SparseTensor):
        raise ValueError("adj must be a torch_sparse SparseTensor")
    # Ensure features is a torch.Tensor
    if not isinstance(features, torch.Tensor):
        raise ValueError("features must be a torch.Tensor")
    # Ensure features has no NaNs or Infs
    if torch.isnan(features).any() or torch.isinf(features).any():
        raise ValueError("features contains NaNs or Infs")
    print(f'features type: {features.dtype}')
    
    results = []
    pbar_outer = tqdm(total=len(range(0, features.size(1), col_size)), desc="Feature Blocks")
    for k in range(0, features.size(1), col_size):
        end_idx = min(k + col_size, features.size(1))
        feature_block = features[:, k:end_idx].to(device)
        result_block = []
        n_rows = adj.size(0)
        pbar_inner = tqdm(total=len(range(0, n_rows, block_size)), desc="Processing", leave=False)
        for i in range(0, n_rows, block_size):
            start_row = i
            num_rows = min(block_size, n_rows - start_row)
            
            block = adj.narrow(0, start_row, num_rows)
            block = block.to(device)
            # Access indices and values
            row_indices = block.storage.row()
            col_indices = block.storage.col()
            values = block.storage.value()

            # Debugging statements
            print(f'Processing block {i // block_size + 1}/{(n_rows + block_size - 1) // block_size}')
            print(f'block shape: {block.sizes()}, device: {block.device()}')
            print(f'feature_block shape: {feature_block.shape}, device: {feature_block.device}')
            print(f'block indices dtype: row={row_indices.dtype}, col={col_indices.dtype}')
            print(f'block values dtype: {values.dtype}')
            print(f'Max row index: {row_indices.max()}, block size: {block.size(0)}')
            print(f'Max col index: {col_indices.max()}, block size: {block.size(1)}')

            try:
                result = block @ feature_block
            except RuntimeError as e:
                print(f'Error during multiplication: {e}')
                raise

            print(f'result shape: {result.shape}')
            if torch.isnan(result).any():
                print('Result contains NaNs')
            if torch.isinf(result).any():
                print('Result contains Infs')
            
            result = result.to('cpu')
            result_block.append(result)
            del result  # Optional, but won't free memory immediately
            pbar_inner.update(1)
        pbar_inner.close()
        result_block = torch.cat(result_block, dim=0)
        results.append(result_block)
        del result_block  # Optional, but won't free memory immediately
        pbar_outer.update(1)
        torch.cuda.empty_cache()  # Clears cache to free up GPU memory
    pbar_outer.close()
    results_cat = torch.cat(results, dim=1)
    del results
    gc.collect()
    return results_cat      
        
def preprocess_savebinary_sepx(graph, split, name='ogbn-papers100M', hops=10, device='cpu', block_size=0, save_path='/scratch/graph_data/', block_col_size=8, col_split=1, kernel='da'):
    print(f"preprocessing {name} dataset")
    nnodes = graph.num_nodes()  
    node_feat = graph.ndata['feat'] 
    features = node_feat.shape[-1]
    col_size = features // col_split
    assert features % col_split == 0
    if kernel == 'dad' or kernel == 'dual':
        norm_adj_dad_cpu = graph2adj_coo_new(graph.edges(), nnodes, output_type='dad')
    if kernel == 'da' or kernel == 'dual':
        norm_adj_da_cpu = graph2adj_coo_new(graph.edges(), nnodes, output_type='da')
    
    
    train_idx = split['train']
    val_idx = split['valid']
    test_idx = split['test']
        
    if block_size == 0:
        if kernel == 'da' or kernel == 'dual':
            norm_adj_da = norm_adj_da_cpu.to(device)
        if kernel == 'dad' or kernel == 'dual':
            norm_adj_dad = norm_adj_dad_cpu.to(device)
    
    for feat_idx in range(0, col_split):
        x = node_feat[:,feat_idx*col_size:(feat_idx+1)*col_size]
        if kernel == 'dad' or kernel == 'dual':
            high_order_features_dad = x.clone()
        if kernel == 'da' or kernel == 'dual':
            high_order_features_da = x.clone()
        file_path = save_path + name + f'train_hop0_{feat_idx}.bin'
        train_hop0 = x[train_idx].clone().numpy()
        train_hop0.tofile(file_path)
        del train_hop0
        gc.collect()
        file_path = save_path + name + f'val_hop0_{feat_idx}.bin'
        val_hop0 = x[val_idx].clone().numpy()
        val_hop0.tofile(file_path)
        del val_hop0
        gc.collect()
        file_path = save_path + name + f'test_hop0_{feat_idx}.bin'
        test_hop0 = x[test_idx].clone().numpy()
        test_hop0.tofile(file_path)
        del test_hop0
        gc.collect()
        del x
        gc.collect()
        
        for i in range(hops):
            if block_size > 0:
                if kernel == 'da' or kernel == 'dual':  
                    high_order_features_da = spmm_blocks_sptensor_new(norm_adj_da_cpu, high_order_features_da, device, block_size, block_col_size)
                if kernel == 'dad' or kernel == 'dual':
                    high_order_features_dad = spmm_blocks_sptensor_new(norm_adj_dad_cpu, high_order_features_dad, device, block_size, block_col_size)
            else:
                if kernel == 'da' or kernel == 'dual':
                    high_order_features_da = norm_adj_da @ high_order_features_da.to(device)
                if kernel == 'dad' or kernel == 'dual':
                    high_order_features_dad = norm_adj_dad @ high_order_features_dad.to(device)
                  
            if kernel == 'da' or kernel == 'dual':
                high_order_features_da = high_order_features_da.cpu()
                file_path = save_path + name + f'train_da_{i}_{feat_idx}.bin'
                train_da = high_order_features_da[train_idx].numpy()
                train_da.tofile(file_path)
                del train_da
                gc.collect()
                file_path = save_path + name + f'val_da_{i}_{feat_idx}.bin'
                val_da = high_order_features_da[val_idx].numpy()
                val_da.tofile(file_path)
                del val_da
                gc.collect()
                file_path = save_path + name + f'test_da_{i}_{feat_idx}.bin'
                test_da = high_order_features_da[test_idx].numpy()
                test_da.tofile(file_path)
                del test_da
                gc.collect()
            if kernel == 'dad' or kernel == 'dual':
                high_order_features_dad = high_order_features_dad.cpu()
                file_path = save_path + name + f'train_dad_{i}_{feat_idx}.bin'
                train_dad = high_order_features_dad[train_idx].numpy()
                train_dad.tofile(file_path)
                del train_dad
                gc.collect()
                file_path = save_path + name + f'val_dad_{i}_{feat_idx}.bin'
                val_dad = high_order_features_dad[val_idx].numpy()
                val_dad.tofile(file_path)
                del val_dad
                gc.collect()
                file_path = save_path + name + f'test_dad_{i}_{feat_idx}.bin'
                test_dad = high_order_features_dad[test_idx].numpy()
                test_dad.tofile(file_path)
                del test_dad
                gc.collect()
        if kernel == 'da' or kernel == 'dual':
            del high_order_features_da
        if kernel == 'dad' or kernel == 'dual':
            del high_order_features_dad
        gc.collect()
    
    return 

def gnnlab_save(edge_index, x, save_dir, name, y, split, use_pca=False, PCA_dim=64, QUANT=False, data_type='int8'):
    
    data_path = save_dir + 'gnnlab/' + name + '/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    labels = y.numpy().astype(np.uint64)
    # print(f'first 100 labels: {labels[:100]}')
   
    labels.tofile(data_path + 'label.bin')
    nnodes = x.shape[0]
    row, col = edge_index
    values = np.ones(row.shape[0])
    adj = create_coo(values, row, col, nnodes, nnodes)
    adj = adj.to_scipy(layout='csr')
    indptr = adj.indptr.astype(np.uint32)
    indptr.tofile(data_path + 'indptr.bin')
    indices = adj.indices.astype(np.uint32)
    indices.tofile(data_path + 'indices.bin')           
    features = x.numpy().astype(np.float32)
    features.tofile(data_path + 'feat.bin')
    trainingset = split['train'].numpy().astype(np.uint32)
    trainingset.tofile(data_path + 'train_set.bin')
    validationset = split['valid'].numpy().astype(np.uint32)
    validationset.tofile(data_path + 'valid_set.bin')
    testingset = split['test'].numpy().astype(np.uint32)
    testingset.tofile(data_path + 'test_set.bin')
    num_classes = y.max().item() + 1
    if name == 'ogbn-papers100M':
        num_classes = 172
    print('Writing meta file...')
    with open(f'{data_path}meta.txt', 'w') as f:
        f.write('{}\t{}\n'.format('NUM_NODE', nnodes))
        f.write('{}\t{}\n'.format('NUM_EDGE', adj.nnz))
        f.write('{}\t{}\n'.format('FEAT_DIM', x.shape[1]))
        f.write('{}\t{}\n'.format('NUM_CLASS', num_classes))
        f.write('{}\t{}\n'.format('NUM_TRAIN_SET', len(split['train'])))
        f.write('{}\t{}\n'.format('NUM_VALID_SET', len(split['valid'])))
        f.write('{}\t{}\n'.format('NUM_TEST_SET', len(split['test'])))

#main function to read in data and do preprocessing, finally save the data to .pt file
def main():
    #add arguments
    parser = argparse.ArgumentParser(description='General Training Pipeline')
    parser_add_main_args(parser)
    args = parser.parse_args()
    print("train_prop: ",args.train_prop)
    print("valid_prop: ",args.valid_prop)
    print("runs: ",args.runs)
    print("dataset: ",args.dataset)
    print("sub_dataset: ",args.sub_dataset)
    print("rand_split: ",args.rand_split)
    print("rand_split_class: ",args.rand_split_class)
    print("label_num_per_class: ",args.label_num_per_class)
    print("self_loops: ",args.self_loops)
    print("training_hops: ",args.training_hops)
    print("data_dir: ",args.data_dir)
    print("protocol: ",args.protocol)
    print("kernel_type: ",args.kernel_type)
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")    
    print(f"device: {device}")
    now = time.time()
    ### Load and preprocess data ###
    
    dataset = load_dataset(args.data_dir, args.dataset, args)
    graph = dataset.graph['dgl']
    # Apply DGL transformations based on args
    ndata = graph.ndata
    if not args.directed and args.dataset != 'ogbn-proteins':
        graph = dgl.to_bidirected(graph)
    if args.self_loops:
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
    graph.ndata.update(ndata)

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    
    if args.rand_split:
        split_idx = dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                        
    elif args.rand_split_class:
        split_idx = dataset.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                        
    elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products', 'amazon2m', 'igb']:
        split_idx = dataset.load_fixed_splits()
                        
    else:
        if args.dataset == 'fb100':
            name='fb100-'+args.sub_dataset
        else:
            name=args.dataset
        split_idx = load_fixed_splits(args.data_dir, dataset, name=name, protocol=args.protocol)
        
    
    if args.tognnlab:
        quantize = True if args.data_type in ['int8', 'e4m3', 'e3m4', 'bf16', 'fp16'] else False
        if args.dataset == 'igb':
            name = 'igb_' + args.dataset_size
        else:
            name = args.dataset
        
        gnnlab_save(graph.edges(), graph.ndata['feat'], args.save_dir, name, labels, split_idx, use_pca=args.use_pca, PCA_dim=args.pca_dim, QUANT=quantize, data_type=args.data_type)
        exit()
    
    labels = graph.ndata['label']
    # print(f'train_data length: {len(split_idx["train"])}')
    time_start = time.time()
     
    dir_path = args.save_dir + 'preprocessed/' + str(args.dataset)  + '/'
    if args.dataset == 'igb':
        dir_path = dir_path + '/' + args.dataset_size + '/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    preprocess_savebinary_sepx(graph, split_idx, args.dataset, args.training_hops,device, args.block_size,dir_path, args.col_size, args.col_split, args.kernel_type)
    process_end = time.time()
    if args.dataset == 'igb' or args.dataset == 'ogbn-papers100M':
        data_dict = {"labels": labels, "splits": split_idx, "in_dim": graph.ndata['feat'].shape[1], "num_nodes": graph.number_of_nodes()}
    else:
        data_dict = {"labels": labels, "splits": split_idx, "in_dim": graph.ndata['feat'].shape[1], "num_nodes": graph.number_of_nodes(), "edge_index": graph.edges()}
    
    torch.save(data_dict, dir_path + args.dataset + '_aux.pt')
    print(f"Preprocessing done, processing time: {process_end - time_start:.2f} sec, loading time: {time_start - now:.2f} sec, total time: {time.time() - now:.2f} sec")
                
    return

if __name__ == "__main__":
    main()