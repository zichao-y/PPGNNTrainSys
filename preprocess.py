import torch
import argparse
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_sparse import SparseTensor
from sklearn.neighbors import kneighbors_graph
import numpy as np
import scipy
from parse import parser_add_main_args
from dataset import load_dataset
from data_utils import load_fixed_splits
import os
from tqdm import tqdm
import time
import gc
from igb.dataloader import IGB260MDGLDataset
from ogb.nodeproppred import DglNodePropPredDataset
import dgl


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
        norm_adj_dad_cpu = graph2adj_coo(graph.edges(), nnodes, output_type='dad')
    if kernel == 'da' or kernel == 'dual':
        norm_adj_da_cpu = graph2adj_coo(graph.edges(), nnodes, output_type='da')
    
    
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
                    high_order_features_da = spmm_blocks_sptensor(norm_adj_da_cpu, high_order_features_da, device, block_size, block_col_size)
                if kernel == 'dad' or kernel == 'dual':
                    high_order_features_dad = spmm_blocks_sptensor(norm_adj_dad_cpu, high_order_features_dad, device, block_size, block_col_size)
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

def pagraph_save(edge_index, x, save_dir, name, y, split):
    data_path = save_dir + 'pagraph/' + name + '/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    nnodes = x.shape[0]
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(nnodes, nnodes))
    adj = adj.to_scipy(layout='coo')
    scipy.sparse.save_npz(data_path + 'adj.npz', adj)
    # convert x to numpy array
    np.save(data_path + 'feat.npy', x.numpy())
    np.save(data_path + 'labels.npy', y.numpy())
    # convert split into masks for train, valid, test
    train_mask = torch.zeros(nnodes, dtype=torch.bool)
    train_mask[split['train']] = 1
    valid_mask = torch.zeros(nnodes, dtype=torch.bool)
    valid_mask[split['valid']] = 1
    test_mask = torch.zeros(nnodes, dtype=torch.bool)
    test_mask[split['test']] = 1
    np.save(data_path + 'train.npy', train_mask.numpy())
    np.save(data_path + 'val.npy', valid_mask.numpy())
    np.save(data_path + 'test.npy', test_mask.numpy())
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
    adj = SparseTensor(row=row, col=col, sparse_sizes=(nnodes, nnodes))
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
    if args.dataset == 'igb':
        dataset = IGB260MDGLDataset(args)
        graph = dataset[0]
        c = args.num_classes
        d = 1024
        train_idx = torch.nonzero(graph.ndata['train_mask'], as_tuple=True)[0]
        val_idx = torch.nonzero(graph.ndata['val_mask'], as_tuple=True)[0]
        test_idx = torch.nonzero(graph.ndata['test_mask'], as_tuple=True)[0]
        split_idx = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        
    elif args.dataset == 'ogbn-papers100M':
        dataset = DglNodePropPredDataset(name=args.dataset, root=f'{args.data_dir}/ogb')
        graph, labels = dataset[0]
        # Convert the graph to an undirected (bidirected) graph
        g_sym = dgl.to_bidirected(graph)
        g_sym.ndata.update(graph.ndata)
        graph=g_sym
        # Obtain the split indices
        split_idx = dataset.get_idx_split()
        split_idx = {key: torch.as_tensor(
            split_idx[key]) for key in split_idx}
        c = 172
        d = graph.ndata['feat'].shape[1]
        n = graph.num_nodes()
        graph.ndata['label'] = labels.long()
    else:
        dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)
        if len(dataset.label.shape) == 1:
            dataset.label = dataset.label.unsqueeze(1)
        
        if args.rand_split:
            split_idx = dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                            
        elif args.rand_split_class:
            split_idx = dataset.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                            
        elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products', 'amazon2m']:
            split_idx = dataset.load_fixed_splits()
                           
        else:
            if args.dataset == 'fb100':
                name='fb100-'+args.sub_dataset
            else:
                name=args.dataset
            split_idx = load_fixed_splits(args.data_dir, dataset, name=name, protocol=args.protocol)

        if args.dataset in ('mini', '20news'):
            adj_knn = kneighbors_graph(dataset.graph['node_feat'], n_neighbors=args.knn_num, include_self=True)
            edge_index = torch.tensor(adj_knn.nonzero(), dtype=torch.long)
            dataset.graph['edge_index']=edge_index

        # whether or not to symmetrize
        if not args.directed and args.dataset != 'ogbn-proteins':
            dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        # whether or not to add self loops
        if args.self_loops:
            dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
            dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=dataset.graph['num_nodes'])
        edge_index = dataset.graph['edge_index']
        graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=dataset.graph['num_nodes'])
        graph.ndata['feat'] = dataset.graph['node_feat'].contiguous()
        graph.ndata['label'] = dataset.label.long().contiguous()
        
    
    if args.tognnlab:
        quantize = True if args.data_type in ['int8', 'e4m3', 'e3m4', 'bf16', 'fp16'] else False
        if args.dataset == 'igb':
            name = 'igb_' + args.dataset_size
        else:
            name = args.dataset
        if args.dataset == 'ogbn-papers100M':
            gnnlab_save(graph.edges(), graph.ndata['feat'], args.save_dir, name, labels, split_idx, use_pca=args.use_pca, PCA_dim=args.pca_dim, QUANT=quantize, data_type=args.data_type)
        else:
            gnnlab_save(dataset.graph['edge_index'], dataset.graph['node_feat'], args.save_dir, name, dataset.label, split_idx, use_pca=args.use_pca, PCA_dim=args.pca_dim, QUANT=quantize, data_type=args.data_type)
        exit()
    
    if args.dataset == 'igb' or args.dataset == 'ogbn-papers100M':
        labels = graph.ndata['label']
    else:
        labels = dataset.label
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
        data_dict = {"labels": labels, "splits": split_idx, "in_dim": dataset.graph['node_feat'].shape[1], "num_nodes": dataset.graph['num_nodes'], "edge_index": dataset.graph['edge_index']}
    
    torch.save(data_dict, dir_path + args.dataset + '_aux.pt')
    print(f"Preprocessing done, processing time: {process_end - time_start:.2f} sec, loading time: {time_start - now:.2f} sec, total time: {time.time() - now:.2f} sec")
                
    return

if __name__ == "__main__":
    main()