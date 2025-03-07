import os
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch_geometric.datasets import HeterophilousGraphDataset, WikiCS
import numpy as np
from scipy import sparse as sp
from torch.utils.data import Dataset
from torch_sparse import SparseTensor
import gdown
import h5py
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

class ChunkedDataset(Dataset):
    def __init__(self, labels, file_path, index, chunk_size, name, type='simple', num_hop=1):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.index = index
        self.length = len(index)
        self.num_hop = num_hop
        self.name = name
        self.type = type
        self.labels = labels
        self.chunked_index = []
        self.__create_chunk__()
        
    def __create_chunk__(self):
        # sort self.index
        self.index = torch.sort(self.index)[0]
        # create chunks of data
        num_chunks = self.length // self.chunk_size
        if self.length % self.chunk_size > 0:
            num_chunks += 1
        self.chunked_index = [self.index[i * self.chunk_size:(i + 1) * self.chunk_size] for i in range(num_chunks-1)]
        self.chunked_index.append(self.index[(num_chunks-1) * self.chunk_size:])
        return
        
    def __getitem__(self, idx):
        # return a chunk of data
        # read data from file and return data at position of chunked_index[idx]
        # get the start and end index of the chunk
        start = self.chunked_index[idx][0]
        end = self.chunked_index[idx][-1]+1
        relative_index = self.chunked_index[idx] - start
        train_data =  load_data_chunk(self.file_path, start=start, end=end, index=relative_index, name=self.name, type=self.type, num_hop=self.num_hop)
        train_label = self.labels[self.chunked_index[idx]]
        return train_data, train_label

    def __len__(self):
        N = self.length // self.chunk_size
        if self.length % self.chunk_size > 0:
            N += 1
        return N
    
class BatchedDataset(Dataset):
    def __init__(self, train_data, labels, batch_size):
        self.train_data = train_data.contiguous()
        self.length = len(train_data)
        self.labels = labels.contiguous()
        self.batch_size = batch_size
        
    def __getitem__(self, idx):
        # assume the train_data is a list of data, each time getitem fetch a batch of data instead of a single data
        # the batch is contigous in train_data, starting from idx * batch_size
        start = idx * self.batch_size
        end = min((idx+1) * self.batch_size, self.length)
        train_data = self.train_data[start:end]
        train_label = self.labels[start:end]
        return train_data, train_label

    def __len__(self):
        N = self.length // self.batch_size
        if self.length % self.batch_size > 0:
            N += 1
        return N
    

def load_data_chunk(file_path, start, end, index, name, type='simple', num_hop=1):
    x = []
    for i in range(num_hop+1):       
        file_name = file_path + name + f'_hop{i}.hdf5'
        with h5py.File(file_name, "r") as file:
        # Load a slice of the dataset
            if type == 'simple':
                feat_chunk = file['x_simple'][start:end]
            else:
                feat_chunk = file['x_cat'][start:end]
        feat_chunk = torch.from_numpy(feat_chunk) 
        feat = feat_chunk[index]
        x.append(feat)
    x = torch.stack(x, dim=1)
    return x    

class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        assert self.x.size(0) == self.y.size(0)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


def load_fixed_splits(data_dir, dataset, name, protocol):
    splits_lst = []
    if name in ['cora', 'citeseer', 'pubmed'] and protocol == 'semi':
        splits = {}
        splits['train'] = torch.as_tensor(dataset.train_idx)
        splits['valid'] = torch.as_tensor(dataset.valid_idx)
        splits['test'] = torch.as_tensor(dataset.test_idx)
        splits_lst.append(splits)
    elif name in ['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'film', 'cornell', 'texas', 'wisconsin']:
        for i in range(10):
            splits_file_path = '{}/geom-gcn/splits/{}'.format(data_dir, name) + '_split_0.6_0.2_' + str(i) + '.npz'
            splits = {}
            with np.load(splits_file_path) as splits_file:
                splits['train'] = torch.BoolTensor(splits_file['train_mask'])
                splits['valid'] = torch.BoolTensor(splits_file['val_mask'])
                splits['test'] = torch.BoolTensor(splits_file['test_mask'])
            splits_lst.append(splits)
    elif name in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions', 'chameleon-filtered', 'squirrel-filtered']:
        print(">>>>> Using fixed split for < {} >".format(name))
        torch_dataset = HeterophilousGraphDataset(name=name.capitalize(), root=data_dir)
        data = torch_dataset[0]
        for i in range(data.train_mask.shape[1]):
            splits = {}
            splits['train'] = torch.where(data.train_mask[:,i])[0]
            splits['valid'] = torch.where(data.val_mask[:,i])[0]
            splits['test'] = torch.where(data.test_mask[:,i])[0]
            # print("splits[train]: ", splits['train'])
            # print("splits[valid]: ", splits['valid'])
            # print("splits[test]: ", splits['test'])
            splits_lst.append(splits)
    elif name in ['wikics']:
        torch_dataset = WikiCS(root="dataset/wikics/")
        data = torch_dataset[0]
        for i in range(data.train_mask.shape[1]):
            splits = {}
            splits['train'] = torch.where(data.train_mask[:,i])[0]
            splits['valid'] = torch.where(torch.logical_or(data.val_mask, data.stopping_mask)[:,i])[0]
            splits['test'] = torch.where(data.test_mask[:])[0]
            splits_lst.append(splits)
    elif name in ['pokec']:
        split = np.load(f'../data/{name}/{name}-splits.npy', allow_pickle=True)
        for i in range(split.shape[0]):
            splits = {}
            splits['train'] = torch.from_numpy(np.asarray(split[i]['train']))
            splits['valid'] = torch.from_numpy(np.asarray(split[i]['valid']))
            splits['test'] = torch.from_numpy(np.asarray(split[i]['test']))
            splits_lst.append(splits)
    elif name in ['genius', 'fb100-Penn94', 'twitch-gamer', 'arxiv-year', 'snap-patents']:
        if not os.path.exists(f'../data/splits/{name}-splits.npy'):
            gdown.download(
                id=splits_drive_url[name], \
                output=f'../data/splits/{name}-splits.npy', quiet=False)
        split = np.load(f'../data/splits/{name}-splits.npy', allow_pickle=True)
        print(f'split.shape: {split.shape}')
        for i in range(split.shape[0]):
            splits = {}
            splits['train'] = torch.from_numpy(np.asarray(split[i]['train']))
            splits['valid'] = torch.from_numpy(np.asarray(split[i]['valid']))
            splits['test'] = torch.from_numpy(np.asarray(split[i]['test']))
            splits_lst.append(splits)
        
    else:
        raise NotImplementedError

    return splits_lst


def class_rand_splits(label, label_num_per_class):
    train_idx, non_train_idx = [], []
    idx = torch.arange(label.shape[0])
    class_list = label.squeeze().unique()
    valid_num, test_num = 500, 1000
    for i in range(class_list.shape[0]):
        c_i = class_list[i]
        idx_i = idx[label.squeeze() == c_i]
        n_i = idx_i.shape[0]
        rand_idx = idx_i[torch.randperm(n_i)]
        train_idx += rand_idx[:label_num_per_class].tolist()
        non_train_idx += rand_idx[label_num_per_class:].tolist()
    train_idx = torch.as_tensor(train_idx)
    non_train_idx = torch.as_tensor(non_train_idx)
    non_train_idx = non_train_idx[torch.randperm(non_train_idx.shape[0])]
    valid_idx, test_idx = non_train_idx[:valid_num], non_train_idx[valid_num:valid_num + test_num]

    return train_idx, valid_idx, test_idx


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.quantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label

def to_planetoid(dataset):
    """
        Takes in a NCDataset and returns the dataset in H2GCN Planetoid form, as follows:
        x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ty => the one-hot labels of the test instances as numpy.ndarray object;
        ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        split_idx => The ogb dictionary that contains the train, valid, test splits
    """
    split_idx = dataset.get_idx_split('random', 0.25)
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    graph, label = dataset[0]

    label = torch.squeeze(label)

    print("generate x")
    x = graph['node_feat'][train_idx].numpy()
    x = sp.csr_matrix(x)

    tx = graph['node_feat'][test_idx].numpy()
    tx = sp.csr_matrix(tx)

    allx = graph['node_feat'].numpy()
    allx = sp.csr_matrix(allx)

    y = F.one_hot(label[train_idx]).numpy()
    ty = F.one_hot(label[test_idx]).numpy()
    ally = F.one_hot(label).numpy()

    edge_index = graph['edge_index'].T

    graph = defaultdict(list)

    for i in range(0, label.shape[0]):
        graph[i].append(i)

    for start_edge, end_edge in edge_index:
        graph[start_edge.item()].append(end_edge.item())

    return x, tx, allx, y, ty, ally, graph, split_idx


def to_sparse_tensor(edge_index, edge_feat, num_nodes):
    """ converts the edge_index into SparseTensor
    """
    num_edges = edge_index.size(1)

    (row, col), N, E = edge_index, num_nodes, num_edges
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]

    value = edge_feat[perm]
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(N, N), is_sorted=True)

    # Pre-process some important attributes.
    adj_t.storage.rowptr()
    adj_t.storage.csr2csc()

    return adj_t


def normalize(edge_index):
    """ normalizes the edge_index
    """
    adj_t = edge_index.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    return adj_t


def gen_normalized_adjs(dataset):
    """ returns the normalized adjacency matrix
    """
    row, col = dataset.graph['edge_index']
    N = dataset.graph['num_nodes']
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)
    D_isqrt = deg.pow(-0.5)
    D_isqrt[D_isqrt == float('inf')] = 0

    DAD = D_isqrt.view(-1,1) * adj * D_isqrt.view(1,-1)
    DA = D_isqrt.view(-1,1) * D_isqrt.view(-1,1) * adj
    AD = adj * D_isqrt.view(1,-1) * D_isqrt.view(1,-1)
    return DAD, DA, AD

def convert_to_adj(edge_index,n_node):
    '''convert from pyg format edge_index to n by n adj matrix'''
    adj=torch.zeros((n_node,n_node))
    row,col=edge_index
    adj[row,col]=1
    return adj

def adj_mul(adj_i, adj, N):
    adj_i_sp = torch.sparse_coo_tensor(adj_i, torch.ones(adj_i.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_sp = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_j = torch.sparse.mm(adj_i_sp, adj_sp)
    adj_j = adj_j.coalesce().indices()
    return adj_j


dataset_drive_url = {
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y', 
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ', 
    'wiki_views': '1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP', # Wiki 1.9M 
    'wiki_edges': '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u', # Wiki 1.9M 
    'wiki_features': '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK' # Wiki 1.9M
}

splits_drive_url = {
    'snap-patents' : '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N', 
    'pokec' : '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_',
}
