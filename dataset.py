import numpy as np
import torch
import scipy
import scipy.io
import gdown
import os
import dgl
from os import path
from google_drive_downloader import GoogleDriveDownloader as gdd
from ogb.nodeproppred import DglNodePropPredDataset
from igb.dataloader import IGB260MDGLDataset

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

class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25, label_num_per_class=20):
        """
        split_type: 'random' for random splitting, 'class' for splitting with equal node num per class
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        label_num_per_class: num of nodes per class
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        elif split_type == 'class':
            train_idx, valid_idx, test_idx = class_rand_splits(self.label, label_num_per_class=label_num_per_class)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_dataset(data_dir, dataname, args=None):
    if dataname in ('ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M'):
        dataset = load_ogb_dataset(data_dir, dataname)
    elif dataname == 'pokec':
        dataset = load_pokec_mat(data_dir)
    elif dataname == "wiki":
        dataset = load_wiki(data_dir)
    elif dataname == "igb":
        dataset = load_igb(args)
    else:
        raise ValueError('Invalid dataname')
    return dataset


# def load_ogb_dataset(data_dir, name):
#     dataset = NCDataset(name)
#     ogb_dataset = NodePropPredDataset(name=name, root=f'{data_dir}/ogb')
#     dataset.graph = ogb_dataset.graph
#     dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
#     dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

#     def ogb_idx_to_tensor():
#         split_idx = ogb_dataset.get_idx_split()
#         tensor_split_idx = {key: torch.as_tensor(
#             split_idx[key]) for key in split_idx}
#         return tensor_split_idx

#     dataset.load_fixed_splits = ogb_idx_to_tensor
#     dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
#     return dataset

def load_ogb_dataset(data_dir, name):
    dataset = NCDataset(name)
    data = DglNodePropPredDataset(name=name, root=f'{data_dir}/ogb')
    graph, labels = data[0]
    graph.ndata['label'] = labels.long()
    dataset.graph['dgl'] = graph
    dataset.label = labels.long()
    split_idx = data.get_idx_split()
    split_idx = {key: torch.as_tensor(
            split_idx[key]) for key in split_idx}
    
    def ogb_idx_to_tensor():
        return split_idx
    
    dataset.load_fixed_splits = ogb_idx_to_tensor
    return dataset

def load_igb(args):
    dataset = NCDataset('igb')
    data = IGB260MDGLDataset(args)
    graph = data[0]
    dataset.graph['dgl'] = graph
    def idgl_idx_to_tensor():
        graph = dataset.graph['dgl']
        train_idx = torch.nonzero(graph.ndata['train_mask'], as_tuple=True)[0]
        val_idx = torch.nonzero(graph.ndata['val_mask'], as_tuple=True)[0]
        test_idx = torch.nonzero(graph.ndata['test_mask'], as_tuple=True)[0]
        split_idx = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_idx
    dataset.load_fixed_splits = idgl_idx_to_tensor
    dataset.label = graph.ndata['label']
    return dataset


def load_pokec_mat(data_dir):
    """ requires pokec.mat """
    if not path.exists(f'{data_dir}/pokec/pokec.mat'):
        gdd.download_file_from_google_drive(
            file_id= dataset_drive_url['pokec'], \
            dest_path=f'{data_dir}/pokec/pokec.mat', showsize=True)
        print('pokec.mat downloaded')

    fulldata = scipy.io.loadmat(f'{data_dir}/pokec/pokec.mat')

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    label = torch.tensor(fulldata['label'].flatten(), dtype=torch.long)
    # Create DGL graph
    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    g.ndata['feat'] = node_feat
    g.ndata['label'] = label
    dataset.graph['dgl'] = g
    dataset.label = label
    return dataset


def load_wiki(data_dir):

    if not path.exists(f'{data_dir}wiki_features2M.pt'):
        gdown.download(id=dataset_drive_url['wiki_features'], \
            output=f'{data_dir}wiki_features2M.pt', quiet=False)
    
    if not path.exists(f'{data_dir}wiki_edges2M.pt'):
        gdown.download(id=dataset_drive_url['wiki_edges'], \
            output=f'{data_dir}wiki_edges2M.pt', quiet=False)

    if not path.exists(f'{data_dir}wiki_views2M.pt'):
        gdown.download(id=dataset_drive_url['wiki_views'], \
            output=f'{data_dir}wiki_views2M.pt', quiet=False)


    dataset = NCDataset("wiki") 
    features = torch.load(f'{data_dir}wiki_features2M.pt')
    edges = torch.load(f'{data_dir}wiki_edges2M.pt').T
    row, col = edges
    # print(f"edges shape: {edges.shape}")
    label = torch.load(f'{data_dir}wiki_views2M.pt') 
    num_nodes = label.shape[0]

    # Create DGL graph
    g = dgl.graph((edges[0], edges[1]), num_nodes=num_nodes)
    g.ndata['feat'] = features
    g.ndata['label'] = label

    dataset.graph['dgl'] = g
    dataset.label = label 
    return dataset 


def load_fixed_splits(data_dir, dataset, name, protocol):
    if name in ['cora', 'citeseer', 'pubmed'] and protocol == 'semi':
        splits = {}
        splits['train'] = torch.as_tensor(dataset.train_idx)
        splits['valid'] = torch.as_tensor(dataset.valid_idx)
        splits['test'] = torch.as_tensor(dataset.test_idx)
    elif name in ['pokec']:
        split = np.load(f'{data_dir}/{name}/{name}-splits.npy', allow_pickle=True)
        splits = {}
        splits['train'] = torch.from_numpy(np.asarray(split[0]['train']))
        splits['valid'] = torch.from_numpy(np.asarray(split[0]['valid']))
        splits['test'] = torch.from_numpy(np.asarray(split[0]['test'])) 
    else:
        raise NotImplementedError

    return splits

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


