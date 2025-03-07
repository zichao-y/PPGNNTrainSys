import numpy as np
import torch
import scipy
import scipy.io
import gdown
from data_utils import rand_train_test_idx, dataset_drive_url, class_rand_splits
from torch_geometric.datasets import WikiCS
import os
from os import path
from google_drive_downloader import GoogleDriveDownloader as gdd
from ogb.nodeproppred import NodePropPredDataset
from igb import download as igb_download


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


def load_dataset(data_dir, dataname, sub_dataname=''):
    if dataname in ('ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M'):
        dataset = load_ogb_dataset(data_dir, dataname)
    elif dataname == 'pokec':
        dataset = load_pokec_mat(data_dir)
    elif dataname == 'wikics':
        dataset = load_wikics_dataset(data_dir)
    elif dataname == "wiki":
        dataset = load_wiki(data_dir)
    elif dataname == "igb":
        dataset = load_igb_dataset(data_dir, sub_dataname)
    else:
        raise ValueError('Invalid dataname')
    return dataset


def load_ogb_dataset(data_dir, name):
    dataset = NCDataset(name)
    ogb_dataset = NodePropPredDataset(name=name, root=f'{data_dir}/ogb')
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    def ogb_idx_to_tensor():
        split_idx = ogb_dataset.get_idx_split()
        tensor_split_idx = {key: torch.as_tensor(
            split_idx[key]) for key in split_idx}
        return tensor_split_idx

    dataset.load_fixed_splits = ogb_idx_to_tensor
    dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    return dataset


def load_wikics_dataset(data_dir):
    wikics_dataset = WikiCS(root='dataset/wikics/')
    data = wikics_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset('wikics')
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

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
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = fulldata['label'].flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)
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

    # print(f"features shape: {features.shape[0]}")
    # print(f"Label shape: {label.shape[0]}")
    dataset.graph = {"edge_index": edges, 
                     "edge_feat": None, 
                     "node_feat": features, 
                     "num_nodes": num_nodes}
    dataset.label = label 
    return dataset 

def load_igb_dataset(data_dir, size):
    
    if not path.exists(f'{data_dir}/IGB/{size}/'):
        igb_download.download_dataset(path=f'{data_dir}/IGB/', dataset_type='homogeneous', dataset_size=size)
    
    if size == 'tiny':
        num_nodes = 100000
    elif size == 'small':
        num_nodes = 1000000
    elif size == 'medium':
        num_nodes = 10000000
    elif size == 'large':
        num_nodes = 100000000
    elif size == 'full':
        num_nodes = 269346174

    if size == 'large' or size == 'full':
        path_feat = os.path.join(data_dir, 'IGB', 'full', 'processed', 'paper', 'node_feat.npy')
        node_feature = np.memmap(path_feat, dtype='float32', mode='r',  shape=(num_nodes,1024))
    else:
        path_feat = os.path.join(data_dir, 'IGB', size, 'processed', 'paper', 'node_feat.npy')            
        node_feature = np.load(path_feat)
    node_feature = torch.from_numpy(node_feature)

    if size == 'large' or size == 'full':
        path_label = os.path.join(data_dir, 'IGB', 'full', 'processed', 'paper', 'node_label_19.npy')
        node_label = np.memmap(path_label, dtype='int32', mode='r', shape=(num_nodes))
    else:
        path_label = os.path.join(data_dir, 'IGB', size, 'processed', 'paper', 'node_label_19.npy')            
        node_label = np.load(path_label)
    node_label = torch.from_numpy(node_label).to(torch.long)

    path_edge = os.path.join(data_dir, 'IGB', size, 'processed', 'paper__cites__paper', 'edge_index.npy')
    edge_index = np.load(path_edge)
    edge_index = torch.from_numpy(edge_index)
    edge_index = edge_index.t()
    print(f"edge_index shape: {edge_index.shape}")

    dataset = NCDataset(f'IGB_{size}')
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feature,
                     'num_nodes': num_nodes}
    dataset.label = node_label
    return dataset





