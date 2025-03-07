import torch
import torch.nn.functional as F
import numpy as np
import time
from sklearn.metrics import roc_auc_score, f1_score
import gds_read
import concurrent.futures


def eval_f1(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true, y_pred, average='micro')
        acc_list.append(f1)

    return sum(acc_list) / len(acc_list)


def eval_acc(y_true, y_pred):
    start= time.time()
    acc_list = []
    # print("y_pred shape: ", y_pred.shape)
    # print("y_true shape: ", y_true.shape)
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    # print("y_pred: ", y_pred)
    # print("y_true: ", y_true)

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))
    
    eval_time = time.time() - start
    # print(f'Eval time in accuracy on CPU: {eval_time:.4f}')

    return sum(acc_list) / len(acc_list)

def eval_acc_gpu(y_true, y_pred):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    
    acc_list = []
    y_pred = y_pred.argmax(dim=-1, keepdim=True)
    
    for i in range(y_true.shape[1]):
        # Use torch.isnan() to create a mask of non-NaN (labeled) values
        is_labeled = ~torch.isnan(y_true[:, i])
        correct = (y_true[is_labeled, i] == y_pred[is_labeled, i]).float()
        acc_list.append(correct.sum() / correct.numel())
    
    end.record()
    
    # Waits for everything to finish running
    torch.cuda.synchronize()
    
    eval_time = start.elapsed_time(end)
    # print(f'Eval time in accuracy on GPU: {eval_time / 1000:.4f} seconds')
    
    return sum(acc_list) / len(acc_list)

def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            try:
                score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
            except Exception as e:
                print("Error Message: ", e)
                print("is_label: ", is_labeled)
                print("i: ", i)
                print("y_true[is_labeled, i]: ", y_true[is_labeled, i])
                print("y_pred[is_labeled, i]: ", y_pred[is_labeled, i])
                exit()

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)

@torch.no_grad()
def evaluate_batch_preload(model, split_idx, labels_all, train_data, valid_data, test_data, eval_func, criterion, device, num_classes, args):
    model = model.to(device)
    model.eval()
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']
    train_idx_cmp = torch.arange(0, len(split_idx['train']))
    valid_idx_cmp = torch.arange(0, len(split_idx['valid']))
    test_idx_cmp = torch.arange(0, len(split_idx['test']))
    labels_all = labels_all.to(device)

    if args.eval_batch:
        train_idx_batch = list(torch.split(train_idx_cmp, args.batch_size))
        valid_idx_batch = list(torch.split(valid_idx_cmp, args.batch_size))
        test_idx_batch = list(torch.split(test_idx_cmp, args.batch_size))

        num_train_batch = len(train_idx_batch)
        num_valid_batch = len(valid_idx_batch)
        num_test_batch = len(test_idx_batch)

        out_train = []
        for i in range (num_train_batch):
            train_idx_batch[i] = train_idx_batch[i]
            if args.cat_input or args.method == 'SGC':
                train_batch = train_data[train_idx_batch[i], :].to(device)
            else:    
                train_batch = train_data[train_idx_batch[i], :, :].to(device)
            out = model(train_batch)
            out_train.append(out)
        out_train = torch.cat(out_train, dim=0)
        label_train = labels_all[train_idx]
        train_acc = eval_func(label_train, out_train)

        out_valid = []
        for i in range (num_valid_batch):
            valid_idx_batch[i] = valid_idx_batch[i]
            if args.cat_input or args.method == 'SGC':
                valid_batch = valid_data[valid_idx_batch[i], :].to(device)
            else:
                valid_batch = valid_data[valid_idx_batch[i], :, :].to(device)
            out = model(valid_batch)
            out_valid.append(out)
        out_valid = torch.cat(out_valid, dim=0)
        label_valid = labels_all[valid_idx]
        valid_acc = eval_func(label_valid, out_valid)

        out_test = []
        for i in range (num_test_batch):
            test_idx_batch[i] = test_idx_batch[i]
            if args.cat_input or args.method == 'SGC':
                test_batch = test_data[test_idx_batch[i], :].to(device)
            else:
                test_batch = test_data[test_idx_batch[i], :, :].to(device)
            out = model(test_batch)
            out_test.append(out)
        out_test = torch.cat(out_test, dim=0)
        label_test = labels_all[test_idx]
        test_acc = eval_func(label_test, out_test)

    else:
        train_data = train_data.to(device)
        valid_data = valid_data.to(device)
        test_data = test_data.to(device)
        train_acc = eval_func(labels_all[train_idx], model(train_data))
        out_valid = model(valid_data)
        valid_acc = eval_func(labels_all[valid_idx], out_valid)
        test_acc = eval_func(labels_all[test_idx], model(test_data)) 
    
    labels_valid = labels_all[valid_idx]
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins', 'questions'):
        if labels_valid.shape[1] == 1:
            true_label = F.one_hot(labels_valid,  num_classes).squeeze(1)
        else:
            true_label = labels_valid
        valid_loss = criterion(out_valid, true_label.squeeze(1).to(torch.float))
    else:
        out_valid = F.log_softmax(out_valid, dim=1)
        valid_loss = criterion(out_valid, labels_valid.squeeze(1))

    return train_acc, valid_acc, test_acc, valid_loss, out_valid   

def seq_eval(model, num_batch, in_dim, element_size, args, idx, file_path_da, file_path_dad, file_path_ori):
    out = []
    for i in range(num_batch):
        data_offset = i * args.eval_batch_size * (in_dim//args.col_split) * element_size
        if i == num_batch - 1:
            data_length = (len(idx) - i * args.eval_batch_size) * (in_dim//args.col_split) * element_size
        else:
            data_length = args.eval_batch_size * (in_dim//args.col_split) * element_size
        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
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
        data_batch = []
        results = [future.result() for future in futures]
        if args.method == 'SGC':
            for i in range(args.col_split):                           
                reshaped_data = results[i].view(torch.float32).view(-1, in_dim//args.col_split)
                data_batch.append(reshaped_data)
            data_batch = torch.cat(data_batch, dim=1)
        else:
            buffer_elements = []
            for i in range(args.col_split):
                reshaped_data = results[i].view(torch.float32).view(-1, in_dim//args.col_split)
                buffer_elements.append(reshaped_data)
            buffer_elements = torch.cat(buffer_elements, dim=1)
            data_batch.append(buffer_elements)
            
            for i in range(0, args.training_hops):
                buffer_elements = []
                for split_idx in range(args.col_split):
                    reshaped_data = results[args.col_split * i + split_idx + args.col_split].view(torch.float32).view(-1, in_dim//args.col_split)
                    buffer_elements.append(reshaped_data)
                buffer_elements = torch.cat(buffer_elements, dim=1)
                data_batch.append(buffer_elements)

            if args.cat_input:
                data_batch = torch.cat(data_batch, dim=1)
            else:
                data_batch = torch.stack(data_batch, dim=1)
        out.append(model(data_batch))
    return torch.cat(out, dim=0)
        

@torch.no_grad()
def evaluate_batch_gds(model, split, data_path, labels, in_dim, element_size, eval_func, criterion, device, num_classes,args):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    # print("Invalid labels at the beginning:", labels[labels == -1])
    model.eval()
    out_train = []
    out_valid = []
    out_test = []
    
    # labels = labels.to(device)
    train_idx = split['train']
    valid_idx = split['valid']
    test_idx = split['test']
    labels_train = labels[train_idx].to(device)
    labels_valid = labels[valid_idx].to(device)
    labels_test = labels[test_idx].to(device)
    train_file_path_ad = data_path + str(args.dataset) + 'train_ad'
    train_file_path_da = data_path + str(args.dataset) + 'train_da'
    train_file_path_dad = data_path + str(args.dataset) + 'train_dad'
    train_file_path_ori = data_path + str(args.dataset) + 'train_hop0'
    valid_file_path_ad = data_path + str(args.dataset) + 'val_ad'
    valid_file_path_da = data_path + str(args.dataset) + 'val_da'
    valid_file_path_dad = data_path + str(args.dataset) + 'val_dad'
    valid_file_path_ori = data_path + str(args.dataset) + 'val_hop0'
    test_file_path_ad = data_path + str(args.dataset) + 'test_ad'
    test_file_path_da = data_path + str(args.dataset) + 'test_da'
    test_file_path_dad = data_path + str(args.dataset) + 'test_dad'
    test_file_path_ori = data_path + str(args.dataset) + 'test_hop0'


    num_train_batch = len(train_idx) // args.eval_batch_size
    if len(train_idx) % args.eval_batch_size != 0:
        num_train_batch += 1
    num_valid_batch = len(valid_idx) // args.eval_batch_size
    if len(valid_idx) % args.eval_batch_size != 0:
        num_valid_batch += 1
    num_test_batch = len(test_idx) // args.eval_batch_size
    if len(test_idx) % args.eval_batch_size != 0:
        num_test_batch += 1

    # print(f'num_train_batch: {num_train_batch}, num_valid_batch: {num_valid_batch}, num_test_batch: {num_test_batch}')
    out_train = seq_eval(model, num_train_batch, in_dim, element_size, args, train_idx, train_file_path_da, train_file_path_dad, train_file_path_ori)
    out_valid = seq_eval(model, num_valid_batch, in_dim, element_size, args, valid_idx, valid_file_path_da, valid_file_path_dad, valid_file_path_ori)
    out_test = seq_eval(model, num_test_batch, in_dim, element_size, args, test_idx, test_file_path_da, test_file_path_dad, test_file_path_ori)
    
    train_acc = eval_func(labels_train, out_train)
    valid_acc = eval_func(labels_valid, out_valid)
    test_acc = eval_func(labels_test, out_test)

    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    eval_time = start.elapsed_time(end)
    # print(f'Eval w/o loss time on GPU: {eval_time / 1000:.4f} seconds')
    out_valid = out_valid.to('cpu')
    labels_valid = labels_valid.to('cpu')
    # print(f'shape of out_valid: {out_valid.shape}, shape of label_valid: {label_valid.shape}')
    # print("Invalid labels:", label_valid[label_valid == -1])
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins', 'questions'):
        if labels_valid.shape[1] == 1:
            true_label = F.one_hot(labels_valid,  num_classes).squeeze(1)
        else:
            true_label = labels_valid
        valid_loss = criterion(out_valid, true_label.squeeze(1).to(torch.float))  
    else:
        out_valid = F.log_softmax(out_valid, dim=1)
        if labels_valid.dim()>1:
            valid_loss = criterion(out_valid, labels_valid.squeeze(1))
        else:
            valid_loss = criterion(out_valid, labels_valid)

    return train_acc, valid_acc, test_acc, valid_loss, out_valid
        


@torch.no_grad()
def evaluate_batch_sage(model, sub_graph_loader, node_feat, labels, split_idx, eval_func, criterion, device, args):
    
    model.eval()
    out = model.inference(node_feat, sub_graph_loader, device)
    train_acc = eval_func(labels[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(labels[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(labels[split_idx['test']], out[split_idx['test']])
    # print(f' out is on device: {out.device}')
    # print(f' labels is on device: {labels.device}')
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
    return train_acc, valid_acc, test_acc, valid_loss, out


@torch.no_grad()
def evaluate_sage(model, adj, node_feat, labels, split_idx, eval_func, criterion, device, args):
    model.eval()
    out = model(node_feat, adj)
    train_acc = eval_func(labels[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(labels[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(labels[split_idx['test']], out[split_idx['test']])
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
    return train_acc, valid_acc, test_acc, valid_loss, out




@torch.no_grad()
def evaluate_cpu(model, dataset, split_idx, eval_func, criterion, args, result=None):
    model.eval()

    model.to(torch.device("cpu"))
    dataset.label = dataset.label.to(torch.device("cpu"))
    adjs_, x = dataset.graph['adjs'], dataset.graph['node_feat']
    adjs = []
    adjs.append(adjs_[0])
    for k in range(args.rb_order - 1):
        adjs.append(adjs_[k + 1])
    if args.method == 'nodeformer':
        out, _ = model(x, adjs)
    elif args.method == 'difformer':
        out = model(x, adjs[0])
    elif args.methond == 'HOGA':
        out = model(x)
    else:
        out = model(x, adjs[0])

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins', 'questions'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out
