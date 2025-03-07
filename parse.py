from gnns import *
import argparse
import json

def parse_method(args, c, d, device):
    if args.method == 'SIGN':
        model = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                    out_channels=c, num_layers=args.num_layers,
                    dropout=args.dropout).to(device)
    elif args.method == 'SGC':
        model = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                    out_channels=c, num_layers=1,
                    dropout=args.dropout).to(device)
    elif args.method == 'HOGA':     
        model=HOGA(args.num_layers,d,args.hidden_channels, args.num_heads, c, \
                  dropout=args.dropout, attn_drop=args.attn_dropout,in_drop=args.input_dropout, \
                  training_hops=args.training_hops, mlplayers=args.mlplayers, \
                  use_post_residul=args.use_post_res,mlp_hidden=args.mlp_hidden, \
                  mlp_dropout=args.mlp_dropout).to(device)
    else:
        raise ValueError('Invalid method')
    return model

def load_args():
    parser = argparse.ArgumentParser(description='General Training Pipeline')
    
    # Add your “main” arguments
    parser_add_main_args(parser)
    
    # Add an argument for the config file
    parser.add_argument('--model_config', type=str, default=None,
                        help='Path to JSON config file to override defaults')
    parser.add_argument('--pipeline_config', type=str, default=None,
                        help='Path to JSON config file to override defaults')
    parser.add_argument('--sys_json', type=str, default='system_par.json', help='Path to system parameter JSON file ')
    
    # First, parse known args to get config file paths.
    args, remaining_args = parser.parse_known_args()
    
    config = {}
    
    # If user has provided a model config file, load and update the config defaults.
    if args.model_config is not None:
        with open(args.model_config, 'r') as f:
            model_config = json.load(f)
        config.update(model_config)
    
    # Similarly, load pipeline config if provided.
    if args.pipeline_config is not None:
        with open(args.pipeline_config, 'r') as f:
            pipeline_config = json.load(f)
        config.update(pipeline_config)
    
    # Now update parser defaults with the config values.
    parser.set_defaults(**config)
    
    # Reparse the command-line. This time, any command-line argument provided
    # will override the defaults coming from the JSON config files.
    args = parser.parse_args()
    
    return args    


def parser_add_main_args(parser):
    # dataset, protocol
    parser.add_argument('--method', '-m', type=str, default='nodeformer')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--json_path', type=str, default='./')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--eval_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0])
    parser.add_argument('--host_mem_thr', type=int, nargs='+', default=[0.4, 0.7, 0.5])
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.6,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.2,
                        help='validation label proportion')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--self_loops', action='store_true', help='add self loops')
    parser.add_argument('--protocol', type=str, default='semi',
                        help='protocol for cora datasets with fixed splits, semi or supervised')
    parser.add_argument('--kernel_type', type=str, default='da')
    parser.add_argument('--GPUcap', type=int, default=4)
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--label_num_per_class', type=int, default=20, help='labeled nodes randomly selected')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'],
                        help='evaluation metric')
    parser.add_argument('--pin_memory', action='store_true', help='use pin memory for data loader')
    parser.add_argument('--save_result', action='store_true', help='whether to save result')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='./model/')
    parser.add_argument('--trail_profile', action='store_true', help='whether to profile memory usage')
    # hyper-parameter for model arch and training
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--eval_load_host', action='store_true')
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--use_residual', action='store_true', help='use residual link for each GNN layer')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--use_act', action='store_true', help='use non-linearity for each layer')
    parser.add_argument('--batch_size', type=int, default=10000)
    # hyper-parameter for models
    parser.add_argument('--test_batch_size', type=int, default=1024) #
    parser.add_argument('--num_workers', type=int, default=4) #
    parser.add_argument('--hidden_dim', type=int, default=256) 
    parser.add_argument('--attn_dropout', type=float, default=0)      
    parser.add_argument('--test_start_epoch', type=int, default=100)
    parser.add_argument('--patience', type=int, default=50, 
                        help='Patience for early stopping')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
    parser.add_argument('--testing_hops', type=int, default=10)
    parser.add_argument('--training_hops', type=int, default=4)
    parser.add_argument('--plot_curve', action='store_true')
    parser.add_argument('--input_dropout', type=float, default=0.0)
    parser.add_argument('--load_all', action='store_true')
    parser.add_argument('--full_path', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./data/')
    parser.add_argument('--epoch_thr', type=int, default=100)
    parser.add_argument('--acc_thr', type=float, default=0.5)
    parser.add_argument('--mlp_only', action='store_true')
    parser.add_argument('--save_step', type=int, default=20)
    parser.add_argument('--chunk_size', type=int, default=1000)
    parser.add_argument('--graph_sample', action='store_true')
    parser.add_argument('--eval_batch', action='store_true')
    parser.add_argument('--mlplayers', type=int, default=0)
    parser.add_argument('--use_post_res', type=int, default=0)
    parser.add_argument('--input_type', type=str, default='hl')
    parser.add_argument('--json_file_path', type=str, default='./results/timing/')
    parser.add_argument('--full_batch', action='store_true')
    parser.add_argument('--save_json', action='store_true')
    parser.add_argument('--cat_input', action='store_true')
    parser.add_argument('--block_size', type=int, default=0)
    parser.add_argument('--col_size', type=int, default=8)
    parser.add_argument('--mlp_hidden', type=int, default=128)
    parser.add_argument('--mlp_dropout', type=float, default=0.0)
    parser.add_argument('--topagraph', action='store_true')
    parser.add_argument('--tolegion', action='store_true')
    parser.add_argument('--tognnlab', action='store_true')
    parser.add_argument('--save_infer_time', action='store_true')
    parser.add_argument('--enable_profiling', action='store_true')
    parser.add_argument('--mode', type=str, default='uvm')
    parser.add_argument('--enable_profiling_local', action='store_true')
    parser.add_argument('--eval_batch_size', type=int, default=1000)
    parser.add_argument('--col_split', type=int, default=1)
    #for load igb datasets
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

    

    
    






