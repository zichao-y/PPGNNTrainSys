import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAtt(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.0):
        super(MultiheadAtt, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Linear projections for queries, keys, and values
        self.query_projection = nn.Linear(input_dim, input_dim)
        self.key_projection = nn.Linear(input_dim, input_dim)
        self.value_projection = nn.Linear(input_dim, input_dim)

        # Linear projection for the output of the attention heads
        self.output_projection = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        print("use our multihead att model!!!!")

    def reset_parameters(self):
        self.query_projection.reset_parameters()
        self.key_projection.reset_parameters()
        self.value_projection.reset_parameters()
        self.output_projection.reset_parameters()
        

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections for queries, keys, and values
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        # Reshape the projected queries, keys, and values
        # query = query.view(batch_size * self.num_heads, -1, self.head_dim)
        # key = key.view(batch_size * self.num_heads, -1, self.head_dim)
        # value = value.view(batch_size * self.num_heads, -1, self.head_dim)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        query = query.reshape(batch_size * self.num_heads, -1, self.head_dim)
        key = key.reshape(batch_size * self.num_heads, -1, self.head_dim)
        value = value.reshape(batch_size * self.num_heads, -1, self.head_dim)

        # Compute the scaled dot-product attention
        attention_scores = torch.bmm(query, key.transpose(1, 2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Apply the mask (if provided)
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # Compute the output of the attention heads
        attention_output = torch.bmm(attention_probs, value)

        # Reshape and project the output of the attention heads
        # attention_output = attention_output.view(batch_size, -1, self.input_dim)
        attention_output = attention_output.view(batch_size, self.num_heads, -1, self.head_dim)
        attention_output = attention_output.permute(0, 2, 1, 3)
        attention_output = attention_output.reshape(batch_size, -1, self.num_heads * self.head_dim)
        attention_output = self.output_projection(attention_output)

        return attention_output, attention_probs


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        print(f'in_channels: {in_channels}, hidden_channels: {hidden_channels}, out_channels: {out_channels}, num_layers: {num_layers}')
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, scale=None, zero_point=None):    
        for i, lin in enumerate(self.lins[:-1]):
            # print(f'x.shape: {x.shape}')
            # print(f'lin: {lin}')
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout,bns=True):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
                self.bns.append(nn.BatchNorm1d(hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.norm=bns
        self.reset_parameters()
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers -1: 
                if self.norm:
                    x = self.dropout(self.prelu(self.bns[layer_id](x)))
                else:
                    x = self.dropout(self.prelu(x))
        return x


class HOGA(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, input_dim, emb_dim, heads,out_channels, dropout=0.0, attn_drop=0.0,in_drop=0.0, training_hops=0, mlplayers=0, use_post_residul=0, mlp_hidden=128, mlp_dropout=0.5):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''
        super(HOGA, self).__init__()
        self.num_layers = num_layer
        heads = heads
        in_channels = input_dim
        hidden_channels = emb_dim
        self.in_drop = in_drop
        self.attn_drop = attn_drop
        self.dropout = dropout
        self.mlp_hidden = mlp_hidden
        self.mlp_dropout = mlp_dropout
        self.num_mlplayers = mlplayers
        self.use_post_residul = use_post_residul
        self.lins = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()
        self.mlps = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.attns = torch.nn.ModuleList()
        
        if self.num_mlplayers==0:
            for _ in range(training_hops+1):
                self.lins.append(torch.nn.Linear(in_channels, hidden_channels, bias=False))
        else:
            for _ in range(training_hops+1):
                self.mlps.append(FeedForwardNet(in_channels, self.mlp_hidden, hidden_channels, self.num_mlplayers, self.mlp_dropout, False))
               
        for _ in range(self.num_layers):
            self.attns.append(MultiheadAtt(hidden_channels, heads, dropout=self.attn_drop))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.gates.append(torch.nn.Linear(hidden_channels, hidden_channels, bias=False))
        
        self.last_gate = torch.nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.last_attn = MultiheadAtt(hidden_channels, heads, dropout=self.attn_drop) 

        if self.use_post_residul==1:
            self.res_fc = torch.nn.Linear(in_channels, hidden_channels, bias=False)  
            self.prelu = nn.PReLU()      
           
        self.final_layer = torch.nn.Linear(hidden_channels, out_channels, bias=False)
        self.output_norm = torch.nn.LayerNorm(hidden_channels)
        

    def reset_parameters(self):
        for lins in self.lins:
            lins.reset_parameters()
        for gates in self.gates:
            gates.reset_parameters()
        for mlp in self.mlps:
            mlp.reset_parameters()
        for attn in self.attns:
            attn.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        self.last_gate.reset_parameters()
        self.last_attn.reset_parameters()
        if self.use_post_residul==1:
            self.res_fc.reset_parameters()
        self.output_norm.reset_parameters()
        self.final_layer.reset_parameters()
        

    def forward(self, x):
        num_hops = x.shape[1]
        residual = x[:,0,:]
        x = F.dropout(x, p=self.in_drop, training=self.training)
        if self.num_mlplayers == 0: 
            out_feat = []
            for i in range(num_hops):  
                x_i = self.lins[i](x[:,i,:])
                out_feat.append(x_i)
            x = torch.stack(out_feat, dim=1)
        else:
            out_feat = []
            for i in range(num_hops):  
                x_i = self.mlps[i](x[:,i,:])
                out_feat.append(x_i)
            x = torch.stack(out_feat, dim=1)

        x = F.dropout(x, p=self.dropout, training=self.training)
            
        for i, attn_layer in enumerate(self.attns):
            x_shift = x[:,:-1,:]
            h = self.gates[i](x_shift)  
            h = F.sigmoid(h)        
            x = attn_layer(x, x, x)[0]
            x = torch.cat((x[:,0,:].unsqueeze(1), h * x[:,1:,:] + (1-h) * x_shift), dim=1)
            x = self.lns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)   

        h_last = self.last_gate(x[:,0,:])
        x = self.last_attn(x[:,0,:], x, x)[0]
        x = x.squeeze()
        h_last = F.sigmoid(h_last)
        x = h_last * x
                       
        if self.use_post_residul==1:
            x = x + self.res_fc(residual)
            x = self.prelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output_norm(x)
        x = self.final_layer(x).squeeze()
        
        return x
