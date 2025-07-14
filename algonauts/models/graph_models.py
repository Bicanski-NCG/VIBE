import torch
import torch.nn as nn
from torch_geometric.nn.models import GAT


class FixedNetworkGNN(nn.Module):


    def __init__(self,adjacency_matrix):

        super().__init__()
        edge_index,edge_attr = self.preprocess_adjacency(adjacency_matrix)

        self.register_buffer('edge_index',edge_index)

        self.register_buffer('edge_attr',edge_attr)

    
    def preprocess_adjacency(self,adjacency_matrix):
           
        edge_index = []

        edge_attr = []
        N = adjacency_matrix.shape[0]
        for i in range(N):
            for j in range(N):
                if i ==j:
                    pass
                else:
                    if adjacency_matrix[i,j]>0.0:
                        edge_index.append([i,j])

                        edge_attr.append(adjacency_matrix[i,j])

        return torch.tensor(edge_index).t().contiguous(),torch.tensor(edge_attr).unsqueeze(-1)


class FixedNetworkGraphAttention(FixedNetworkGNN):





    def __init__(self,dim,adjacency_matrix,num_layers = 5):
           
        super().__init__(adjacency_matrix)


        


        self.graph_attention_net = GAT(in_channels= dim,
                                        hidden_channels= dim,
                                        out_channels = dim,
                                        num_layers=  num_layers,
                                        dropout = 0.3, act = nn.GELU())

       

        self.norm = nn.LayerNorm((dim,adjacency_matrix.shape[-1]))

       

    def forward(self,x):

        x= self.norm(x)

        
        # B-batches, S - timepoints, D - nodes. 
        B,S,D = x.shape

        # We want to treat each timeseries as a node-feature vector
        x_flat = x.view(B * D, S)

        # this would be needed for batch_norm or things like that
        #batch_tensor = torch.arange(B, device=x.device).repeat_interleave(D)
        
        # create one big adjacency matrix (each batch becomes a separate disconnected component of this graph, which are all identical copies)
        edge_indices_list = []
        edge_attributes_list = []
        for i in range(B):
           
            offset_edge_index = self.edge_index + (i * D)
            edge_indices_list.append(offset_edge_index)
            edge_attributes_list.append(self.edge_attr)
        
        # Concatenate all edge indices and edge attributes
        batched_edge_index = torch.cat(edge_indices_list, dim=1)
        batched_edge_attr = torch.cat(edge_attributes_list, dim=0) # Concatenate along the edge dimension (dim=0)

        
        x = self.graph_attention_net(x_flat,batched_edge_index,batched_edge_attr)
        # --- GNN Integration Ends Here ---

        # The output of GNN will be (B * D, gnn_out_channels)
        out = x.view(B, D, -1)

        # want shape (B,S,D in the end)
        out = out.transpose(-1,-2)


        return out








  
class FixedNetworkGraphAttention1D(FixedNetworkGNN):





    def __init__(self,dim,adjacency_matrix,num_layers = 1,hidden = 3):
           
        super().__init__(adjacency_matrix)


        S =dim
        self.graph_attention_net = GAT(in_channels= 1,
                                        hidden_channels= hidden,
                                        out_channels = 1,
                                        num_layers=  num_layers,
                                        dropout = 0.5, act = nn.GELU())

     
        self.norm = nn.LayerNorm((S,adjacency_matrix.shape[-1],))

       

    def forward(self,x):

        x= self.norm(x)

        
        # B-batches, S - timepoints, D - nodes. 
        B,S,D = x.shape

        # want to do 1-d convolution
        x_flat = x.view(B * S*D,1)

        # this would be needed for batch_norm or things like that
        #batch_tensor = torch.arange(B, device=x.device).repeat_interleave(D)
        
        # create one big adjacency matrix (each batch becomes a separate disconnected component of this graph, which are all identical copies)
        edge_indices_list = []
        edge_attributes_list = []
        for i in range(B*S):
           
            offset_edge_index = self.edge_index + (i * D)
            edge_indices_list.append(offset_edge_index)
            edge_attributes_list.append(self.edge_attr)
        
        # Concatenate all edge indices and edge attributes
        batched_edge_index = torch.cat(edge_indices_list, dim=1)
        batched_edge_attr = torch.cat(edge_attributes_list, dim=0) # Concatenate along the edge dimension (dim=0)

        
        x = self.graph_attention_net(x_flat,batched_edge_index,batched_edge_attr)
        # --- GNN Integration Ends Here ---

        # The output of GNN will be (B*S*D,1)
        out = x.view(B, S,D)



        return out



    def preprocess_adjacency(self,adjacency_matrix):
           
        edge_index = []

        edge_attr = []
        N = adjacency_matrix.shape[0]
        for i in range(N):
            for j in range(N):
                if i ==j:
                    pass
                else:
                    if adjacency_matrix[i,j]>0.0:
                        edge_index.append([i,j])

                        edge_attr.append(adjacency_matrix[i,j])

        return torch.tensor(edge_index).t().contiguous(),torch.tensor(edge_attr).unsqueeze(-1)





def pad_to_fixed_length(input_tensor, fixed_length=600, pad_value=0):
    _, S,_ = input_tensor.shape

    if S > fixed_length:
        raise ValueError(f"Input length ({S}) is greater than the fixed length ({fixed_length}).")

    # Calculate padding needed
    padding_needed = fixed_length - S
    
    padded_tensor = F.pad(input_tensor, (0,0, padding_needed,0), 'constant', pad_value)

    return padded_tensor,padding_needed


