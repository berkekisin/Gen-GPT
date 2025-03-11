import math
import inspect
from loguru import logger
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import index_to_mask

from g3.transformer_layers import LayerNorm, Block
from g3.utils import get_dense_mask, top_k_top_p_filtering
from g3.transformer_layers import GPTConfig

class GraphEmbedding(nn.Module):
    def __init__(self, embed_map: dict, embed_dim: int):
        super().__init__()
        
        self.node_type_embed = nn.Embedding(embed_map['node_type'], embed_dim)
        self.degree_embed = nn.Embedding(embed_map['degree_type'], embed_dim)
    
    def get_node_type_emb(self, idx):
        return self.node_type_embed(idx)
    
    def forward(self, idx):   
        assert idx[:,:,1].max() <= 4
        node_type_embed =  self.node_type_embed(idx[:,:,0]) 
        #degree_embed =  self.degree_embed(idx[:,:,1]) 
        return node_type_embed #+ degree_embed
    

class GenGPT_Edge(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.num_edge_types = cfg.embed_map['edge_type']
        self.num_node_types = cfg.embed_map['node_type']
        self.num_special_types = cfg.embed_map['special_type']
        self.graph_embed = GraphEmbedding(cfg.embed_map, cfg.embed_dim)
        self.pos_embed = nn.Embedding(cfg.block_size, cfg.embed_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        
        self.transformer_blocks = nn.ModuleList(
            [
                Block(cfg, cfg.embed_dim, cfg.num_heads, cfg.dropout, cfg.bias, edge_attn=True)
                for _ in range(cfg.num_layers)
            ]
        )
        
        self.ln = LayerNorm(cfg.embed_dim, bias=cfg.bias)

       # classifiers for predictions
        self.num_node_target = cfg.target_sizes['node']
        self.node_type_classifier = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=cfg.bias),
            nn.GELU(),
            LayerNorm(cfg.embed_dim, bias=cfg.bias),
            nn.Linear(cfg.embed_dim, self.num_node_target, bias=False),
        )
        self.num_special_target = cfg.target_sizes['special']
        self.special_type_classifier = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=cfg.bias),
            nn.GELU(),
            LayerNorm(cfg.embed_dim, bias=cfg.bias),
            nn.Linear(cfg.embed_dim, self.num_special_target, bias=False),
        )
        
        self.num_edge_target = cfg.target_sizes['edge']
        self.edge_type_classifier = nn.Sequential(
            nn.Linear(3*cfg.embed_dim, cfg.embed_dim, bias=cfg.bias),
            nn.GELU(),
            LayerNorm(cfg.embed_dim, bias=cfg.bias),
            nn.Linear(cfg.embed_dim, self.num_edge_target, bias=False),
        )
        logger.info(f"Embedding sizes: node:{cfg.embed_map['node_type']}, edge:{cfg.embed_map['edge_type']}, degree:{cfg.embed_map['degree_type']}")
        logger.info(f"target_sizes: atom:{self.num_node_target}, edge:{self.num_edge_target}, special:{self.num_special_target}")
        

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.num_layers))
                
        # register buffers
        pred_mask = torch.ones(cfg.block_size-1, cfg.block_size-1, dtype=torch.bool).tril(diagonal=0)
        row_indices = torch.arange(cfg.block_size-1).view(1,-1,1)
        zero_tensor = torch.tensor([0])
        self.register_buffer('pred_mask_train', pred_mask)
        self.register_buffer('row_indices_train', row_indices)
        self.register_buffer('zero_tensor', zero_tensor)


    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos_embed.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_idx(self, idx, bias_matrix):
        token_emb = self.graph_embed(idx)

        x = self.dropout(token_emb)
        for block in self.transformer_blocks:
            x = block(x, bias_matrix)
        return self.ln(x)
    
    def get_logits(self, tokens, num_nodes_batch, bias_matrix):
        # pass through transformer
        x = self.forward_idx(tokens, bias_matrix)
        x_mask = get_dense_mask(x, num_nodes_batch)
        real_x = x[x_mask]
        
        # calculate edge logits
        # construct input: [x_v, x_w, node_type_emb] JANS ADVICE
        # x_v is the emb of the node that is used for node_type prediction of next node which we also predict the connectivity
        # x_w is the emb of the node that the connectivity is predicted its like this [1, 1,2, 1,2,3... ]
        # node_type is the node type of the node that is generated
        batch_dim = x.shape[0]
        arange_tensor = torch.arange(1,num_nodes_batch.max(), device=x.device).repeat(batch_dim).view(batch_dim,-1)
        arange_mask = arange_tensor < num_nodes_batch.unsqueeze(-1)
        edge_repeat = arange_tensor[arange_mask] 
        #to get the proper mask we have to calculate (num_nodes[0] -= 1) like in forward but we can also use [1:]
        # because of that we also do size=real_x.shape[0]+1
        remove_last_elem_mask = index_to_mask(torch.cumsum(num_nodes_batch,dim=0), size=real_x.shape[0]+1)[1:] == False
        edge_x = real_x[remove_last_elem_mask] # remove last element in each graph since its not used for predicting node type
        x_v = torch.repeat_interleave(edge_x, edge_repeat, dim=0)
        assert x_v.shape[0] == (num_nodes_batch * (num_nodes_batch-1)).sum()/2
        
        node_type_emb = self.graph_embed.get_node_type_emb(tokens[:,1:,0])[x_mask[:,1:]]
        node_type_emb = torch.repeat_interleave(node_type_emb, edge_repeat, dim=0)
        assert node_type_emb.shape[0] == (num_nodes_batch * (num_nodes_batch-1)).sum()/2
        
        row_limits = num_nodes_batch.view(-1,1,1)
        row_mask = self.row_indices_train < (row_limits - 1)
        final_mask = self.pred_mask_train.unsqueeze(0) * row_mask
        x_w = x[:,:-1]
        x_w = x_w.unsqueeze(dim=1).expand(-1, x_w.shape[1],x_w.shape[1], -1)
        x_w = x_w[final_mask]
        assert x_w.shape[0] == (num_nodes_batch * (num_nodes_batch-1)).sum()/2
        
        x_edge = torch.cat((x_v, x_w, node_type_emb), dim=-1)
        logits_edge_type = self.edge_type_classifier(x_edge)
        
        #calculate node_type logits
        logits_node_type = self.node_type_classifier(real_x)
        
        # calculate special type logits
        logits_special_token = self.special_type_classifier(real_x)
        
        return {'edge': logits_edge_type, 'node': logits_node_type, 'special':logits_special_token}, x_mask

    def forward(
        self,
        tokens,
        num_nodes_batch,
        bias_matrix,
        special_target,
        edge_type_target
    ):
        # get the logits (tokens shape: batch_size, max_seq_length, [node_type, special_token] ) 
        # bias_matrix: (batch_size, max_seq_length, max_seq_length)
        logits_dict, mask= self.get_logits(tokens, num_nodes_batch, bias_matrix)
      
        # calculate loss
        mt_loss = {}
        
        mt_loss['edge_type']  = F.cross_entropy(input=logits_dict['edge'], target=edge_type_target)
        
        # we have to mask the node logits since the last node of each graph does not predict node type
        # each node predicts the node type of next node to ensure no information leakage
        cum_num_nodes = logits_dict['node'].shape[0]
        graph_start_index = torch.cat((self.zero_tensor,num_nodes_batch[:-1]),dim=0)
        target_mask = index_to_mask(graph_start_index.cumsum(dim=0), size=cum_num_nodes) == False
        num_nodes_batch[0] -= 1
        node_logit_mask = index_to_mask(num_nodes_batch.cumsum(dim=0), size=cum_num_nodes) == False
        assert cum_num_nodes - num_nodes_batch.shape[0] == target_mask.sum() == node_logit_mask.sum()
        mt_loss['node_type'] = F.cross_entropy(input=logits_dict['node'][node_logit_mask],target=tokens[mask][target_mask,0])
        
        # each node predicts whether to continue or stop for the next node ensuring no information leakage
        mt_loss['special_type'] = F.cross_entropy(input=logits_dict['special'], target=special_target)
        
        #calculate acc
        edge_pred = torch.argmax(logits_dict['edge'], dim=-1)
        edge_type_acc = (edge_pred == edge_type_target).sum() / edge_type_target.shape[0]
        node_type_acc = (torch.argmax(logits_dict['node'][node_logit_mask], dim=-1) == tokens[mask][target_mask,0]).sum() / tokens[mask][target_mask,0].shape[0]
        special_type_acc = (torch.argmax(logits_dict['special'], dim=-1) == special_target).sum() / special_target.shape[0]
        acc = {'node_type':node_type_acc,
               'edge_type': edge_type_acc,
               'special_type': special_type_acc
            }
        
        #calculate f1
        macro_f1_edge = f1_score(edge_pred.cpu().numpy(), edge_type_target.cpu().numpy(), average='macro')
        
        return mt_loss, acc, macro_f1_edge


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        logger.info(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        logger.info(f"using fused AdamW: {use_fused}")

        return optimizer
    
    # ["H", "C", "N", "O", "F"]
    # special: 0 stop, 1 continue
    @torch.no_grad()
    def generate(self, num_data, num_steps, top_k, top_p=0.0):
        node_token = torch.randint(0, self.num_node_target, size=(num_data,1), device=self.cfg.device)
        degree_token = torch.zeros(size=(num_data,1), device=self.cfg.device)
        sequence = torch.cat((node_token, degree_token), dim=-1).view(num_data,1,-1).to(torch.int)
        adj_matrix = torch.zeros(size=(num_data, num_steps+1, num_steps+1), device=self.cfg.device).to(torch.int)
        adj_matrix.diagonal(dim1=-2, dim2=-1).fill_(self.num_edge_types - 1)
        special_token = torch.ones(size=(num_data,1), device=self.cfg.device)
        
        for i in range(num_steps):
            x = self.forward_idx(sequence,  adj_matrix[:,:i+1, :i+1])
            
            # generate special token
            logits_special_token = self.special_type_classifier(x[:,-1,:])
            special_token_next = torch.argmax(logits_special_token, dim=-1).unsqueeze(dim=1)
            special_token = torch.cat((special_token, special_token_next), dim=-1)
            
            # generate next node            
            logits_node_type = self.node_type_classifier(x[:,-1, :])
            logits_node_type = top_k_top_p_filtering(logits_node_type, top_k=top_k.node, top_p=top_p)
            node_type_probs = F.softmax(logits_node_type, dim=-1)
            node_type_next = torch.multinomial(node_type_probs, num_samples=1)
            
            #generate edges [x_v, x_w, node_type_emb]
            x_v = x[:, -1, :].unsqueeze(1).expand(x.shape[0], sequence.shape[1], x.shape[-1])
            node_type_emb = self.graph_embed.get_node_type_emb(node_type_next)
            node_type_emb = node_type_emb.expand(x.shape[0],sequence.shape[1],x.shape[-1])
            x_edge = torch.cat((x_v, x, node_type_emb), dim=-1)
            
            logits_edge_type = self.edge_type_classifier(x_edge)
            logits_edge_type = top_k_top_p_filtering(logits_edge_type, top_k=top_k.edge, top_p=top_p)
            edge_type_probs = F.softmax(logits_edge_type.view(-1, logits_edge_type.shape[-1]), dim=-1)
            edge_types_next = torch.multinomial(edge_type_probs, num_samples=1)
            
            # update the graphs
            adj_matrix[:,i+1, :i+1] = edge_types_next.view(adj_matrix.shape[0],-1).to(torch.int)
            adj_matrix = torch.tril(adj_matrix, diagonal=-1) + torch.tril(adj_matrix, diagonal=-1).transpose(1,2) 
            adj_matrix.diagonal(dim1=-2, dim2=-1).fill_(self.num_edge_types - 1)
            degree_next = adj_matrix[:,i+1, :i+1].sum(dim=-1).unsqueeze(dim=1)
            degree_next = torch.clamp(degree_next, max=4) # degree cant be bigger then 4 carbon is 4 
            token_next = torch.cat((node_type_next, degree_next), dim=-1)
            sequence = torch.cat((sequence, token_next.unsqueeze(dim=1)), dim=1)
            if (special_token[:,-1] == 0).all():
                break
        
        #make sure to set last token to stop
        special_token[:,-1] = 0
        
        return sequence, adj_matrix, special_token
       
       
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.cfg
        L, H, Q, T = (
            cfg.num_layers,
            cfg.num_heads,
            cfg.embed_dim // cfg.num_heads,
            cfg.block_size,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
    

if __name__ == "__main__":
  
    zero = torch.tensor([0])
    a = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    num_nodes = torch.tensor([2,4,5,4])
    target_mask = index_to_mask(torch.cat((zero,num_nodes[:-1]),dim=0).cumsum(dim=0), size=a.shape[0])
    

    num_nodes[0] -= 1
    num_nodes = num_nodes.cumsum(dim=0)
    mask = index_to_mask(num_nodes) != True
    
    print(a[mask])
    print(a[target_mask])
    
   

   

   


    
    
