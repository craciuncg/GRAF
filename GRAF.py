import torch
import torch.nn as nn
from GAT import GAT
from transformers import AutoTokenizer, AutoModel
import utils

class GRAF(nn.Module):
    def __init__(
        self, 
        bert_variant='readerbench/RoBERT-small',
        device='cpu',
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fact_bert = AutoModel.from_pretrained(bert_variant).to(device)
        # self.fact_gat = GAT(
        #     in_features=self.fact_bert.config.hidden_size,
        #     out_features=self.fact_bert.config.hidden_size,
        #     n_heads=6,
        #     device=device
        # )
        
        #self.kg_bert = AutoModel.from_pretrained(bert_variant).to(device)
        # self.kg_gat = GAT(
        #     in_features=self.kg_bert.config.hidden_size,
        #     out_features=self.kg_bert.config.hidden_size,
        #     n_heads=6,
        #     device=device
        # )
        
        self.cos = nn.CosineSimilarity(dim=-1).to(device)
        
        self.question_bert = AutoModel.from_pretrained(bert_variant).to(device)
    
        self.attention_layer = nn.TransformerEncoderLayer(d_model=self.question_bert.config.hidden_size, nhead=2, device=device)
        self.ffn = nn.Sequential(
            nn.Linear(self.question_bert.config.hidden_size, 1),
            nn.Sigmoid()
        ).to(device)
    '''
    {
        nodes: {},
        edges: {},
        mat_node: torch.tensor(),
        mat_edges: torch.tensor()
    }
    '''
    def forward(self, input: dict, fact_graph: dict, kg: dict):
        
        fact_graph_nodes_embeddings = self.fact_bert(**fact_graph['nodes']).last_hidden_state[:, 0]
        fact_graph_edges_embeddings = self.fact_bert(**fact_graph['edges']).last_hidden_state[:, 0]
        
        # fact_graph_embeddings = self.fact_gat(
        #     fact_graph_nodes_embeddings,
        #     fact_graph_edges_embeddings,
        #     fact_graph['mat_nodes'],
        #     fact_graph['mat_edges']
        # )
  
        # kg_nodes_embeddings = self.kg_bert(**kg['nodes']).last_hidden_state[:, 0]
        # kg_edges_embeddings = self.kg_bert(**kg['edges']).last_hidden_state[:, 0]
        
        # kg_embeddings = self.kg_gat(
        #     kg_nodes_embeddings,
        #     kg_edges_embeddings,
        #     kg['mat_nodes'],
        #     kg['mat_edges']
        # )
        
        #h = torch.matmul(self.cos(fact_graph_embeddings.unsqueeze(-2), kg_embeddings), kg_embeddings)
        #h = torch.sum(kg_embeddings, dim=0).unsqueeze(0)
        #h = fact_graph_embeddings
        #h = kg_embeddings
        h = torch.cat((fact_graph_nodes_embeddings, fact_graph_edges_embeddings), dim=0)
        
        question_choice_embedding = self.question_bert(**input).last_hidden_state[:, 0]
        
        h_prime = torch.cat((question_choice_embedding, h), dim=0)
        
        out = self.attention_layer(h_prime)
        
        if out.ndim > 2:
            out = out[:, 0]
        
        out = out[0]
        return self.ffn(out)
    
    