import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModel

class CrossBERT(nn.Module):
    
    def __init__(
        self,
        variant='readerbench/jurBERT-base',
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.tokenizer = AutoTokenizer.from_pretrained(variant)
        self.bert = AutoModel.from_pretrained(variant).cuda()
        
        self.embedding_size = self.bert.config.hidden_size
        
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_size, 1),
            nn.Sigmoid()
        ).cuda()
        
    def forward(self, q, d):
        input = torch.cat((q, d), dim=-1)
        
        out = self.bert(input)
        
        return out