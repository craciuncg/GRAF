import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModel, AutoConfig

class QBERT(nn.Module):
    def __init__(
        self,
        variant='readerbench/jurBERT-base',
        device='cpu',
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        configuration = AutoConfig.from_pretrained(variant)
        configuration.hidden_dropout_prob = 0
        configuration.attention_probs_dropout_prob = 0

        self.device = device
        self.bert = AutoModel.from_pretrained(variant, config=configuration)
        self.bert.to(self.device)

        self.embedding_size = self.bert.config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_size, 1),
            nn.Sigmoid()
        )
        self.mlp.to(self.device)

    def forward(self, input):
        out = self.bert(**input).last_hidden_state[:, 0, :]
        out = self.mlp(out)
        return out