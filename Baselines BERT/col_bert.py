import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModel, AutoConfig
import gc


class ColBERT(nn.Module):
    def __init__(
        self,
        variant='readerbench/jurBERT-base',
        siamese=False,
        similarity=True,
        dropout=0.5,
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        configuration = AutoConfig.from_pretrained(variant)
        configuration.hidden_dropout_prob = dropout
        configuration.attention_probs_dropout_prob = dropout
        
        self.siamese = siamese
        self.tokenizer = AutoTokenizer.from_pretrained(variant)
        self.bert1 = AutoModel.from_pretrained(variant, config=configuration).cuda()
        
        if not siamese:
            self.bert2 = AutoModel.from_pretrained(variant, config=configuration).cuda()
        
        self.cos = nn.CosineSimilarity()
        self.loss_fn = nn.CosineEmbeddingLoss()

    def forward(self, question, choice, label):
        out1 = self.bert1(**question).last_hidden_state[:, 0]
        
        if self.siamese:
            out2 = self.bert1(**choice).last_hidden_state[:, 0]
        else:
            out2 = self.bert2(**choice).last_hidden_state[:, 0]
        
        with torch.no_grad():
            out = self.cos(out1, out2)
        
        loss = self.loss_fn(out1, out2, label)
        
        out1 = out1.detach().cpu()
        out2 = out2.detach().cpu()
        del out1
        del out2
        # for _ in range(1):
        #     gc.collect()
        #     torch.cuda.empty_cache()
        
        return out, loss
        