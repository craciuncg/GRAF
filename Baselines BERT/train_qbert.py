from transformers import AutoModel, AutoTokenizer, AutoModel, get_scheduler
import torch
import utils
from torch.utils.data import DataLoader
from datasets import load_dataset
import transformers
from torch.utils.data import Sampler
from collections import defaultdict
from datasets import DatasetDict
import sys
import os

VARIANT_NAME = sys.argv[1]
VARIANT_ID = sys.argv[2]

OUTPUT_DIR = VARIANT_NAME + '/' + VARIANT_ID + '/'
TRAIN_SET = 'train_law.csv'
VALIDATION_SET = 'validation_law.csv'

os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(VARIANT_ID)

rows_train = utils.load_csv('train_law.csv')
rows_train = [row for row in rows_train if len(row[-1]) == 1]

rows_validation = utils.load_csv('validation_law.csv')
rows_validation = [row for row in rows_validation if len(row[-1]) == 1]

import re

def remove_number_from_string(input_string):
    result = re.sub(r'^\d+\s*', '', input_string)
    return result

train_set = load_dataset('csv', data_files=TRAIN_SET)['train']
validation_set = load_dataset('csv', data_files=VALIDATION_SET)['train']

train_set = train_set.remove_columns(column_names=['choice_index','context','bert_input','prompt'])
validation_set = validation_set.remove_columns(column_names=['choice_index','context','bert_input','prompt'])

train_set = train_set.rename_column(original_column_name='question_index', new_column_name='index')
validation_set = validation_set.rename_column(original_column_name='question_index', new_column_name='index')

def tokenize(samples):
    index = samples['index']
    question = samples['question'].strip()
    choice = samples['choice'].strip()
    label = samples['label'].strip()
    
    letter = choice[0]
    q = remove_number_from_string(question)[1:].strip().lower()
    c = choice[2:].strip().lower()
    
    tokenized_samples = {}
    tokenized_input = tokenizer(q + '[SEP]' + c, padding=False, truncation=False, add_special_tokens=True)
    
    for k,v in tokenized_input.items():
        tokenized_samples[k] = v
    
    tokenized_samples['index'] = index
    tokenized_samples['question'] = q
    tokenized_samples['choice'] = c
    
    if letter in label:
        tokenized_samples['label'] = [1]
    else:
        tokenized_samples['label'] = [0]
    
    return tokenized_samples

train_encoded = train_set.map(tokenize, batched=False, remove_columns=['question', 'choice'])
validation_encoded = validation_set.map(tokenize, batched=False, remove_columns=['question', 'choice'])

class GroupedByIndexSampler(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle

        # Group indices by the "index" column value
        self.index_groups = defaultdict(list)
        for idx, item in enumerate(data_source):
            self.index_groups[item['index']].append(idx)

        # Convert the defaultdict to a list of index groups
        self.groups = list(self.index_groups.values())

    def __iter__(self):
        # Shuffle the groups if you want (optional)
        if not self.shuffle:
            torch.manual_seed(0)  # For reproducibility
        indices = torch.randperm(len(self.groups)).tolist()
        for i in indices:
            yield self.groups[i]

    def __len__(self):
        return len(self.groups)
    
def collate_fn(samples):
    max_length_input = max([len(sample['input_ids']) for sample in samples])
    
    for i, sample in enumerate(samples):
        input_ids = sample['input_ids']
        token_type_ids = sample['token_type_ids']
        attention_mask = sample['attention_mask']
        
        input_ids = input_ids + [0] * (max_length_input - len(input_ids))
        token_type_ids = token_type_ids + [0] * (max_length_input - len(token_type_ids))
        attention_mask = attention_mask + [0] * (max_length_input - len(attention_mask))
        
        samples[i]['input_ids'] = input_ids
        samples[i]['token_type_ids'] = token_type_ids
        samples[i]['attention_mask'] = attention_mask
    
    collated_samples = {
        'input_ids': [],
        'token_type_ids': [],
        'attention_mask': [],
        'label': []
    }

    for key, l in collated_samples.items():
        for sample in samples:
            l.append(sample[key])
        collated_samples[key] = torch.tensor(collated_samples[key])
    
    return collated_samples

sampler_train = GroupedByIndexSampler(validation_encoded, shuffle=True)
sampler_eval = GroupedByIndexSampler(validation_encoded, shuffle=False)
validation_dataloader = DataLoader(validation_encoded, batch_sampler=sampler_eval, collate_fn=collate_fn, pin_memory=False)
train_dataloader = DataLoader(train_encoded, batch_sampler=sampler_train, collate_fn=collate_fn, pin_memory=False)

def col_batch(batch):
    cbatch = {
        'input': {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': []
        },
        'label': []
    }
    
    cbatch['input']['input_ids'] = batch['input_ids'].int()
    cbatch['input']['attention_mask'] = batch['attention_mask'].int()
    cbatch['input']['token_type_ids'] = batch['token_type_ids'].int()
    
    cbatch['label'] = batch['label'].float()
    
    return cbatch

from qbert import QBERT
import torch.nn as nn

model = QBERT(variant=VARIANT_ID, device='cuda')

EPOCHS = 50
lr = 1e-7
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.BCELoss()

epoch_train_loss = []
epoch_eval_loss = []
step_train_loss = []
step_eval_loss = []

for epoch in range(EPOCHS):
    train_loss = 0
    model.train()
    for batch in train_dataloader:
        cbatch = col_batch(batch)

        for k, v in cbatch['input'].items():
            cbatch['input'][k] = v.cuda()

        cbatch['label'] = cbatch['label'].cuda()
        
        out = model(cbatch['input'])
        optimizer.zero_grad()
        loss = loss_fn(out, cbatch['label'])
        optimizer.step()
        step_train_loss.append(loss.item())
        train_loss += step_train_loss[-1]
        del loss

    train_loss = train_loss / len(train_dataloader)
    epoch_train_loss.append(train_loss)

    eval_loss = 0
    model.eval()
    for batch in validation_dataloader:
        ccbatch = col_batch(batch)

        for k, v in cbatch['input'].items():
            cbatch['input'][k] = v.cuda()

        cbatch['label'] = cbatch['label'].cuda()
        
        with torch.no_grad():
            out = model(cbatch['input'])
            loss = loss_fn(out, cbatch['label'])

        step_eval_loss.append(loss.item())
        eval_loss += step_eval_loss[-1]

    eval_loss = eval_loss / len(validation_dataloader)
    epoch_eval_loss.append(eval_loss)
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'train_loss': train_loss,
            'eval_loss': eval_loss
        },
        OUTPUT_DIR + 'model_' + str(epoch) + '.pth'
    )

import matplotlib.pyplot as plt

plt.plot(epoch_train_loss)
plt.title('Train Epoch Loss ' + VARIANT_NAME + ' ' + VARIANT_ID.split('/')[-1])
plt.tight_layout()
plt.savefig('Train Epoch Loss ' + VARIANT_NAME + ' ' + VARIANT_ID.split('/')[-1] + '.png')
plt.show()

plt.plot(epoch_eval_loss)
plt.title('Eval Epoch Loss ' + VARIANT_NAME + ' ' + VARIANT_ID.split('/')[-1])
plt.tight_layout()
plt.savefig('Eval Epoch Loss ' + VARIANT_NAME + ' ' + VARIANT_ID.split('/')[-1] + '.png')
plt.show()