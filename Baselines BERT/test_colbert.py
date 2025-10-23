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

from col_bert import ColBERT

VARIANT_ID = sys.argv[1]
MODEL_PATH = sys.argv[2]

FOLDER = './test_set_law/'

tokenizer = AutoTokenizer.from_pretrained(VARIANT_ID)

model = ColBERT(variant=VARIANT_ID, dropout=0.1)

checkpoint = torch.load(MODEL_PATH, weights_only=False)

model.load_state_dict(checkpoint['model_state_dict'])

def tokenize(samples):
    index = samples['index']
    question = samples['question'].strip()
    choice = samples['choice'].strip()
    label = samples['label'].strip()
    
    letter = choice[0]
    q = remove_number_from_string(question)[1:].strip().lower()
    c = choice[2:].strip().lower()
    
    tokenized_samples = {}
    
    tokenized_question = tokenizer(q, padding=False, truncation=False, add_special_tokens=False)
    tokenized_choice = tokenizer(c, padding=False, truncation=False, add_special_tokens=False)
    
    # tokenized_samples['question_ids'] = tokenized_question['input_ids']
    for k,v in tokenized_question.items():
        tokenized_samples['question_' + k] = v
    
    for k,v in tokenized_choice.items():
        tokenized_samples['choice_' + k] = v
    
    #tokenized_samples['choice_ids'] = tokenized_choice['input_ids']
    
    tokenized_samples['index'] = index
    
    if letter in label:
        tokenized_samples['label'] = 1
    else:
        tokenized_samples['label'] = -1
    
    return tokenized_samples

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
    max_length_questions = max([len(sample['question_input_ids']) for sample in samples])
    max_length_choice = max([len(sample['choice_input_ids']) for sample in samples])
    
    for i, sample in enumerate(samples):
        question_input_ids = sample['question_input_ids']
        question_token_type_ids = sample['question_token_type_ids']
        question_attention_mask = sample['question_attention_mask']
        
        choice_input_ids = sample['choice_input_ids']
        choice_token_type_ids = sample['choice_token_type_ids']
        choice_attention_mask = sample['choice_attention_mask']
        
        question_input_ids = question_input_ids + [0] * (max_length_questions - len(question_input_ids))
        question_token_type_ids = question_token_type_ids + [0] * (max_length_questions - len(question_token_type_ids))
        question_attention_mask = question_attention_mask + [0] * (max_length_questions - len(question_attention_mask))
        
        choice_input_ids = choice_input_ids + [0] * (max_length_choice - len(choice_input_ids))
        choice_token_type_ids = choice_token_type_ids + [0] * (max_length_choice - len(choice_token_type_ids))
        choice_attention_mask = choice_attention_mask + [0] * (max_length_choice - len(choice_attention_mask))
        
        samples[i]['question_input_ids'] = question_input_ids
        samples[i]['question_token_type_ids'] = question_token_type_ids
        samples[i]['question_attention_mask'] = question_attention_mask
        
        samples[i]['choice_input_ids'] = choice_input_ids
        samples[i]['choice_token_type_ids'] = choice_token_type_ids
        samples[i]['choice_attention_mask'] = choice_attention_mask
    
    collated_samples = {
        'question_input_ids': [],
        'question_token_type_ids': [],
        'question_attention_mask': [],
        'choice_input_ids': [],
        'choice_token_type_ids': [],
        'choice_attention_mask': [],
        'label': [],
        'index': []
    }

    for key, l in collated_samples.items():
        for sample in samples:
            l.append(sample[key])
        collated_samples[key] = torch.tensor(collated_samples[key])
    
    return collated_samples

def col_batch(batch):
    cbatch = {
        'question' : {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': []
        },
        'choice' : {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': []
        },
        'label': [],
        'index': []
    }
    
    l = len(batch['label'])
    
    cbatch['question']['input_ids'] = batch['question_input_ids']
    cbatch['question']['attention_mask'] = batch['question_attention_mask']
    cbatch['question']['token_type_ids'] = batch['question_token_type_ids']
    
    cbatch['choice']['input_ids'] = batch['choice_input_ids']
    cbatch['choice']['attention_mask'] = batch['choice_attention_mask']
    cbatch['choice']['token_type_ids'] = batch['choice_token_type_ids']
    
    cbatch['question']['input_ids'] = torch.cat((torch.tensor([[tokenizer.cls_token_id]] * l), cbatch['question']['input_ids'], torch.tensor([[tokenizer.sep_token_id]] * l)), dim=-1).int()
    cbatch['question']['attention_mask'] = torch.cat((torch.tensor([[1]] * l), cbatch['question']['attention_mask'], torch.tensor([[1]] * l)), dim=-1).int()
    cbatch['question']['token_type_ids'] = torch.cat((torch.tensor([[0]] * l), cbatch['question']['token_type_ids'], torch.tensor([[0]] * l)), dim=-1).int()
    
    cbatch['choice']['input_ids'] = torch.cat((torch.tensor([[tokenizer.cls_token_id]] * l), cbatch['choice']['input_ids'], torch.tensor([[tokenizer.sep_token_id]] * l)), dim=-1).int()
    cbatch['choice']['attention_mask'] = torch.cat((torch.tensor([[1]] * l), cbatch['choice']['attention_mask'], torch.tensor([[1]] * l)), dim=-1).int()
    cbatch['choice']['token_type_ids'] = torch.cat((torch.tensor([[0]] * l), cbatch['choice']['token_type_ids'], torch.tensor([[0]] * l)), dim=-1).int()
    
    cbatch['label'] = batch['label'].int()
    cbatch['index'] = batch['index'].int()
    
    return cbatch

import re

def remove_number_from_string(input_string):
    result = re.sub(r'^\d+\s*', '', input_string)
    return result

model.eval()

def test_model(file_name: str):
    print(file_name)
    validation_set = load_dataset('csv', data_files=file_name)['train']
    
    validation_set = validation_set.remove_columns(column_names=['choice_index','context','bert_input','prompt'])

    validation_set = validation_set.rename_column(original_column_name='question_index', new_column_name='index')
    validation_encoded = validation_set.map(tokenize, batched=False, remove_columns=['question', 'choice'])
    
    sampler_eval = GroupedByIndexSampler(validation_encoded, shuffle=False)
    validation_dataloader = DataLoader(validation_encoded, batch_sampler=sampler_eval, collate_fn=collate_fn, pin_memory=False)
    
    correct = 0
    num = 0
    
    for batch in validation_dataloader:
        cbatch = col_batch(batch)

        for k, v in cbatch['question'].items():
            cbatch['question'][k] = v.cuda()

        for k, v in cbatch['choice'].items():
            cbatch['choice'][k] = v.cuda()

        cbatch['label'] = cbatch['label'].cuda()
        with torch.no_grad():
            out, _ = model(cbatch['question'], cbatch['choice'], cbatch['label'])
        
        n = torch.sum(cbatch['label'] == 1).item()
        
        _, labels = torch.topk(cbatch['label'], n)
        _, ans = torch.topk(out, n)
        
        if labels.tolist() == ans.tolist():
            correct += 1
        num += 1
        
        #print(out, cbatch['label'])
    print(correct / num)
    return 0

def test_inm():
    inm_civil_file = FOLDER + 'inm/inm_civil.csv_test.csv'
    inm_penal_file = FOLDER + 'inm/inm_penal.csv_test.csv'
    inm_pr_civila_file = FOLDER + 'inm/inm_pr_civila.csv_test.csv'
    inm_pr_penala_file = FOLDER + 'inm/inm_pr_penala.csv_test.csv'
    
    # prompt_types = utils.get_prompt_type(model_name)
    
    # civil_rows = utils.load_csv(inm_civil_file)
    # penal_rows = utils.load_csv(inm_penal_file)
    # pr_civila_rows = utils.load_csv(inm_pr_civila_file)
    # pr_penala_rows = utils.load_csv(inm_pr_penala_file)
    
    civil_results = test_model(inm_civil_file)
    #utils.save_csv(model_name + '_' + 'INM Drept Civil.csv', civil_results)
    penal_results = test_model(inm_penal_file)
    #utils.save_csv(model_name + '_' + 'INM Drept Penal.csv', penal_results)
    pr_civila_results = test_model(inm_pr_civila_file)
    #utils.save_csv(model_name + '_' + 'INM Drept Procesual Civil.csv', pr_civila_results)
    pr_penala_results = test_model(inm_pr_penala_file)
    #utils.save_csv(model_name + '_' + 'INM Drept Procesual Penal.csv', pr_penala_results)

def test_inppa():
    civil_file = FOLDER + 'inppa/inppa_civil.csv_test.csv'
    penal_file = FOLDER + 'inppa/inppa_penal.csv_test.csv'
    pr_civila_file = FOLDER + 'inppa/inppa_pr_civila.csv_test.csv'
    pr_penala_file = FOLDER + 'inppa/inppa_pr_penala.csv_test.csv'
    oepa_file = FOLDER + 'inppa/inppa_oepa.csv_test.csv'
    
    # prompt_types = utils.get_prompt_type(model_name)
    
    # civil_rows = utils.load_csv(civil_file)
    # penal_rows = utils.load_csv(penal_file)
    # pr_civila_rows = utils.load_csv(pr_civila_file)
    # pr_penala_rows = utils.load_csv(pr_penala_file)
    # oepa_rows = utils.load_csv(oepa_file)
    
    civil_results = test_model(civil_file)
    #utils.save_csv(model_name + '_' + 'INPPA Drept Civil.csv', civil_results)
    penal_results = test_model(penal_file)
    #utils.save_csv(model_name + '_' + 'INPPA Drept Penal.csv', penal_results)
    pr_civila_results = test_model(pr_civila_file)
    #utils.save_csv(model_name + '_' + 'INPPA Drept Procesual Civil.csv', pr_civila_results)
    pr_penala_results = test_model(pr_penala_file)
    #utils.save_csv(model_name + '_' + 'INPPA Drept Procesual Penal.csv', pr_penala_results)
    oepa_results = test_model(oepa_file)
    #utils.save_csv(model_name + '_' + 'INPPA OEPA.csv', oepa_results)

def test_promovare():
    civil_file = FOLDER + 'promovare/inm_civil.csv_test.csv'
    penal_file = FOLDER + 'promovare/inm_penal.csv_test.csv'
    pr_civila_file = FOLDER + 'promovare/inm_pr_civila.csv_test.csv'
    pr_penala_file = FOLDER + 'promovare/inm_pr_penala.csv_test.csv'
    administrativ_file = FOLDER + 'promovare/inm_administrativ.csv_test.csv'
    comercial_file = FOLDER + 'promovare/inm_comercial.csv_test.csv'
    familiei_file = FOLDER + 'promovare/inm_familiei.csv_test.csv'
    international_file = FOLDER + 'promovare/inm_international.csv_test.csv'
    muncii_file = FOLDER + 'promovare/inm_muncii.csv_test.csv'
    
    # prompt_types = utils.get_prompt_type(model_name)
    
    # civil_rows = utils.load_csv(civil_file)
    # penal_rows = utils.load_csv(penal_file)
    # pr_civila_rows = utils.load_csv(pr_civila_file)
    # pr_penala_rows = utils.load_csv(pr_penala_file)
    # administrativ_rows = utils.load_csv(administrativ_file)
    # comercial_rows = utils.load_csv(comercial_file)
    # familiei_rows = utils.load_csv(familiei_file)
    # international_rows = utils.load_csv(international_file)
    # muncii_rows = utils.load_csv(muncii_file)
    
    civil_results = test_model(civil_file)
    #utils.save_csv(model_name + '_' + 'Promovare Drept Civil.csv', civil_results)
    penal_results = test_model(penal_file)
    # #utils.save_csv(model_name + '_' + 'Promovare Drept Penal.csv', penal_results)
    pr_civila_results = test_model(pr_civila_file)
    # #utils.save_csv(model_name + '_' + 'Promovare Drept Procesual Civil.csv', pr_civila_results)
    pr_penala_results = test_model(pr_penala_file)
    # #utils.save_csv(model_name + '_' + 'Promovare Drept Procesual Penal.csv', pr_penala_results)
    administrativ_results = test_model(administrativ_file)
    # #utils.save_csv(model_name + '_' + 'Promovare Drept Administrativ.csv', administrativ_results)
    comercial_results = test_model(comercial_file)
    # #utils.save_csv(model_name + '_' + 'Promovare Drept Comercial.csv', comercial_results)
    familiei_results = test_model(familiei_file)
    # #utils.save_csv(model_name + '_' + 'Promovare Dreptul Familiei.csv', familiei_results)
    international_results = test_model(international_file)
    # #utils.save_csv(model_name + '_' + 'Promovare Drept International.csv', international_results)
    muncii_results = test_model(muncii_file)
    # #utils.save_csv(model_name + '_' + 'Promovare Dreptul Muncii.csv', muncii_results)

test_inm()
test_inppa()
test_promovare()