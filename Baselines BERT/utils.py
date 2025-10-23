import csv
from pathlib import Path
import os
import json
import prompts

def load_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)

        rows = list(reader)
    
    rows = [row for row in rows if row != []]
    return rows

def save_csv(file_path, rows, heading=None):
    delimiter = ','
    with open(file_path, 'w', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        if heading:
            writer.writerow(heading)
        writer.writerows(rows)

def get_prompt_type(model_name: str):
    if 'flan' in model_name.lower():
        return {
            'single' : prompts.t5_single_prompt,
            'multiple' : prompts.t5_multiple_prompt,
            'logic' : prompts.t5_logic_prompt,
            'reading' : prompts.t5_reading_prompt,
            'argumentation' : prompts.t5_argumentation_prompt
        }
    elif 'mistral' in model_name.lower():
        return {
            'single' : prompts.mistral_single_prompt,
            'multiple' : prompts.mistral_multiple_prompt,
            'logic' : prompts.mistral_logic_prompt,
            'reading' : prompts.mistral_reading_prompt,
            'argumentation' : prompts.mistral_argumentation_prompt
        }
    elif 'falcon' in model_name.lower():
        return {
            'single' : prompts.falcon_single_prompt,
            'multiple' : prompts.falcon_multiple_prompt,
            'logic' : prompts.falcon_logic_prompt,
            'reading' : prompts.falcon_reading_prompt,
            'argumentation' : prompts.falcon_argumentation_prompt
        }

def get_choice_prompt(task: str):
    if 'promovare' in task.lower():
        return 'single'
    elif 'logic' in task.lower():
        return 'logic'
    elif 'reading' in task.lower():
        return 'reading'
    elif 'argumentation' in task.lower():
        return 'argumentation'
    return 'multiple'

def map_to_spec(t: str):
    if 'drept administrativ' == t.lower():
        return 'administrativ'
    elif 'drept procesual penal' == t.lower():
        return 'pr penala'
    elif 'drept procesual civil' == t.lower():
        return 'pr civila'
    elif 'drept civil' == t.lower():
        return 'civil'
    elif 'drept penal' == t.lower():
        return 'penal'
    elif 'dreptul familiei' == t.lower():
        return 'familiei'
    elif 'dreptul muncii' == t.lower():
        return 'muncii'
    elif 'drept comercial' == t.lower():
        return 'comercial'
    elif 'drept fiscal' == t.lower():
        return 'fiscal'
    elif 'drept international privat' == t.lower() or 'drept internațional privat' == t.lower():
        return 'international'
    elif 'Organizarea si Exercitarea Profesiei de Avocat' == t:
        return 'oepa'
    return ''

def get_entry_prompt(entry: list):
    context = entry[1]
    if map_to_spec(context) == '':
        return ''
    elif context.isupper():
        return 'single'
    return 'multiple'

SPLIT_SIZE = 50
INTER_LEAVE = 25

def split_laws(laws: list[str]):
    splitted = []
    
    for law in laws:
        text = law
        
        text = ' '.join(text.split('\n'))
        words = text.split(' ')
        
        for i in range(int(len(words) / SPLIT_SIZE) + 1):
            if i * SPLIT_SIZE - INTER_LEAVE < 0:
                item = ' '.join(words[i * SPLIT_SIZE : min((i + 1) * SPLIT_SIZE, len(words))])
            else:
                item = ' '.join(words[i * SPLIT_SIZE - INTER_LEAVE : min((i + 1) * SPLIT_SIZE - INTER_LEAVE, len(words))])
            if item == '':
                continue
            splitted.append(item)
    
    return splitted

import math

def split_list(l: list, parts: int):
    splitted = []
    size = math.ceil(int(len(l) / parts))
    
    for i in range(0, len(l), size):
        splitted.append(l[i:i + size])
    
    return splitted

def get_child_directories(path: str):
    return list(next(os.walk(path))[1])

def load_knowledge():
    csv.field_size_limit(2147483647)
    knowledge_rows = load_csv('law.csv')
    
    KNOWLEDGE = {}
    for k in knowledge_rows:
        year = k[0]
        t = k[1]
        
        if year not in KNOWLEDGE:
            KNOWLEDGE[year] = {}
        
        if t not in KNOWLEDGE[year]:
            KNOWLEDGE[year][t] = []
    
        if k[-1] == '':
            continue
        KNOWLEDGE[year][t].append(k[-1])
    
    return KNOWLEDGE

def load_knowledge_rag_bm25():
    csv.field_size_limit(2147483647)
    KNOWLEDGE_RAG = {}
    FOLDER = './precomputed1/bm25/'

    for year in get_child_directories(FOLDER):
        year_dir = FOLDER + year + '/'
        
        if year not in KNOWLEDGE_RAG:
            KNOWLEDGE_RAG[year] = {}
            
        all_files = list(Path(year_dir).rglob('*'))
        all_files = [str(f) for f in all_files if str(f).endswith('.json')]
        
        for spec_file in all_files:
            spec_name, _ = os.path.splitext(os.path.basename(spec_file))
            
            with open(spec_file, 'r') as f:
                pcorpus = json.load(f)
                KNOWLEDGE_RAG[year][spec_name] = [doc for doc in pcorpus if doc != '']   
    return KNOWLEDGE_RAG