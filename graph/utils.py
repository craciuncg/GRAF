import csv
import prompts
import prompts_training

def load_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)

        rows = list(reader)
    
    rows = [row for row in rows if row != []]
    return rows

def save_csv(file_path, rows, heading=None, delimiter=','):
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
        
def get_prompt_type_training(model_name: str):
    if 'flan' in model_name.lower():
        return {
            'single' : prompts_training.t5_single_prompt,
            'multiple' : prompts_training.t5_multiple_prompt,
            'logic' : prompts_training.t5_logic_prompt,
            'reading' : prompts_training.t5_reading_prompt,
            'argumentation' : prompts_training.t5_argumentation_prompt
        }
    elif 'mistral' in model_name.lower():
        return {
            'single' : prompts_training.mistral_single_prompt,
            'multiple' : prompts_training.mistral_multiple_prompt,
            'logic' : prompts_training.mistral_logic_prompt,
            'reading' : prompts_training.mistral_reading_prompt,
            'argumentation' : prompts_training.mistral_argumentation_prompt
        }
    elif 'falcon' in model_name.lower():
        return {
            'single' : prompts_training.falcon_single_prompt,
            'multiple' : prompts_training.falcon_multiple_prompt,
            'logic' : prompts_training.falcon_logic_prompt,
            'reading' : prompts_training.falcon_reading_prompt,
            'argumentation' : prompts_training.falcon_argumentation_prompt
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

SPLIT_SIZE = 100
INTER_LEAVE = 25

def split_laws(laws: list[list[str]]):
    splitted = []
    
    for law in laws:
        text = law[4]
        
        text = ' '.join(text.split('\n'))
        words = text.split(' ')
        
        for i in range(int(len(words) / SPLIT_SIZE) + 1):
            if i * SPLIT_SIZE - INTER_LEAVE < 0:
                item = ' '.join(words[i * SPLIT_SIZE : min((i + 1) * SPLIT_SIZE, len(words))])
            else:
                item = ' '.join(words[i * SPLIT_SIZE - INTER_LEAVE : min((i + 1) * SPLIT_SIZE - INTER_LEAVE, len(words))])
            if item == '':
                continue
            splitted.append([law[0], law[1], law[2], law[3], item])
    
    return splitted

def parse_links(links_text: str):
    links = []
    nodes = set()
    
    links_text = links_text.replace('STOP', '')
    
    for link_text in links_text.split('\n'):
        if not '(' in link_text or not ')' in link_text:
            continue
        pieces = link_text.replace('(', '').replace(')', '').split(';')
        pieces = [piece for piece in pieces if piece != '']
        
        if len(pieces) == 3:
            links.append(tuple(pieces))
            
            nodes.add(pieces[0])
            nodes.add(pieces[2])
        # elif len(pieces) == 2:
        #     links.append((pieces[0], '', pieces[1]))
            
        #     nodes.add(pieces[0])
        #     nodes.add(pieces[1])
        elif len(pieces) > 3:
            head = pieces[0]
            rel = pieces[1]
            tail = ' '.join([piece for piece in pieces[2:]])

            links.append((head, rel, tail))
            
            nodes.add(head)
            nodes.add(tail)
        elif len(pieces) == 1:
            nodes.add(pieces[0])
    
    return nodes, links