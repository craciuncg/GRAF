import csv
import prompts
import prompts_training
import torch
import re
from pathlib import Path
import json
from rank_bm25 import BM25Okapi

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

# Split choices
def split_choices(choices: str):
    sep_choices = ['', '', '']
    choice_num = 0
    
    for line in choices.split('\n'):
        if line.strip().startswith('A.') and choice_num == 0:
            choice_num += 1
        elif line.strip().startswith('B.') and choice_num == 1:
            choice_num += 1
        elif line.strip().startswith('C.') and choice_num == 2:
            choice_num += 1
        sep_choices[choice_num - 1] += line + '\n'
    
    return sep_choices

def parse_links(links_text: str):
    links = []
    nodes = set()
    if 'Text1' in links_text or 'Text2' in links_text or 'Entitate;Relație;Entitate' in links_text:
        return nodes, links
    
    
    links_text = links_text.replace('STOP', '')
    
    pattern = r"\([^\(\)]*;[^\(\)]*;[^\(\)]*\)|\([^\(\)]*;[^\(\)]*\)"
    matches = re.findall(pattern, links_text)
    
    for link_text in matches:
        if not '(' in link_text or not ')' in link_text:
            continue
        pieces = link_text.replace('(', '').replace(')', '').split(';')
        pieces = [piece for piece in pieces if piece != '']
        
        if len(pieces) == 3:
            head, rel, tail = tuple(pieces)
            links.append((head.strip(), rel.strip(), tail.strip()))
            
            nodes.add(pieces[0].strip())
            nodes.add(pieces[2].strip())
        elif len(pieces) == 2:
            links.append((pieces[0], '', pieces[1]))
            
            nodes.add(pieces[0])
            nodes.add(pieces[1])
        elif len(pieces) > 3:
            head = pieces[0]
            rel = pieces[1]
            tail = ' '.join([piece for piece in pieces[2:]])

            links.append((head.strip(), rel.strip(), tail.strip()))
            
            nodes.add(head.strip())
            nodes.add(tail.strip())
        # elif len(pieces) == 1:
        #     nodes.add(pieces[0].strip())
    
    return nodes, links

def embed(text, tokenizer, jurbert, device):
    tokenized_article = tokenizer(text)
    l = len(tokenized_article['input_ids'])
    
    embedding = []
    
    i = 0
    while i < l:
        art_slice = {}
        for k,v in tokenized_article.items(): art_slice[k] = torch.tensor(v[i:min(i+512, l)]).unsqueeze(0).to(device)
        with torch.no_grad():
            out = jurbert(**art_slice)
            
            out = out.last_hidden_state[0][0].cpu().tolist()
            
        embedding += out
        
        i += 512
    
    return embedding

def get_corpus_type(file: str):
    branch = ''
    type = ''
    
    if 'edge' in file:
        type = 'edge'
    elif 'node' in file:
        type = 'node'
        
    if 'pr civil' in file:
        branch = 'drept procesual civil'
    elif 'pr penal' in file:
        branch = 'drept procesual penal'
    elif 'civil' in file:
        branch = 'drept civil'
    elif 'penal' in file:
        branch = 'drept penal'
    elif 'administrativ' in file:
        branch = 'drept administrativ'
    elif 'comercial' in file:
        branch = 'drept comercial'
    elif 'international' in file:
        branch = 'drept international privat'
    elif 'family' in file:
        branch = 'dreptul familiei'
    elif 'work' in file:
        branch = 'dreptul muncii'
    
    return type, branch

def get_corpus_file(branch: str, folder='./graf/'):
    if 'procesual civil' in branch:
        return folder + 'law_relext_pr civil.csv'
    if 'procesual penal' in branch:
        return folder + 'law_relext_pr penal.csv'
    if 'civil' in branch:
        return folder + 'law_relext_civil.csv'
    if 'penal' in branch:
        return folder + 'law_relext_penal.csv'
    if 'administrativ' in branch:
        return folder + 'law_relext_administrativ.csv'
    if 'comercial' in branch:
        return folder + 'law_relext_comercial.csv'
    if 'familiei' in branch:
        return folder + 'law_relext_family.csv'
    if 'international' in branch:
        return folder + 'law_relext_international.csv'
    if 'munci' in branch:
        return folder + 'law_relext_work.csv'
    
    return ''

import re

def remove_number_from_string(input_string):
    result = re.sub(r'^\d+\s*', '', input_string)
    return result

def get_files_with_extension(folder: str, ext: str):
    all_files = list(Path(folder).rglob('*'))
    all_files = [str(f) for f in all_files if str(f).endswith(ext)]
    
    return all_files

def load_laws_bm25(folder='.', join=True):
    all_files = list(Path(folder).rglob('*'))
    all_files = [str(f) for f in all_files if str(f).endswith('.json')]
    
    DATA = {
        'node': {},
        'edge' : {}
    }
    
    for file in all_files:
        types, branch = get_corpus_type(file)
        
        with open(file, 'r') as f:
            DATA[types][branch] = json.load(f)
            if join:
                for i, text in DATA[types][branch]:
                    DATA[types][branch][i] = ' '.join(text)
            
    return DATA

def reduce_string(s: str, nlp):
    doc = nlp(s)
    
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

def bm25_retrieve(query: str, corpus: list, nlp, topn=10):
    pquery = reduce_string(query, nlp)
    
    bm25 = BM25Okapi(corpus)
    scores = sorted(enumerate(bm25.get_scores(pquery)), key=lambda x : x[1], reverse=True)
    
    indices = [index for index, _ in scores[:topn]]
    
    return indices

def get_KG(file: str, N=100):
    nodes = set()
    links = []

    if N == None:
        laws = load_csv(file)
    else:
        laws = load_csv(file)[:N]

    for law in laws:
        nodes_law, links_law = parse_links(law[5].strip())
        
        links += links_law
        nodes.update(nodes_law)
        
    nodes_list = sorted(list(nodes))
    edges_list = sorted(list(set([link[1] for link in links])))

    nodes_dict = dict([tuple(reversed(p)) for p in list(enumerate(nodes_list))])
    edges_dict = dict([tuple(reversed(p)) for p in list(enumerate(edges_list))])
    
    link_dict_nodes = {}
    link_dict_edges = {}

    for link in links:
        head, relation, tail = link
        if head not in link_dict_nodes:
            link_dict_nodes[head] = set()
        if head not in link_dict_edges:
            link_dict_edges[head] = set()
        if tail not in link_dict_edges:
            link_dict_edges[tail] = set()
        if tail not in link_dict_nodes:
            link_dict_nodes[tail] = set()
        
        link_dict_nodes[head].add(tail)
        link_dict_edges[head].add(relation)
        
        link_dict_nodes[tail].add(head)
        link_dict_edges[tail].add(relation)
        
    topology_nodes = {}
    topology_edges = {}

    for i, node in enumerate(nodes_list):
        if i not in topology_nodes:
            topology_nodes[i] = set()
        if i not in topology_edges:
            topology_edges[i] = set()
        
        for tail in link_dict_nodes[node]:
            topology_nodes[i].add(nodes_dict[tail])
        for edge in link_dict_edges[node]:
            topology_edges[i].add(edges_dict[edge])
    
    mat_nodes = torch.zeros(size=(len(nodes_list), len(nodes_list)))

    for node, neighbs in topology_nodes.items():
        for neighb in neighbs:
            mat_nodes[node][neighb] = 1
            
    mat_edges = torch.zeros(size=(len(nodes_list), len(edges_list)))

    for node, neighbs in topology_edges.items():
        for neighb in neighbs:
            mat_edges[node][neighb] = 1
            
    return nodes_list, edges_list, topology_nodes, topology_edges, mat_nodes, mat_edges

def get_KG_string(graph_str: str):
    nodes = set()
    links = []
    nodes_law, links_law = parse_links(graph_str.strip())
    if nodes_law == set() or links_law == []:
        return [], [], [], [], [], []
    
    links += links_law
    nodes.update(nodes_law)
        
    nodes_list = sorted(list(nodes))
    edges_list = sorted(list(set([link[1] for link in links])))

    nodes_dict = dict([tuple(reversed(p)) for p in list(enumerate(nodes_list))])
    edges_dict = dict([tuple(reversed(p)) for p in list(enumerate(edges_list))])
    
    link_dict_nodes = {}
    link_dict_edges = {}

    for link in links:
        head, relation, tail = link
        if head not in link_dict_nodes:
            link_dict_nodes[head] = set()
        if head not in link_dict_edges:
            link_dict_edges[head] = set()
        if tail not in link_dict_edges:
            link_dict_edges[tail] = set()
        if tail not in link_dict_nodes:
            link_dict_nodes[tail] = set()
        
        link_dict_nodes[head].add(tail)
        link_dict_edges[head].add(relation)
        
        link_dict_nodes[tail].add(head)
        link_dict_edges[tail].add(relation)
        
    topology_nodes = {}
    topology_edges = {}

    for i, node in enumerate(nodes_list):
        if i not in topology_nodes:
            topology_nodes[i] = set()
        if i not in topology_edges:
            topology_edges[i] = set()
        
        for tail in link_dict_nodes[node]:
            topology_nodes[i].add(nodes_dict[tail])
        for edge in link_dict_edges[node]:
            topology_edges[i].add(edges_dict[edge])
    
    mat_nodes = torch.zeros(size=(len(nodes_list), len(nodes_list)))

    for node, neighbs in topology_nodes.items():
        for neighb in neighbs:
            mat_nodes[node][neighb] = 1
            
    mat_edges = torch.zeros(size=(len(nodes_list), len(edges_list)))

    for node, neighbs in topology_edges.items():
        for neighb in neighbs:
            mat_edges[node][neighb] = 1
            
    return nodes_list, edges_list, topology_nodes, topology_edges, mat_nodes, mat_edges

from collections import deque, defaultdict

def remap_graph(nodes_names, edges_names, graph):
    nodes, edges, node_adj_list, edge_adj_list, node_adj_matrix, edge_adj_matrix = graph
    
    new_node_adj_list = {}
    new_edge_adj_list = {}
    
    nodes_ids = {id : i for i, id in enumerate(nodes)}
    edges_ids = {id: i for i, id in enumerate(edges)}
    
    for k, v in node_adj_list.items():
        new_node_adj_list[nodes_ids[k]] = set([nodes_ids[id] for id in v])
    
    for k, v in edge_adj_list.items():
        new_edge_adj_list[nodes_ids[k]] = set([edges_ids[id] for id in v])
        
    new_nodes = []
    new_edges = []
    
    for node in nodes:
        new_nodes.append(nodes_names[node])
    
    for edge in edges:
        new_edges.append(edges_names[edge])
    
    return new_nodes, new_edges, new_node_adj_list, new_edge_adj_list, node_adj_matrix, edge_adj_matrix

def sample_graph(knowledge_graph, node_ids, edge_ids, depth, nodes_limit=10000):
    # Unpack the knowledge graph tuple
    nodes, edges, node_adj_list, edge_adj_list, node_adj_matrix, edge_adj_matrix = knowledge_graph
    
    # Initialize new graph components
    new_nodes = set()
    new_edges = set()
    new_node_adj_list = defaultdict(set)
    new_edge_adj_list = defaultdict(set)
    edge_nodes = set()
    
    # Add initial edges and nodes connected by these edges
    added_nodes = 0
    #new_edges.update(edge_ids)
    for edge in edge_ids:
        for node, connected_edges in edge_adj_list.items():
            if edge in connected_edges:
                if added_nodes < nodes_limit:
                    edge_nodes.add(node)
                    new_edge_adj_list[node].add(edge)
                    new_edges.add(edge)
                    added_nodes += 1

    # Combine the nodes connected by edges with the initially provided node_ids
    initial_nodes = set(node_ids)
    
    for node1 in set(node_ids).union(edge_nodes):
        for node2 in set(node_ids).union(edge_nodes):
            if node2 in node_adj_list[node1]:
                new_node_adj_list[node1].add(node2)
    
    # BFS initialization
    queue = deque([(node, 0) for node in initial_nodes])  # (current_node, current_depth)
    visited_nodes = set(initial_nodes)
    visited_edges = set(edge_ids)
    
    # Add initial nodes to the new graph
    #new_nodes.update(initial_nodes)
    new_nodes = initial_nodes
    
    while queue:
        current_node, current_depth = queue.popleft()
        
        if current_depth >= depth:
            continue
        
        # Explore neighboring nodes and edges
        for neighbor in node_adj_list[current_node]:
            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                queue.append((neighbor, current_depth + 1))
                new_nodes.add(neighbor)
            
            # Add the edge connecting the current node and neighbor
            for edge in edge_adj_list[current_node]:
                if edge in edge_adj_list[neighbor] and edge not in visited_edges:
                    visited_edges.add(edge)
                    new_edges.add(edge)
                    new_edge_adj_list[current_node].add(edge)
                    new_edge_adj_list[neighbor].add(edge)
                    
                    # Explore the connected nodes through this edge
                    queue.append((neighbor, current_depth + 1))
    
    new_nodes = new_nodes.union(edge_nodes)
    
    # Build the adjacency lists for the new graph
    for node in new_nodes:
        new_node_adj_list[node] = node_adj_list[node].intersection(new_nodes)
        new_edge_adj_list[node] = edge_adj_list[node].intersection(new_edges)
    
    # Convert the adjacency lists to adjacency matrices
    node_idx_map = {node: idx for idx, node in enumerate(new_nodes)}
    edge_idx_map = {edge: idx for idx, edge in enumerate(new_edges)}
    
    node_size = len(new_nodes)
    edge_size = len(new_edges)
    
    new_node_adj_matrix = torch.zeros((node_size, node_size), dtype=torch.float32)
    new_edge_adj_matrix = torch.zeros((node_size, edge_size), dtype=torch.float32)
    
    for node in new_nodes:
        node_idx = node_idx_map[node]
        for neighbor in new_node_adj_list[node]:
            neighbor_idx = node_idx_map[neighbor]
            new_node_adj_matrix[node_idx][neighbor_idx] = 1.0
            
        for edge in new_edge_adj_list[node]:
            edge_idx = edge_idx_map[edge]
            new_edge_adj_matrix[node_idx][edge_idx] = 1.0
    
    # Return the new knowledge graph
    return remap_graph(nodes, edges, (
        list(new_nodes), 
        list(new_edges), 
        new_node_adj_list, 
        new_edge_adj_list, 
        new_node_adj_matrix, 
        new_edge_adj_matrix
    ))