## T5 prompt template

t5_single_prompt = lambda context, question, answer_choices: \
f'''
Răspunde la următoarea întrebare de legalitate din {context.lower()} cu doar una dintre literele din lista [A, B, C]. Un singur răspuns este corect:

{context}
{question}
{answer_choices}
'''
t5_multiple_prompt = lambda context, question, answer_choices: \
f'''
Răspunde la următoarea întrebare de legalitate din {context.lower()}. Cel mult 2 răspunsuri sunt corecte.
Dacă un singur răspuns este corect, vei răspunde doar cu litera răspunsului corect.
Dacă 2 răspunsuri sunt corecte, vei răspunde doar cu literele răspunsurilor corecte
Răspunde cu doar unul dintre simbolurile din lista [A, B, C, AB, AC, BC]:

{context}
{question}
{answer_choices}
'''
t5_logic_prompt = lambda context, question, answer_choices: \
f'''
Răspunde la următoarea întrebare de logică cu doar una dintre literele din lista [A, B, C, D, E]. Un singur răspuns este corect:

{context}
{question}
{answer_choices}
'''
t5_reading_prompt = lambda context, question, answer_choices: \
f'''
Răspunde la următoarea întrebare de înțelegere a textului cu doar una dintre literele din lista [A, B, C, D, E]. Un singur răspuns este corect:

{context}
{question}
{answer_choices}
'''
t5_argumentation_prompt = lambda context, question, answer_choices: \
f'''
Răspunde la următoarea întrebare de argumentare cu doar una dintre literele din lista [A, B, C, D, E]. Un singur răspuns este corect:

{context}
{question}
{answer_choices}
'''

## Mistral prompt templates

mistral_single_prompt = lambda context, question, answer_choices: \
f'''
[INST] Răspunde la următoarea întrebare de legalitate din {context.lower()}. Un singur răspuns este corect. Tu vei răspunde doar cu litera răspunsului corect:

{context}
{question}
{answer_choices}

[/INST]
'''
mistral_multiple_prompt = lambda context, question, answer_choices: \
f'''
[INST] Răspunde la următoarea întrebare de legalitate din {context.lower()}. Cel mult 2 răspunsuri sunt corecte.
Dacă un singur răspuns este corect, vei răspunde doar cu litera răspunsului corect.
Dacă 2 răspunsuri sunt corecte, vei răspunde doar cu literele răspunsurilor corecte:

{context}
{question}
{answer_choices}

[/INST]
'''
mistral_logic_prompt = lambda context, question, answer_choices: \
f'''
[INST] Răspunde la următoarea întrebare de logică. Un singur răspuns este corect. Tu vei răspunde doar cu litera răspunsului corect:

{context}
{question}
{answer_choices}

[/INST]
'''
mistral_reading_prompt = lambda context, question, answer_choices: \
f'''
[INST] Răspunde la următoarea întrebare de înțelegere a textului. Un singur răspuns este corect. Tu vei răspunde doar cu litera răspunsului corect:

{context}
{question}
{answer_choices}

[/INST]
'''
mistral_argumentation_prompt = lambda context, question, answer_choices: \
f'''
[INST] Răspunde la următoarea întrebare de argumentare. Un singur răspuns este corect. Tu vei răspunde doar cu litera răspunsului corect:

{context}
{question}
{answer_choices}

[/INST]
'''

## Falcon prompt templates

falcon_single_prompt = lambda context, question, answer_choices: \
f'''
Răspunde la următoarea întrebare de legalitate din {context.lower()} cu doar una dintre literele din lista [A, B, C]. Un singur răspuns este corect:

>>QUESTION<<
{context}
{question}
{answer_choices}
>>ANSWER<<
'''
falcon_multiple_prompt = lambda context, question, answer_choices: \
f'''
Răspunde la următoarea întrebare de legalitate din {context.lower()}. Cel mult 2 răspunsuri sunt corecte.
Dacă un singur răspuns este corect, vei răspunde doar cu litera răspunsului corect.
Dacă 2 răspunsuri sunt corecte, vei răspunde doar cu literele răspunsurilor corecte
Răspunde cu doar unul dintre simbolurile din lista [A, B, C, AB, AC, BC]:

>>QUESTION<<
{context}
{question}
{answer_choices}
>>ANSWER<<
'''
falcon_logic_prompt = lambda context, question, answer_choices: \
f'''
Răspunde la următoarea întrebare de logică cu doar una dintre literele din lista [A, B, C, D, E]. Un singur răspuns este corect:

>>QUESTION<<
{context}
{question}
{answer_choices}
>>ANSWER<<
'''
falcon_reading_prompt = lambda context, question, answer_choices: \
f'''
Răspunde la următoarea întrebare de înțelegere a textului cu doar una dintre literele din lista [A, B, C, D, E]. Un singur răspuns este corect:

>>QUESTION<<
{context}
{question}
{answer_choices}
>>ANSWER<<
'''
falcon_argumentation_prompt = lambda context, question, answer_choices: \
f'''
Răspunde la următoarea întrebare de argumentare cu doar una dintre literele din lista [A, B, C, D, E]. Un singur răspuns este corect:

>>QUESTION<<
{context}
{question}
{answer_choices}
>>ANSWER<<
'''