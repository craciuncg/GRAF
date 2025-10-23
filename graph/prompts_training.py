## Mistral prompt templates

mistral_single_prompt = lambda context, question, answer_choices, correct: \
f'''<s>
[INST] Răspunde la următoarea întrebare de legalitate din {context.lower()}. Un singur răspuns este corect. Tu vei răspunde doar cu litera răspunsului corect:

{context}
{question}
{answer_choices}

[/INST] {correct} </s>
'''
mistral_multiple_prompt = lambda context, question, answer_choices, correct: \
f'''<s>
[INST] Răspunde la următoarea întrebare de legalitate din {context.lower()}. Cel mult 2 răspunsuri sunt corecte.
Dacă un singur răspuns este corect, vei răspunde doar cu litera răspunsului corect.
Dacă 2 răspunsuri sunt corecte, vei răspunde doar cu literele răspunsurilor corecte:

{context}
{question}
{answer_choices}

[/INST] {correct} </s>
'''
mistral_logic_prompt = lambda context, question, answer_choices, correct: \
f'''<s>
[INST] Răspunde la următoarea întrebare de logică. Un singur răspuns este corect. Tu vei răspunde doar cu litera răspunsului corect:

{context}
{question}
{answer_choices}

[/INST] {correct} </s>
'''
mistral_reading_prompt = lambda context, question, answer_choices, correct: \
f'''<s>
[INST] Răspunde la următoarea întrebare de înțelegere a textului. Un singur răspuns este corect. Tu vei răspunde doar cu litera răspunsului corect:

{context}
{question}
{answer_choices}

[/INST] {correct} </s>
'''
mistral_argumentation_prompt = lambda context, question, answer_choices, correct: \
f'''<s>
[INST] Răspunde la următoarea întrebare de argumentare. Un singur răspuns este corect. Tu vei răspunde doar cu litera răspunsului corect:

{context}
{question}
{answer_choices}

[/INST] {correct} </s>
'''

## T5 prompt template

t5_single_prompt = lambda context, question, answer_choices, answer: \
f'''
Răspunde la următoarea întrebare de legalitate din {context.lower()} cu doar una dintre literele din lista [A, B, C]. Un singur răspuns este corect:

{context}
{question}
{answer_choices}
'''
t5_multiple_prompt = lambda context, question, answer_choices, answer: \
f'''
Răspunde la următoarea întrebare de legalitate din {context.lower()}. Cel mult 2 răspunsuri sunt corecte.
Dacă un singur răspuns este corect, vei răspunde doar cu litera răspunsului corect.
Dacă 2 răspunsuri sunt corecte, vei răspunde doar cu literele răspunsurilor corecte
Răspunde cu doar unul dintre simbolurile din lista [A, B, C, AB, AC, BC]:

{context}
{question}
{answer_choices}
'''
t5_logic_prompt = lambda context, question, answer_choices, answer: \
f'''
Răspunde la următoarea întrebare de logică cu doar una dintre literele din lista [A, B, C, D, E]. Un singur răspuns este corect:

{context}
{question}
{answer_choices}
'''
t5_reading_prompt = lambda context, question, answer_choices, answer: \
f'''
Răspunde la următoarea întrebare de înțelegere a textului cu doar una dintre literele din lista [A, B, C, D, E]. Un singur răspuns este corect:

{context}
{question}
{answer_choices}
'''
t5_argumentation_prompt = lambda context, question, answer_choices, answer: \
f'''
Răspunde la următoarea întrebare de argumentare cu doar una dintre literele din lista [A, B, C, D, E]. Un singur răspuns este corect:

{context}
{question}
{answer_choices}
'''