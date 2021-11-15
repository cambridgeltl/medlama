# usage
import json
with open(r'./biolama_all_relation_result.json', 'w') as outfile:
    json.dump(result, outfile, indent=4)

The result json file has the following format:

{
    'relation_1':[
        {'reference':[...],
        'system-1':[...],
        ...},
        {'reference':[...],
        'system-1':[...],
        ...},
    ],
    'relation_2':[
        {'reference':[...],
        'system-1':[...],
        ...},
        {'reference':[...],
        'system-1':[...],
        ...},
    ],
    ...
}

A dictionary 19 relations, for each relation is a list. Each entry in the list is a small dictionary,
in that small dictionary the keys are 
['reference', t5-large', 'facebook/bart-large', 't5-base', 'facebook/bart-base', 't5-small', 'razent/SciFive-large-Pubmed_PMC', 'razent/SciFive-base-Pubmed_PMC']
for each key, the value is a list of tokens.