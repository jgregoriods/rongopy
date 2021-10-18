import json


with open('tablets/tablets.json', 'r') as file:
    tablets = json.load(file)

with open('tablets/tablets_clean.json', 'r') as file:
    tablets_clean = json.load(file)

with open('tablets/tablets_simple.json', 'r') as file:
    tablets_simple = json.load(file)