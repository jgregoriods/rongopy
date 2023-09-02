import json
import pandas as pd


def load_data(filepath):
    if filepath[-5:] == '.json':
        with open(filepath) as file:
            data = json.load(file)
    elif filepath[-4:] == '.txt':
        data = pd.read_csv(filepath, header=None).values
    return data
