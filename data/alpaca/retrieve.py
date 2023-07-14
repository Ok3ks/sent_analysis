import json
from pprint import pp

def process_json(filepath):
    """Processes json from supplied filepath . Returns an id to symbols dictionary
    a list of all symbols and an id to text dictionary """
    symbols_dict = {}; id_to_text= {}; all_symbols = []

    with open(filepath, 'r') as ins:
        obj = json.load(ins)

    ids = obj['id']
    metadata = obj['metadata']
    text = obj['text']

    for a,b,c in zip(ids, metadata, text):
        symbols_dict.update({a:b['symbols']})
        all_symbols.extend(b['symbols'])
        id_to_text.update({a:c})

    return symbols_dict, all_symbols, id_to_text

def count_and_order(all_symbols:dict):
    count_symbols = {}
    for i in all_symbols:
        count_symbols[i] = count_symbols.get(i, 0) + 1
        count_symbols = dict({k:v for k, v in sorted(count_symbols.items(), key=lambda x: x[1], reverse=True)})
    return count_symbols

def get_article(ticker:str, count_symbols, symbols_dict):
    """Obtain articles that correspond to a given ticker""" 
    #add extra validation to input, to enforce ticker format
    
    keys = list(count_symbols.keys())
    temp = []
    if ticker in keys:
        print("The symbol most spoken about is {}, has {} articles".format(keys[0],count_symbols[keys[0]]))
        print("The symbol least spoken about is {}, has {} articles".format(keys[-1],count_symbols[keys[-1]]))
        print("Your requested symbol {}, has {} articles".format(ticker, count_symbols[ticker],))
        for temp_id,temp_list_symbols in symbols_dict.items():
            if ticker in temp_list_symbols:
                temp.append(temp_id)
    else:
        print("There's no information about this ticker")
    return temp

if __name__ == "__main__":
    #import argparse
    #add support for text entry of tickers, and filepath of news.json
    symbols_dict, all_symbols, id_to_text = process_json('training_data/recent_news.json')
    count_symbols = count_and_order(all_symbols)

    ticker = 'XRPUSD'
    articles = get_article(ticker, count_symbols, symbols_dict)
    for a in articles:
        pp(id_to_text[a])