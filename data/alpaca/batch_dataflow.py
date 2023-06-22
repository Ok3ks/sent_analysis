from unstructured.partition.html import partition_html
from unstructured.cleaners.core import clean, replace_unicode_quotes, clean_non_ascii_chars
import os
from os.path import join as j
from os.path import realpath,dirname

import datetime
import json
import requests
from typing import Optional
from pydantic import BaseModel

API_KEY = os.environ.get("ALPACA_API_KEY")
API_SECRET = os.environ.get("ALPACA_API_SECRET")

try:
    os.makedirs(j(dirname(__file__), 'training_data'))
    #modify to save with date
except OSError:
    os.makedirs(j(dirname(__file__), 'training_data'), exist_ok=True)

output_path = realpath(j(dirname(__file__), 'training_data'))

class Document(BaseModel):
    id: str
    group_key: Optional[str] = None
    metadata: Optional[dict] = {}
    text: Optional[list] = []
    chunks: Optional[list] = []
    embeddings: Optional[list] = []

class AlpacaRest():
    """class to process REST"""

    def __init__(self,tickers):
        # using REST

        #Add start and end date as parameters
        self.news = []
        self.tickers = tickers
        self.base_url = "https://data.alpaca.markets/v1beta1/news"
        self.parameters = {"symbols": ",".join(self.tickers),

                            "start": datetime.date(2023, 1, 1).isoformat(),
                            "end": datetime.date(2023, 6, 1).isoformat(),
                            "limit": 50,
                           "exclude_contentless": True,
                           "include_content": True}
        self.headers = {'Apca-Api-Key-Id': f"{API_KEY}",
                        "Apca-Api-Secret-Key": f"{API_SECRET}"}
        self.req = requests.get(self.base_url,
                                params=self.parameters,
                                headers=self.headers,
                                timeout=5)

    def next(self):
        self.out = self.req.json()
        self.news.extend(self.out['news'])
        self.next_page()
    
    def next_page(self):
        counter = 1

        while self.out['next_page_token'] is not None :
            self.next_page_token = self.out['next_page_token']
            self.parameters.update({"page_token":self.next_page_token})
            self.req = requests.get(self.base_url,
                                params=self.parameters,
                                headers=self.headers,
                                timeout=5)
            self.out = self.req.json()
            self.news.extend(self.out['news'])
            counter+=1
            print(f"There are {counter} pages")
    
            

def parse_article(_data):
    # document_id = hashlib.md5(_data['content'].encode()).hexdigest()
    document = Document(id=_data['id'])
    article_elements = partition_html(text=_data['content'])
    _data['content'] = clean_non_ascii_chars(replace_unicode_quotes(
        clean(" ".join([str(x) for x in article_elements]))))
    _data['headline'] = clean_non_ascii_chars(replace_unicode_quotes(clean(_data['headline'])))
    _data['summary'] = clean_non_ascii_chars(
                replace_unicode_quotes(clean(_data['summary'])))
    document.text = [_data['headline'], _data['summary'], _data['content']]
    document.metadata['headline'] = _data['headline']
    document.metadata['summary'] = _data['summary']
    document.metadata['url'] = _data['url']
    document.metadata['symbols'] = _data['symbols']
    document.metadata['author'] = _data['author']
    document.metadata['updated_at'] = _data['updated_at']
    return document

if __name__ == "__main__":

    #Add support for date entry, assert RFC 3339 format
    #Add support for ticker entry 

    news = AlpacaRest(['ETHUSD, BTCUSD'])
    news.next()

    output = {"id":[], "metadata":[],"text":[]}
    parsed = (parse_article(item) for item in news.news)

    for i in parsed:
        output["id"].append(i.id)
        output["text"].append(i.text[-1]) 
        output["metadata"].append(i.metadata)

    print(f"There are {len(output['id'])} news entries")

    with open(j(output_path, 'news.json'), 'w') as out:
        json.dump(output, out)