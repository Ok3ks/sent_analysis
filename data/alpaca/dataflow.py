import json
import os
import logging
from src.paths import DATA_DIR

#from transformers import AutoTokenizer, AutoModel
#from model import chunk,embedding

from bytewax.dataflow import Dataflow
from bytewax.connectors.files import FileOutput
from bytewax.inputs import DynamicInput, StatelessSource
from bytewax.connectors.stdio import StdOutput
from bytewax.outputs import DynamicOutput
from os.path import realpath,dirname
from os.path import join as j
#from websocket import create_connection

import requests
#from transformers import AutoTokenizer, AutoModel
#from qdrant_client import QdrantClient
#from qdrant_client.http.models import Distance, VectorParams
#from qdrant_client.http.api_client import UnexpectedResponse
#from qdrant_client.models import PointStruct
#from database import QdrantClient,QdrantVectorOutput, _QdrantVectorSink
from bytewax.testing import run_main


from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from deploy.demo.sent_gpt3_5_api import EntryText

import json
#import hashlib
from pydantic import BaseModel,Field
from typing import Any, Optional

from unstructured.partition.html import partition_html
from unstructured.cleaners.core import clean, replace_unicode_quotes, clean_non_ascii_chars

#client = QdrantClient(path = "vec_db")
API_KEY = os.environ.get("ALPACA_API_KEY")
API_SECRET = os.environ.get("ALPACA_API_SECRET")

temperature= 0.3
OPENAI_API_KEY= os.environ.get("OPENAI_API_KEY")
MODEL_NAME= os.environ.get("MODEL_NAME")

output_parser = PydanticOutputParser(pydantic_object = EntryText)
format_instructions = output_parser.get_format_instructions()
model = OpenAI(model_name = MODEL_NAME, openai_api_key= OPENAI_API_KEY, temperature = temperature)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class AlpacaSource(StatelessSource):

    """class to process each worker"""
    def __init__(self, tickers):
        #using REST
        self.tickers = tickers
        self.base_url = "https://data.alpaca.markets/v1beta1/news"
        self.parameters = {"symbols":self.tickers[0]}
        self.headers = {'Apca-Api-Key-Id': f"{API_KEY}", 
                                "Apca-Api-Secret-Key": f"{API_SECRET}"}
        self.time_frame = ""
        self.req = requests.get(self.base_url, 
                                params =self.parameters,
                                headers=self.headers)
        #Using websockers
        #self.ws = create_connection("wss://paper-api.alpaca.markets/stream")
        #wss://stream.data.alpaca.markets/v1beta1/news")
        #logger.info(self.webs.status_code)

        #authenticating the websocket
        #self.ws.send(
            #{"action":"auth",
            #json.dumps(
             #"key": f"{API_KEY}",
             #"secret":f"{API_SECRET}"}
            #)
        #)#logger.info(self.ws.recv())

        #self.ws.send(
            #json.dumps(
            #{
               # "action":"subscribe","news":self.tickers}
            #)
        #)

    def next(self):
        print(self.req.status_code)
        out = self.req.json()
        return out['news']
    

class AlpacaNewsInput(DynamicInput):

    """Input class to receive streaming news data
        from the Alpaca real-time news API. Distrubutes number of tickers to workers"""

    def __init__(self, tickers):
        self.tickers = tickers

    def build(self, worker_index, worker_count):
        prods_per_worker = int(len(self.tickers)/worker_count)
        worker_ticker = self.tickers[int(worker_index * prods_per_worker) : int((worker_index * prods_per_worker) + prods_per_worker) ]
        return AlpacaSource(worker_ticker)
    
class Document(BaseModel):
    id: str
    group_key: Optional[str] = None
    metadata: Optional[dict] = {}
    text: Optional[list] = []
    chunks: Optional[list] = []
    embeddings: Optional[list] = []


def parse_article(_data):
    
    #document_id = hashlib.md5(_data['content'].encode()).hexdigest()
    document = Document(id = _data['id'])
    article_elements = partition_html(text=_data['content'])
    
    _data['content'] = clean_non_ascii_chars(replace_unicode_quotes(clean(" ".join([str(x) for x in article_elements]))))
    _data['headline'] = clean_non_ascii_chars(replace_unicode_quotes(clean(_data['headline'])))
    _data['summary'] = clean_non_ascii_chars(replace_unicode_quotes(clean(_data['summary'])))

    document.text = [_data['headline'], _data['summary'], _data['content']]
    document.metadata['headline'] = _data['headline']
    document.metadata['summary'] = _data['summary']
    document.metadata['url'] = _data['url']
    document.metadata['symbols'] = _data['symbols']
    document.metadata['author'] = _data['author']
    document.metadata['created_at'] = _data['created_at']

    #print(document)
    return document

    dirtyblood
    tape 
    proud 

def get_sentiment(text):
    """Obtains the sentiment of a text from pipeline"""
    #print(text)
    sentiment_prompt = PromptTemplate(template="""Assess the sentiment of this user's query \n{comment}. Respond if the sentiment of supplied comment is positive, neutral or negative \n{format_instructions}""",
                                        input_variables=["comment"],
                                        partial_variables= {"format_instructions": format_instructions})
    
    sentiment_chain = LLMChain(llm = model, prompt = sentiment_prompt)
    sentiment = sentiment_chain.run(text[-1])
    print(sentiment)

    return sentiment

def key_value(_data):
    return (_data.id, ";".join(_data.text),)

def output_news(flow:Dataflow, output_path):
    flow.map(key_value)
    flow.output("output", FileOutput(j(output_path, "news"), end = "/n"))
    print(f"File saved to {DATA_DIR}")
    return flow

def output_sentiment(flow:Dataflow, output_path):
    flow.map(key_value)
    flow.map(get_sentiment)
    flow.output("output", FileOutput(j(output_path, "sentiment"), end = "/n"))
    print(f"Sentiment can be found in {output_path}")
    return flow

from datetime import datetime as dt
from datetime import timedelta

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-output',
        choices=["news", "sentiment"], 
        help="use -news to save news, and -sentiment to output sentiments",
        default= "news")

    args = parser.parse_args()
    try:
        os.makedirs(j(dirname(__file__), dt.today().strftime("%d/%m/%y")))
    except OSError:
        os.makedirs(j(dirname(__file__), dt.today().strftime("%d/%m/%y")), exist_ok= True)

    output_path = realpath(j(dirname(__file__), dt.today().strftime("%d/%m/%y")))

    flow = Dataflow()
    flow.input("input", AlpacaNewsInput(tickers = ["BTCUSD", "ETHUSD"]))
    flow.inspect(print)
    flow.flat_map(lambda x:x)
    flow.map(parse_article)

    if args.output == "news":
        flow = output_news(flow, output_path)

    elif args.output == "sentiment":
        flow = output_sentiment(flow, output_path)

    run_main(flow)

