import json
import os
import logging
from src.paths import DATA_DIR

from bytewax.dataflow import Dataflow
from bytewax.connectors.files import FileOutput
from bytewax.inputs import DynamicInput, StatelessSource
from bytewax.connectors.stdio import StdOutput
from bytewax.outputs import DynamicOutput
from os.path import realpath, dirname
from os.path import join as j
# from websocket import create_connection

import requests
from bytewax.testing import run_main
import json
# import hashlib
from pydantic import BaseModel, Field
from typing import Any, Optional

from unstructured.partition.html import partition_html
from unstructured.cleaners.core import clean, replace_unicode_quotes, clean_non_ascii_chars

API_KEY = os.environ.get("ALPACA_API_KEY")
API_SECRET = os.environ.get("ALPACA_API_SECRET")

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

try:
    os.makedirs(j(dirname(__file__), 'training_data'))
    # os.makedirs(j(dirname(__file__), dt.today().strftime("%d/%m/%y")))
except OSError:
    os.makedirs(j(dirname(__file__), 'training_data'), exist_ok=True)
    # os.makedirs(j(dirname(__file__), dt.today().strftime("%d/%m/%y")), exist_ok= True)
    # output_path = realpath(j(dirname(__file__), dt.today().strftime("%d/%m/%y")))

output_path = realpath(j(dirname(__file__), 'training_data'))

class AlpacaSource(StatelessSource):

    """class to process each worker"""

    def __init__(self, tickers):
        # using REST
        self.tickers = tickers
        self.base_url = "https://data.alpaca.markets/v1beta1/news"
        self.parameters = {"symbols": ",".join(self.tickers),
                           "exclude_contentless": True,
                           "include_content": True}
        self.headers = {'Apca-Api-Key-Id': f"{API_KEY}",
                        "Apca-Api-Secret-Key": f"{API_SECRET}"}
        self.time_frame = ""
        self.req = requests.get(self.base_url,
                                params=self.parameters,
                                headers=self.headers,
                                timeout=5)

        # Using websockets for streaming data
        # self.ws = create_connection("wss://paper-api.alpaca.markets/stream")
        # wss://stream.data.alpaca.markets/v1beta1/news")
        # logger.info(self.webs.status_code)

        # authenticating the websocket
        # self.ws.send(
        # {"action":"auth",
        # json.dumps(
        # "key": f"{API_KEY}",
        # "secret":f"{API_SECRET}"}
        # )
        # )#logger.info(self.ws.recv())

        # self.ws.send(
        # json.dumps(
        # {
        # "action":"subscribe","news":self.tickers}
        # )
        # )

    def next(self):
        print(self.req.status_code)
        out = self.req.json()
        print(out['news'])
        #raise StopIteration upon discovery of certain errors
        return out['news']


class AlpacaNewsInput(DynamicInput):

    """Input class to receive streaming news data
        from the Alpaca real-time news API. Distrubutes number of tickers to workers"""

    def __init__(self, tickers):
        self.tickers = tickers

    def build(self, worker_index, worker_count):
        prods_per_worker = int(len(self.tickers)/worker_count)
        worker_ticker = self.tickers[int(worker_index * prods_per_worker): int(
            (worker_index * prods_per_worker) + prods_per_worker)]
        return AlpacaSource(worker_ticker)


class Document(BaseModel):
    id: str
    group_key: Optional[str] = None
    metadata: Optional[dict] = {}
    text: Optional[list] = []
    chunks: Optional[list] = []
    embeddings: Optional[list] = []


def parse_article(_data):

    # document_id = hashlib.md5(_data['content'].encode()).hexdigest()
    document = Document(id=_data['id'])
    article_elements = partition_html(text=_data['content'])

    _data['content'] = clean_non_ascii_chars(replace_unicode_quotes(
        clean(" ".join([str(x) for x in article_elements]))))
    _data['headline'] = clean_non_ascii_chars(
        replace_unicode_quotes(clean(_data['headline'])))
    _data['summary'] = clean_non_ascii_chars(
        replace_unicode_quotes(clean(_data['summary'])))

    document.text = [_data['headline'], _data['summary'], _data['content']]
    document.metadata['headline'] = _data['headline']
    document.metadata['summary'] = _data['summary']
    document.metadata['url'] = _data['url']
    document.metadata['symbols'] = _data['symbols']
    document.metadata['author'] = _data['author']
    document.metadata['created_at'] = _data['created_at']
    return document


def key_value(_data):
    return (_data.id, ";".join(_data.text),)


if __name__ == "__main__":

    flow = Dataflow()
    flow.input("input", AlpacaNewsInput(tickers=["ETHUSD"]))
    flow.inspect(print)
    flow.flat_map(lambda x: x)
    flow.map(parse_article)
    flow.map(key_value)
    flow.output("output", FileOutput(j(output_path, "news"), end="/n"))

    run_main(flow)
