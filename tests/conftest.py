from os.path import join
import pytest
import requests
import json

from src.utils import DATA_DIR


@pytest.fixture
def pred_fixture():
    x = [1,0,1,0,0,0,1,0,1,0,0,1,0]
    return x

@pytest.fixture
def real_fixture():
    y = [1,1,0,0,1,0,1,1,0,0,1,1,0]
    return y

@pytest.fixture
def html_fixture():
    html = requests.get("https://regex.io")
    html = html.text
    return html

@pytest.fixture
def python_raw_text_fixture():

    "Fixtures for regex preprocessing"
    text = """Harrison Ford plays Sergeant Dutch Van Den Broeck of the District of Columbia Police Department. He tries to get the bad guys, 
            but doesn't do a very good job. When we meet up with him he's trying to catch a corrupt undercover officer. Kristin Scott Thomas plays a New Hampshire Senator, Kay Chandler, trying to get reelected. She's running against a candidate who has plenty of money. The last thing she needs is the death of her husband. She's a politician- she can't be bogged down by feelings.
            <br \/><br \/>This story moves slowly and painfully. I was looking at my watch every five minutes wondering when it would be over! The story gets lost in details the director, Sydney Pollack, didn't need to put in. We don't want to know about Dutch's police investigations. They throw in some insight to politicians and the \u0091spin control' they do for campaigns.
            After seeing the movie I'm still wondering why they got involved romantically. Doesn't anybody mourn anymore? Don't you need more than two weeks to even consider going \u0091horizontal' with someone else?<br \/><br \/>
            It was good to see actress, comedian, Chicago native and Second City Alumni Bonnie Hunt. Her role isn't necessarily comic relief, but she was"""
    return text

@pytest.fixture
def text_fixture():
    "Fixtures for vectorizer"

    with open(join(DATA_DIR, "pytest.json"), 'r') as ins:
        obj = json.load(ins)

    return obj