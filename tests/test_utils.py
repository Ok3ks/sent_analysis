
from typing import Any, Dict, List
from src.utils import compute_metrics, parse_html, regex_preproc, extract_features

import numpy as np
import pytest

def test_compute_metrics(pred_fixture, real_fixture):
    #tests format of output
    x,y = compute_metrics(pred_fixture, real_fixture)
    assert isinstance (x, float or int), 'Output should either be a float or integer'
    assert isinstance (y, float or int), 'Output should either be a float or integer'

def test_parse_html(html_fixture):
    assert isinstance(parse_html(html_fixture), str), 'Output should be a string'

def test_regex_preproc(python_raw_text_fixture):
    assert isinstance(regex_preproc(python_raw_text_fixture), str)

def test_extract_features(text_fixture):
    assert isinstance(text_fixture, list), f'{type(text_fixture)}Can only extract features from strings'
    
    processed = [regex_preproc(text['review']) for text in text_fixture]
    vec,features,vectorizer,params = extract_features(processed)

    assert features.get("word_features") is not None, "word_features not found"
    assert features.get("stop_words") is not None, "stop_words not found"

    