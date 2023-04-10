from typing import Optional, List, Dict, Tuple, Any
from pydantic import BaseModel,Field
from fastapi import FastAPI, Request
import uvicorn

import pickle

from src.paths import MODEL_DIR
from os.path import realpath,join

class InferenceInput(BaseModel):
    r"""Input to the Model"""
    text: str = Field(...,
                example = 'Today is a good day',
                title = 'What is on your mind?',
                max_length = 450)

class InferenceResult(BaseModel):
    r"""Output from the Model"""
    label: str = Field(example = "positive",
                        title = "Sentiment")

class InferenceResponse(BaseModel):
    r"""Response from Model"""
    error: str = Field(..., example=False, title='error?')
    results : Dict[str, Any] = Field(..., example={}, title='label and probability results')

class ErrorResponse(BaseModel):  
    error: str = Field(..., example = True, title = 'error?'),
    response: str = Field(..., example = "", title = 'type of error'),
    traceback: Optional[str] = Field(None, example = "", title = "detailed traceback of error")

app: FastAPI = FastAPI(title= "Sentiment Analysis",
            description= "Sentiment Analysis of written words, trained on imdb movie reviews")

@app.get("/")
def home():
    return "Welcome to the website, what is on your mind?"

@app.on_event("startup")
async def startup_event(): 

    label2id = {"positive": 1, "negative": 0}
    label2id = {value:key for key,value in label2id.items()}

    with open('data/pickle/train_vectorizer.pkl', 'rb') as ins:
        train_vectorizer = pickle.load(ins)
    
    with open(join(MODEL_DIR,'svm.pkl'), 'rb') as out:
        model = pickle.load(out)

    app.package = {'system': model, 'vectorizer':train_vectorizer, "label2id":label2id}
    print("model and vectorizer loaded successfully")

@app.post('/api/v1/classify',
  response_model = InferenceResponse,
  responses = {422: {'model': ErrorResponse}, 500: {'model': ErrorResponse}})

def classify(request: Request, body: InferenceInput):
    print('`/api/v1/classify` endpoint called.')

    system = app.package['system']
    vectorizer = app.package['vectorizer']
    label = app.package['label2id']
    text = [body.text]

    result = system.predict(vectorizer.transform(text))
    result = result.tolist()
    result = [(item,label.get(sentiment)) for item,sentiment in zip(text,result)]

    return {"error" : False,
            "results" : result,}

if __name__ == "__main__":
    uvicorn.run(app= app, port = 8080, log_level= 'info')


