from typing import Optional, List, Dict
from pydantic import BaseModel,Field

from fastapi import FastAPI
import uvicorn
from os.path import realpath,join

class InferenceInput(BaseModel):
    r"""Input to the Model"""
    text: str = Field(...,
                example = 'Today is a good day',
                title = 'What is on your mind?',
                max_length = 200)

class InferenceResponse(BaseModel):
    r"""Response from Model"""
    label: int = Field(example = 0,
                    title= "Sentiment from model" )

class InferenceResult(BaseModel):
    r"""Output from the Model"""
    label: str = Field()

class ErrorResponse(BaseModel):  
    error: str = Field(..., example = True, title = 'error?'),
    response: str = Field(..., example = "", title = 'type of error'),
    traceback: Optional[str] = Field(None, example = "", title = "detailed traceback of error")

app: FastAPI(title= "Sentiment Analysis",
            description= "Sentiment Analysis of written words, trained on imdb movie reviews")

@app.get("/")
def home():
    return "Welcome to the website, what is on your mind?"

#@app.on_event("startup")
#def load(): 
    #with open( join(MODEL_DIR, 'svm.pkl'), 'r') as ins:
        #model = pickle.load(ins)
    #print("model loaded successfully")


if __name__ == "__main__":
    uvicorn.run(app= app, port = 8080, log_level= info)

