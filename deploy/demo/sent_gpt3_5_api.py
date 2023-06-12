from typing import Optional, List, Dict, Tuple, Any
from pydantic import BaseModel,Field
from fastapi import FastAPI, Request
from os.path import realpath,join

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from typing import List

import os
import uvicorn
import pickle

class InferenceInput(BaseModel):
    r"""Input to the Model"""
    text: str = Field(...,
                example = 'Today is a good day',
                title = 'What is on your mind?',
                max_length = 1000)

class InferenceResult(BaseModel):
    r"""Output from the Model"""
    label: str = Field(example = "positive",
                        title = "Sentiment")

class EntryText(BaseModel):
    sentiment: List[str] = Field(description= "sentiment of given text")

class InferenceResponse(BaseModel):
    r"""Response from Model"""
    error: str = Field(..., example=False, title='error?')
    sentiment: List[str] = Field(description= "sentiment of given text")

class ErrorResponse(BaseModel):  
    error: str = Field(..., example = True, title = 'error?'),
    response: str = Field(..., example = "", title = 'type of error'),
    traceback: Optional[str] = Field(None, example = "", title = "detailed traceback of error")

app: FastAPI = FastAPI(title= "Sentiment Analysis",
            description= "Sentiment Analysis of written words, using OpenAI GPT-4")

@app.get("/")
def home():
    return "Welcome to the website, what is on your mind?"

@app.on_event("startup")
async def startup_event():
    temperature= 0.3
    OPENAI_API_KEY= os.environ.get("OPENAI_API_KEY")
    MODEL_NAME= os.environ.get("MODEL_NAME")

    output_parser = PydanticOutputParser(pydantic_object = EntryText)
    format_instructions = output_parser.get_format_instructions()
    model = OpenAI(model_name = MODEL_NAME, openai_api_key= OPENAI_API_KEY, temperature = temperature)

    sentiment_prompt = PromptTemplate(template="""Assess the sentiment of this user's query \n{comment}. Respond if the sentiment of supplied comment is positive, neutral or negative \n{format_instructions}""",
    input_variables=["comment"],
    partial_variables= {"format_instructions": format_instructions})

    sentiment_chain = LLMChain(llm = model, prompt = sentiment_prompt)
    app.package = {'system': sentiment_chain}
    print("model loaded successfully")

@app.post('/api/v1/classify', responses = {422: {'model': ErrorResponse}, 500: {'model': ErrorResponse}})

def classify(request: Request, body: InferenceInput):
    print('`/api/v1/classify` endpoint called.')

    system = app.package['system']
    text = [body.text]

    result = system.run(text)

    return {"error" : False,
            "results" : result,}

if __name__ == "__main__":
    uvicorn.run(app= app, port = 8080, log_level= 'info')


