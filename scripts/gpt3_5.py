from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from typing import List

import os

temperature= 0.3
OPENAI_API_KEY= os.environ.get("OPENAI_API_KEY")
MODEL_NAME= os.environ.get("MODEL_NAME")

class EntryText(BaseModel):
    sentiment: List[str] = Field(description= "sentiment of given text")

output_parser = PydanticOutputParser(pydantic_object = EntryText)
format_instructions = output_parser.get_format_instructions()
model = OpenAI(model_name = MODEL_NAME, openai_api_key= OPENAI_API_KEY, temperature = temperature)

sentiment_prompt = PromptTemplate(
    template="""Assess the sentiment of this user's query \n{comment}. Respond if the sentiment of supplied comment is positive, neutral or negative \n{format_instructions}""",
    input_variables=["comment"],
    partial_variables= {"format_instructions": format_instructions})

sentiment_chain = LLMChain(llm = model, prompt = sentiment_prompt)

def get_sentiment(comment):
    result = sentiment_chain.run(comment)
    return result

def get_sentiment_json(json):
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('comment', help ='Add comment here')
    args = parser.parse_args()
    result = sentiment_chain.run(args)
    print(result)