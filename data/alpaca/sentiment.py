import os
import json

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.chains import LLMChain


from os.path import realpath,dirname
from os.path import join as j

temperature= 0.3
OPENAI_API_KEY= os.environ.get("OPENAI_API_KEY")
MODEL_NAME= os.environ.get("MODEL_NAME")

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()
model = OpenAI(model_name = MODEL_NAME, openai_api_key= OPENAI_API_KEY, temperature = temperature)
output_path = realpath(j(dirname(__file__), 'training_data'))

#Label data with GPT3.5
#Filter accurate datapoints with cleanlab

def get_sentiment(text):
    """Obtains the sentiment of a text from pipeline"""
    sentiment_prompt = PromptTemplate(template="""Assess the sentiment of this user's query \n{comment}. You must respond with only one answer. Respond if the sentiment of supplied comment is one of the three options: positive, neutral or negative \n{format_instructions}""",
                                        input_variables=["comment"],
                                        partial_variables= {"format_instructions": format_instructions})
    
    sentiment_chain = LLMChain(llm = model, prompt = sentiment_prompt)
    sentiment = sentiment_chain.predict_and_parse(comment = text)
    print(sentiment)
    return sentiment.replace("\n", "")

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-filepath', help="path to file")
    args = parser.parse_args()
    file = args.filepath

    with open(file, 'r') as ins:
        news = json.load(ins)

    sentiment = {}

    #CHECK IF DIRECTORY HAS BEEN POPULATED BEFORE
    if os.path.isfile(j(output_path, 'train.json')):
        with open(j(output_path, 'train.json'), 'r') as ins:
            sentiment = json.load(ins)

    for key,value in zip(news['id'],news['text']):
        if key not in sentiment.keys():
            sentiment.update({key:get_sentiment(value)})

    #adapt so that json is written continuous in case of any api restrictions
    with open(j(output_path, 'train.json'), 'w') as out:
        json.dump(sentiment, out)
   