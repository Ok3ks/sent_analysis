import os

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from deploy.demo.sent_gpt3_5_api import EntryText

temperature= 0.3
OPENAI_API_KEY= os.environ.get("OPENAI_API_KEY")
MODEL_NAME= os.environ.get("MODEL_NAME")

output_parser = PydanticOutputParser(pydantic_object = EntryText)
format_instructions = output_parser.get_format_instructions()
model = OpenAI(model_name = MODEL_NAME, openai_api_key= OPENAI_API_KEY, temperature = temperature)
output_path = realpath(j(dirname(__file__), 'training_data'))

#Label data with GPT3.5
#Filter accurate datapoints with cleanlab

def get_sentiment(text):
    """Obtains the sentiment of a text from pipeline"""
    #print(text)
    sentiment_prompt = PromptTemplate(template="""Assess the sentiment of this user's query \n{comment}. Respond if the sentiment of supplied comment is positive, neutral or negative \n{format_instructions}""",
                                        input_variables=["comment"],
                                        partial_variables= {"format_instructions": format_instructions})
    
    sentiment_chain = LLMChain(llm = model, prompt = sentiment_prompt)
    sentiment = sentiment_chain.run(text)
    print(sentiment)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-filepath',
        help="path to file")
    args = parser.parse_args()
    file = args.filepath

    with open(file, 'r') as ins:
        news = json.load(ins)

    sentiment = {}

    for key,value in news.items():
        text = "".join(news[key])
        sentiment['sentiment'] = get_sentiment(text)
        sentiment['review'] = "".join(text)
    
    with open(j(output_path, 'train.json'), 'w') as out:
        json.dump(output, out)
   