FROM python:3.9-slim

WORKDIR /app
COPY ["requirements.txt","sent_gpt3_5_api.py","gpt_env.sh" ,"./"]  

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["source" ,"gpt_env.sh"]
CMD ["uvicorn", "sent_gpt3_5_api:app", "--host", "0.0.0.0", "--port", "8080"]