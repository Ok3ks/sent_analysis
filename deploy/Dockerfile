FROM python:3.9-slim

WORKDIR /app
COPY ["sent_gpt3_5_api.py","init_env.sh" ,"auth.sh", "./"]  

RUN pip install --upgrade pip && pip install celery==5.2.2 \
    dotmap==1.3.30 \
    fastapi==0.75.2 \
    label-studio==1.4.1.post1 \
    matplotlib==3.5.2 \
    metaflow==2.6.0 \
    numpy==1.22.3 \
    pandas==1.3.5 \
    plotly==5.7.0 \
    Pillow==9.0.0 \
    pytest==7.1.2 \
    pytorch-lightning==1.6.3 \
    scikit-image==0.19.2 \
    scikit-learn==1.0.2 \
    scipy==1.8.0 \
    torch==1.11.0 \
    torchmetrics==0.8.1 \
    torchvision==0.12.0 \
    tqdm==4.64.0 \
    uvicorn[standard]==0.17.6 \
    wandb==0.12.16 \
    datasets==2.12.0 \
    bs4==0.0.1 \
    streamlit==1.22.0
    
CMD ["source" ,"init_env.sh"]
CMD ["source" ,"auth.sh"]
CMD ["uvicorn", "sent_gpt3_5_api:app", "--host", "0.0.0.0", "--port", "8080"]