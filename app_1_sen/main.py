from fastapi import FastAPI
from torch import nn
import torch
from transformers import RobertaForSequenceClassification, CamembertTokenizer
import uvicorn

sm = nn.Softmax(dim=1)
sentiment_model1 = RobertaForSequenceClassification.from_pretrained("boronbrown48/sentiment_neutral_from_other_v2", num_labels=2)
app = FastAPI()

########################## Routes ##########################

@app.get("/")
def read_root():
    return {"Sentiment_1"}

if __name__ == "__main__":
    uvicorn.run("classificationApp:app")
