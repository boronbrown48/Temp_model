from fastapi import FastAPI
from torch import nn
import torch
from transformers import RobertaForSequenceClassification, CamembertTokenizer
import uvicorn

sm = nn.Softmax(dim=1)
tokenizer = CamembertTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
app = FastAPI()

########################## Routes ##########################

@app.get("/")
def read_root():
    return {"Tokenizer"}

if __name__ == "__main__":
    uvicorn.run("classificationApp:app")
