# main.py
import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MT5ForConditionalGeneration

import requests
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


def encode_str(text, tokenizer, seq_len):
    input_ids = tokenizer.encode(
        text=text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=seq_len
    )
    return input_ids[0]

def analysis(input):
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    model_name = 'Step-1802_checkpoint_lang_pred.pt'

    model.resize_token_embeddings(len(tokenizer))
    # Load the model parameters onto the CPU
    model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
    
    input_ids = encode_str(text=input, tokenizer=tokenizer, seq_len=40)
    input_ids = input_ids.unsqueeze(0)

    output_tokens = model.generate(
        input_ids,
        num_beams=10,
        num_return_sequences=1,
        length_penalty=1,
        no_repeat_ngram_size=2
    )

    # Decode and display the result
    for token_set in output_tokens:
        prediction = tokenizer.decode(token_set, skip_special_tokens=True)
    return prediction

@app.post("/model/{comment}")
async def getItem(comment):
    prediction = analysis(comment)
    return {"prediction":prediction}
