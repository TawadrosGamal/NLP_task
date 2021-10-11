from flask import jsonify
import numpy as np
import pandas as pd
import flask
import sklearn
import ernie
from ernie import SentenceClassifier
from sklearn.preprocessing import LabelEncoder
import argparse
import advertools as adv
import pathlib
import requests
from transformers import AutoTokenizer, AutoConfig
import json


def train (dataset,input_location,output_location):
    classifier = SentenceClassifier(model_path=input_location)
    data=pd.read_json(dataset)
    data= data.filter(['content','has_negtv'], axis=1)
    lb_make = LabelEncoder()
    data["has_negtv"] = lb_make.fit_transform(data["has_negtv"])
    classifier.load_dataset(data, validation_split=0.2)
    classifier.fine_tune(epochs=10, learning_rate=2e-5, training_batch_size=32, validation_batch_size=64,class_weight={0: 0.11, 1: 0.89})
    classifier.dump(output_location)
    print("the new model has been saved to ",output_location)

def word_frequency(dataset):
    #filtering
    data=pd.read_json(dataset)
    data= data.filter(['content','has_negtv'], axis=1)
    #negative
    negative_comments_dataset=data.query("has_negtv == True")
    neg_text_list= list(negative_comments_dataset["content"])
    neg=adv.word_frequency(neg_text_list)[:20]
    neg.index = neg.index.map(str)
    neg.columns = neg.columns.map(str)
    neg_js = str(neg.to_dict()).replace("'", '"')
    #positive
    positive_comments_dataset=data.query("has_negtv == False")
    pos_text_list= list(positive_comments_dataset["content"])
    pos=adv.word_frequency(pos_text_list)[:20]
    pos.index = pos.index.map(str)
    pos.columns = pos.columns.map(str)
    pos_js = str(pos.to_dict()).replace("'", '"')

    return pos_js,neg_js

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def output(chinese_text,model_path):
    MAX_SEQ_LEN=100
    text = chinese_text
    encoded_input = tokenizer(text, pad_to_max_length=MAX_SEQ_LEN, max_length=MAX_SEQ_LEN)
    payload={"instances": [{"input_ids": encoded_input['input_ids'], "attention_mask": encoded_input['attention_mask']}]}
    print(encoded_input)
    print("\n")
    print(payload)
    response = requests.post(url,headers=headers,data=json.dumps(payload))
    print(json.loads(response.text))
    prediction= softmax(json.loads(response.text)['predictions'][0][0])
    print(softmax(json.loads(response.text)['predictions'][0]))
    if(prediction>0.5):
        print("Positive")
        return [{'Sentiment':"Positive"}]
    else:
        print("Negative")
        return [{'Sentiment':"Negative"}]



def get_parser(**kwargs):
    parser = argparse.ArgumentParser(description="NLP_demo")
    parser.add_argument("--prediction", help="make predictions from json app.",type=bool,default=False)
    parser.add_argument("--train", help="train the model on the inputed dataset.",type=bool,default=False)
    parser.add_argument("--word_freq", help="show positive and negative reviews word frequencies.",type=bool,default=False)
    parser.add_argument("--input_location", help="the input location to the model to train the data on.",
    default="sen_analysis",type=str)
    parser.add_argument("--output_location", help="the output location of the new trained model."
    ,type=str)
    parser.add_argument("--dataset_path", help="the location of the desired dataset."
    ,type=str)
    parser.add_argument("--prediction_text", help="the chinese text to know it's sentiment."
    ,type=ascii)


    return parser






if __name__ == "__main__":
    url = "http://localhost:8501/v1/models/bert_chinese:predict"
    headers = {'Content-Type': 'application/json'}
    tokenizer = AutoTokenizer.from_pretrained("./sen_analysis/tokenizer/")
    parser = get_parser()
    args=parser.parse_args()
    #args=vars(args)
    if args.prediction:
        prediction =output(args.prediction_text,args.input_location)
    elif args.train:
        train(args.dataset_path,args.input_location,args.output_location)
    elif args.word_freq:
        word_frequency(args.dataset_path)

    else:
        print("please use one of our selected arguments : --prediction  --train  --word_freq")