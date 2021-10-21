from flask import jsonify
import ernie
import pandas as pd
from ernie import SentenceClassifier
import argparse
import advertools as adv
import json
from datetime import date

def predict_many(dataset):
    data=pd.read_json(dataset)
    data= data.filter(['content'], axis=1)
    chinese_text=data['content'].to_list()
    classifier=SentenceClassifier(model_path='./sen_analysis')
    predictions = classifier.predict(chinese_text)
    outputs=[]
    for prediction in predictions:
        if(prediction[0]>0.5):
            outputs.append("False")
        else:
            outputs.append("True")
    returned_frame={"has_negtv":outputs}
    index=range(0,len(returned_frame['has_negtv']))
    second_column=pd.DataFrame(returned_frame,index=index)
    data['has_negtv']=second_column
    with open("predictions_"+str(date.today())+".json", 'w', encoding='utf-8') as file:
      data.to_json(file, force_ascii=False)

def word_frequency(dataset):
    #filtering
    data=pd.read_json(dataset)
    data= data.filter(['content','has_negtv'], axis=1)
    #negative
    negative_comments_dataset=data.query("has_negtv == True")
    neg_text_list= list(negative_comments_dataset["content"])
    neg=adv.word_frequency(neg_text_list)[:20]
    with open("Neg_"+str(date.today())+".json", 'w', encoding='utf-8') as file:
      neg.to_json(file, force_ascii=False)

    #positive
    positive_comments_dataset=data.query("has_negtv == False")
    pos_text_list= list(positive_comments_dataset["content"])
    pos=adv.word_frequency(pos_text_list)[:20]
    with open("Pos_"+str(date.today())+".json", 'w', encoding='utf-8') as file:
      pos.to_json(file, force_ascii=False)


def output(chinese_text):
    classifier=SentenceClassifier(model_path='./sen_analysis')
    probabilities = classifier.predict_one(chinese_text)

    if(probabilities[0]>0.5):
        print("Positive")
        return [{'Sentiment':"Positive"}]
    else:
        print("Negative")
        return [{'Sentiment':"Negative"}]



def get_parser(**kwargs):
    parser = argparse.ArgumentParser(description="NLP_demo")
    parser.add_argument("--prediction", help="make predictions from json app.",type=bool,default=False)
    parser.add_argument("--predict_many", help="make predictions from json app on a dataset.",type=bool,default=False)
    parser.add_argument("--word_freq", help="show positive and negative reviews word frequencies.",type=bool,default=False)
    parser.add_argument("--dataset_path", help="the location of the desired dataset."
    ,type=str)
    parser.add_argument("--prediction_text", help="the chinese text to know it's sentiment."
    ,type=str)
    return parser


if __name__ == "__main__":
    parser=get_parser()
    args=parser.parse_args()
    #args=vars(args)
    if args.prediction:
        prediction =output(args.prediction_text)
    elif args.predict_many:
        predict_many(args.dataset_path)
    elif args.word_freq:
        word_frequency(args.dataset_path)

    else:
        print("please use one of our selected arguments : --prediction  --predict_many  --word_freq")
