import ernie
from ernie import SentenceClassifier, Models
classifier=SentenceClassifier(model_path='./sen_analysis')

#testing trained model output
sentence = "哦，太好了" # Oh, that's great
sentence2= "有缺陷的产品" # defective product
sentence3= "质量很好"          #great quality
sentence4= "刚吃一点没什么感觉"    #i didn't feel anythhing after usage
sentence5= "不是很好"#not great
sentences=[sentence,sentence2,sentence3,sentence4,sentence5]
def predict(sentences='有缺陷的产品'):
  classifier=SentenceClassifier(model_path='./sen_analysis')

  probabilities = classifier.predict_one(sentences)

  if(probabilities[0]>0.5):
    print("Positive")
  else:
    print("Negative")

predict()
