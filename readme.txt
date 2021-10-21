Usage:


Predict on sentence:

python nlp_api.py --prediction True --prediction_text Chinese text
input : Chinese text as a string
output: json object with Sentiment and prints the prediction (positive / negative)


predict on dataset:

python nlp_api.py --predict_many True --dataset_path DATASET_PATH
input : json dataset with content column
output: json file with content and has_negtv columns (located in same directory as api)
json file name format : prediction_yy-mm-dd.json


extract most frequent comments from dataset:

python nlp_api.py --word_freq True --dataset_path DATASET_PATH
input : json dataset with content and has_negtv columns
output: 2 json files one for 20 most frequent positive comments and the other with the 20 most frequent negative comments (located in same directory as api)
json file name format : Neg_yy-mm-dd.json
json file name format : Pos_yy-mm-dd.json
