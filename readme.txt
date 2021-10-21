python nlp_api.py --prediction True --prediction_text Chinese text
input : Chinese text as a string
output: json with Sentiment and prints the prediction

python nlp_api.py --predict_many True --dataset_path DATASET_PATH
input : json with content column
output: json with content and has_negtv columns
format json file name : prediction_yy-mm-dd


python nlp_api.py --word_freq True --dataset_path DATASET_PATH
input : json with content and has_negtv columns
output: 2 json files one for 20 most frequent positive comments and the other with the 20 most frequent negative comments
format json file name : Neg_yy-mm-dd
format json file name : Pos_yy-mm-dd
