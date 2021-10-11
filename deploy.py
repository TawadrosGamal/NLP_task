import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification
MAX_SEQ_LEN = 100

model = TFAutoModelForSequenceClassification.from_pretrained("sen_analysis")
callable = tf.function(model.call)
concrete_function = callable.get_concrete_function([tf.TensorSpec([None, MAX_SEQ_LEN], tf.int32, name="input_ids"), tf.TensorSpec([None, MAX_SEQ_LEN], tf.int32, name="attention_mask")])
model.save('saved_model/bert_chinese/1',signatures=concrete_function)

#docker command

#docker run -p 8501:8501 --mount type=bind,source=/root/NLP_models/saved_model/bert_chinese,target=/models/bert_chinese -e MODEL_NAME=bert_chinese -t tensorflow/serving