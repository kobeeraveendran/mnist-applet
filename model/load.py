import tensorflow as tf
from tensorflow import keras
from keras import models

def init():
    json_file = open('model_config.json', 'r')
    loaded_json_model = json_file.read()
    json_file.close()

    loaded_model = models.model_from_json(loaded_json_model)
    loaded_model = models.load_weights('model.h5')
    print('Loaded model from disk')
    loaded_model.compile(optimizer = 'rmsprop', 
                         loss = 'categorical_crossentropy', 
                         metrics = ['accuracy'])

    graph = tf.get_default_graph()

    return loaded_model, graph