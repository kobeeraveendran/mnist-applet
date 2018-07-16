import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json

def init():
    json_file = open('model/model_config.json', 'r')
    loaded_json_model = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_json_model)
    loaded_model.load_weights('model/model.h5')
    # loaded_model = keras.models.Model().load_weights('model.h5')
    print('Loaded model from disk')
    loaded_model.compile(optimizer = 'rmsprop', 
                         loss = 'categorical_crossentropy', 
                         metrics = ['accuracy'])

    graph = tf.get_default_graph()

    return loaded_model, graph