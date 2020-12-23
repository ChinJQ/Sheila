from preprocess import *
import keras
import tensorflow as tf
from keras.models import model_from_json


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")


folder_path = r'./DataValidation/NotSheila'
# Getting the MFCC
sample_1 = wav2mfcc(folder_path +'/1.wav')
sample_2 = wav2mfcc(folder_path +'/2.wav')
sample_3 = wav2mfcc(folder_path +'/3.wav')
sample_4 = wav2mfcc(folder_path +'/4.wav')
sample_5 = wav2mfcc(folder_path +'/5.wav')
sample_6 = wav2mfcc(folder_path +'/6.wav')
sample_7 = wav2mfcc(folder_path +'/7.wav')
sample_8 = wav2mfcc(folder_path +'/8.wav')
sample_9 = wav2mfcc(folder_path +'/9.wav')
sample_10 = wav2mfcc(folder_path +'/10.wav')



# We need to reshape it remember?
sample_1_reshaped = sample_1.reshape(1, 20, 25, 1)
sample_2_reshaped = sample_2.reshape(1, 20, 25, 1)
sample_3_reshaped = sample_3.reshape(1, 20, 25, 1)
sample_4_reshaped = sample_4.reshape(1, 20, 25, 1)
sample_5_reshaped = sample_5.reshape(1, 20, 25, 1)
sample_6_reshaped = sample_6.reshape(1, 20, 25, 1)
sample_7_reshaped = sample_7.reshape(1, 20, 25, 1)
sample_8_reshaped = sample_8.reshape(1, 20, 25, 1)
sample_9_reshaped = sample_9.reshape(1, 20, 25, 1)
sample_10_reshaped = sample_10.reshape(1, 20, 25, 1)


# Perform forward pass
print("Result:  ", get_labels()[0][
    np.argmax(model.predict(sample_1_reshaped))
      ]
    )
#print(np.argmax(model.predict(sample_6_reshaped)))
print("Result:  ", get_labels()[0][
    np.argmax(model.predict(sample_2_reshaped))
      ]
    )
print("Result:  ", get_labels()[0][
    np.argmax(model.predict(sample_3_reshaped))
      ]
    )
print("Result:  ", get_labels()[0][
    np.argmax(model.predict(sample_4_reshaped))
      ]
    )
print("Result:  ", get_labels()[0][
    np.argmax(model.predict(sample_5_reshaped))
      ]
    )
print("Result:  ", get_labels()[0][
    np.argmax(model.predict(sample_6_reshaped))
      ]
    )
print("Result:  ", get_labels()[0][
    np.argmax(model.predict(sample_7_reshaped))
      ]
    )
print("Result:  ", get_labels()[0][
    np.argmax(model.predict(sample_8_reshaped))
      ]
    )
print("Result:  ", get_labels()[0][
    np.argmax(model.predict(sample_9_reshaped))
      ]
    )
print("Result:  ", get_labels()[0][
    np.argmax(model.predict(sample_10_reshaped))
      ]
    )
