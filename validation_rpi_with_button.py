import time
start = time.time()

import pyaudio
import wave

form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 44100 # 44.1kHz sampling rate
chunk = 4096 # 2^12 samples for buffer
record_secs = 3 # seconds to record
dev_index = 2 # device index found by p.get_device_info_by_index(ii)
wav_output_filename = 'recordedsound.wav' # name of .wav file


import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
GPIO.setwarnings(False) # Ignore warning for now
GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set initial value to be pulled low (off)
GPIO.setup(8, GPIO.OUT, initial=GPIO.LOW) # Set pin 8 to be an output pin 
#GPIO.setmode(GPIO.BCM)
#GPIO.setup(14,GPIO.OUT)

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


print("Press button")
end = time.time()
print(end - start)

while True: # Run forever
    if GPIO.input(10) == GPIO.HIGH:
        GPIO.output(8, GPIO.HIGH) # LED turn on
        
        print("Button was pushed!")

        audio = pyaudio.PyAudio() # create pyaudio instantiation

        # create pyaudio stream
        stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                            input_device_index = dev_index,input = True, \
                            frames_per_buffer=chunk)
        
        print("Start recording")
        frames = []

        # loop through stream and append audio chunks to frame array
        for ii in range(0,int((samp_rate/chunk)*record_secs)):
            data = stream.read(chunk, exception_on_overflow = False)
            frames.append(data)

        GPIO.output(8, GPIO.LOW) # LED turn off
        print("finished recording")


        # stop the stream, close it, and terminate the pyaudio instantiation
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # save the audio frames as .wav file
        wavefile = wave.open(wav_output_filename,'wb')
        wavefile.setnchannels(chans)
        wavefile.setsampwidth(audio.get_sample_size(form_1))
        wavefile.setframerate(samp_rate)
        wavefile.writeframes(b''.join(frames))
        wavefile.close()
                
        folder_path = r'recordedsound.wav'

        
        start = time.time()

        
        # Getting the MFCC
        sample_1 = wav2mfcc(folder_path)


        # We need to reshape it remember?
        sample_1_reshaped = sample_1.reshape(1, 20, 25, 1)


        # Perform forward pass
        print("Result:  ", get_labels()[0][
            np.argmax(model.predict(sample_1_reshaped))
            ]
            )
        
        end = time.time()
        print(end - start)
