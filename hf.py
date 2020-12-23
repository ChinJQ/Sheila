import threading
from array import array

import wave
import pyaudio
import RPi.GPIO as GPIO
import time
from preprocess import *
import keras
import tensorflow as tf
from keras.models import model_from_json

try: 
    import queue
except ImportError:
    import Queue as queue
    
from queue import Queue, Full

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(14,GPIO.OUT)

start = time.time()

CHUNK_SIZE = 1024
MIN_VOLUME = 500
# if the recording thread can't consume fast enough, the listener will start discarding
BUF_MAX_SIZE = CHUNK_SIZE * 10

form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 44100 # 44.1kHz sampling rate
chunk = 4096 # 2^12 samples for buffer
record_secs = 1 # seconds to record
dev_index = 2 # device index found by p.get_device_info_by_index(ii)
wav_output_filename = 'test1.wav' # name of .wav file

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
model._make_predict_function()
print("Loaded model from disk")

end = time.time()
print(end - start)
print("Speak")

def main():
    stopped = threading.Event()
    q = Queue(maxsize=int(round(BUF_MAX_SIZE / CHUNK_SIZE)))

    listen_t = threading.Thread(target=listen, args=(stopped, q))
    listen_t.start()
    record_t = threading.Thread(target=record, args=(stopped, q))
    record_t.start()

    try:
        while True:
            listen_t.join(0.1)
            record_t.join(0.1)
    except KeyboardInterrupt:
        stopped.set()

    listen_t.join()
    record_t.join()


def record(stopped, q):
    while True:
        if stopped.wait(timeout=0):
            break
        chunk = q.get()
        vol = max(chunk)
        if vol >= MIN_VOLUME:
                GPIO.input(14) == GPIO.HIGH
                print("Start recording")
            
                frames = []

                # loop through stream and append audio chunks to frame array
                for i in range(0,int(samp_rate/CHUNK_SIZE*record_secs)):
                    data = stream.read(CHUNK_SIZE, exception_on_overflow = False)
                    #data = stream.read(chunk)
                    frames.append(data)

                GPIO.output(14, GPIO.LOW) # LED turn off
                print("finished recording")

                # stop the stream, close it, and terminate the pyaudio instantiation
                stream.stop_stream()
                stream.close()
                pyaudio.PyAudio().terminate()

                # save the audio frames as .wav file
                wavefile = wave.open(wav_output_filename,'wb')
                wavefile.setnchannels(chans)
                wavefile.setsampwidth(pyaudio.PyAudio().get_sample_size(form_1))
                wavefile.setframerate(samp_rate)
                wavefile.writeframes(b''.join(frames))
                wavefile.close()
                        
                folder_path = r'test1.wav'        
                
                start = time.time()

                # Getting the MFCC
                sample_1 = wav2mfcc(folder_path)

                # Reshape it
                sample_1_reshaped = sample_1.reshape(1, 20, 25, 1)
                
                # Perform forward pass
                print("Result:  ", get_labels()[0][np.argmax(model.predict(sample_1_reshaped))])
                
                end = time.time()
                print(end - start)
                print("\n")
            
                if  get_labels()[0][np.argmax(model.predict(sample_1_reshaped))] == 'Sheila':
                    for x in range(0, 5):
                        GPIO.output(14,GPIO.HIGH)
                        time.sleep(0.5)
                        GPIO.output(14,GPIO.LOW)
                        time.sleep(0.5)
                  
                GPIO.output(14,GPIO.LOW)
                time.sleep(5)
                global stream
                stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,channels=1,rate=44100,input=True,frames_per_buffer=1024)


        else:
            print('o')
            GPIO.output(14,GPIO.LOW)
            
def listen(stopped, q):
    global stream
    stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,channels=1,rate=44100,input=True,frames_per_buffer=1024)

    while True:
        if stopped.wait(timeout=0):
            break
        try:
            q.put(array('h', stream.read(CHUNK_SIZE, exception_on_overflow = False)))
        except Full:
            pass  # discard


if __name__ == '__main__':
    main()

