import pyaudio
import wave
import pickle
from sys import byteorder
from array import array
from struct import pack
from sklearn.neural_network import MLPClassifier
import subprocess
from utils import extract_feature
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, updater
from telegram import File
from telegram.files.audio import Audio

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 30

def message (update, context):
    update.message.reply_text(update.message.text.upper())

def command_handler(update, context):
    update.message.reply_text(f"Command: {update.message.text}")

def receive_image(update, context):
    try:
        print(update)
        obj = context.bot.getFile(file_id=update.message.document.file_id)
        obj.download()
        update.message.reply_text("Guardando imagen...")
    except Exception as e:
        print(str(e))

def receive_audio(update, context):
    try:
        audio_obj = context.bot.getFile(file_id=update.message.audio.file_id)
        audio_obj.download()
        update.message.reply_text("Guardando audio...")
        update.message.reply_text("Guardado!")
    except Exception as e:
        print(str(e))


def voice_process(update, context):
    try:
        audio_obj = context.bot.get_file(file_id=update.message.voice.file_id)
        audio_obj.download("file.ogg")  
        update.message.reply_text("Procesando audio y detectando estado...")
    except Exception as e:
        print(str(e))

    src_filename = 'file.ogg'
    dest_filename = 'file.wav'

    process = subprocess.run(['ffmpeg', '-i', src_filename, dest_filename])
    if process.returncode != 0:
        raise Exception("Something went wrong")   

    model = pickle.load(open("result/mlp_classifier.model", "rb"))
    filename = "file.wav"
    features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    result = model.predict(features)[0]
    print("Estado de animo:", result)
    update.message.reply_text("Su estado de animo es:")  
    update.message.reply_text(result)  
    
def start(update, context):
    update.message.reply_text("Bienvenido, para utilizar las funciones de AlexaUG, por favor, envía una nota de voz")
    

def main():
    updater = Updater(token="1713033116:AAHjRC7U-CS8-Q_4kpYKIQm3e7uIsiTLMNg", use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text, message))
    dp.add_handler(MessageHandler(Filters.command, command_handler))
    dp.add_handler(MessageHandler(Filters.document, receive_image))
    dp.add_handler(MessageHandler(Filters.voice, voice_process))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()



    
    