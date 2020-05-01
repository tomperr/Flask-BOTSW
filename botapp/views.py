from flask import Flask, render_template, request, Response
from multiprocessing import Process, Value
from threading import Thread
from time import sleep
from models import bot

import numpy as np
import cv2
from mss import mss
from PIL import Image
from PIL import Image
import pandas as pd
import pickle
from sklearn.externals import joblib
from fastai2.vision.all import *
from fastai2.vision.widgets import *
from fastai2.basics import *
import mouse as m
import time as t
import random as r

def grand_parent_labeler(self, o):
	"Label `item` with the parent folder name."
	return Path(o).parent.name

path = Path('D:/Dev/Python/final2.pkl') # Path to get the Ml model
imagePath = Path('./Patch/') # Path to get images
frameRateNonAnalysis = 20 # Programm will analyse 1 frame of 20
bounding_box = {'top': 0, 'left': 0, 'width': 1800, 'height': 1200} # Screen configuration | if you are using a second screen modify this value
averageMouseDeplacement = 1 # Allows the programm to move the mouse | time in second
averageWaitingTime = 1 # Allows the programm to wait in order to net get ban
Refill = 1 # 1 = yes / 2 = no
TypeName = 'Dungeon' # 'Dungeon' / 'Rifts'
actual_run = bot.Run(load_learner(path), imagePath, frameRateNonAnalysis, bounding_box, averageMouseDeplacement, averageWaitingTime, Refill, TypeName)

app = Flask(__name__, static_folder='static')

stop_run = True
saved_timer = 0

# Config options - Make sure you created a 'config.py' file.
# app.config.from_object('config.py')
# To get one variable, tape app.config['MY_VARIABLE']

@app.route('/')
@app.route('/index/')
def index():
    return render_template('index.html')

def my_function():
	global stop_run
	global actual_run
	while not stop_run:
		sleep(0.01)
		actual_run.main()


def manual_run():
	global actual_run
	t = Thread(target=my_function)
	t.start()
	return render_template("index.html")


@app.route("/stop", methods=['GET'])
def set_stop_run():
	global stop_run
	stop_run = True
	return "Application stopped"


@app.route("/run", methods=['GET'])
def run_process():
	global stop_run
	stop_run = False
	return Response(manual_run(), mimetype="text/html")

@app.route('/', methods=['POST'])
def startAstop():
	global stop_run
	if 'start' in request.form:
		stop_run = False
		return Response(manual_run(), mimetype="text/html")
	elif 'stop' in request.form:
		stop_run = True
		return render_template("index.html")

@app.route('/ajax', methods = ['GET'])
def temp():
	global stop_run
	global saved_timer

	appState = "Processing"
	if stop_run:
		appState = "Application stopped"

	send = {
		"state" : appState,
		"nbvictory" : actual_run._nbVictory
	}

	if actual_run._lastTime != actual_run._time['last'] and actual_run._time['last'] != saved_timer:
		send['timer'] = "{0:.2f}".format(actual_run._time['last'])
		saved_timer = send['timer']

	if actual_run._prev_preds and len(actual_run._prev_preds) > 4:
		send["stage"] = actual_run._prev_preds[0]

	return send

if __name__ == "__main__":
    app.run()
