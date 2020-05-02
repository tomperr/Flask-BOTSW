import warnings
warnings.filterwarnings("ignore")

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

class Run:
	def __init__(self, path, imagePath, frameRateNonAnalysis, bounding_box, averageMouseDeplacement, averageWaitingTime, Refill, TypeName):
		self._start = False
		self._path = path
		self._imagePath = imagePath
		self._frameRateNonAnalysis = frameRateNonAnalysis
		self._bounding_box = bounding_box
		self._averageMouseDeplacement = averageMouseDeplacement
		self._averageWaitingTime = averageWaitingTime
		self._Refill = Refill
		self._TypeName = TypeName
		self._typeRun = {
			'Dungeon' : [['Victory', 3, 1], ['Cross', 1, 1]],
			'Rifts' : [['Damage', 5, 1], ['Cross-Rift', 1, 1]]
		}

		self._bestLocation = (0,0)
		self._ui_template = {
						'Rejouer' : {'image':cv2.imread(str(self._imagePath/'Rejouer.jpg'),0), 'match_score':9037656.0, 'name':'Rejouer'},
						'Cross' : {'image':cv2.imread(str(self._imagePath/'Cross.jpg'),0), 'match_score':100000.0, 'name':'Cross'},
						'Oui' : {'image':cv2.imread(str(self._imagePath/'Oui.jpg'),0), 'match_score':6000312.0, 'name':'Oui'},
						'Fermer' : {'image':cv2.imread(str(self._imagePath/'Fermer.jpg'),0), 'match_score':3665168.0, 'name':'Fermer'},
						'+90' : {'image':cv2.imread(str(self._imagePath/'+90.jpg'),0), 'match_score':3532480.0, 'name':'+90'},
						'Ok' : {'image':cv2.imread(str(self._imagePath/'Ok.jpg'),0), 'match_score':1445952.0, 'name':'Ok'},
						'Shop' : {'image':cv2.imread(str(self._imagePath/'Shop.jpg'),0), 'match_score':3532480.0, 'name':'Shop'},
						'Non' : {'image':cv2.imread(str(self._imagePath/'Non.jpg'),0), 'match_score':200056.0, 'name':'Non'},
						'Preparation' : {'image':cv2.imread(str(self._imagePath/'Preparation.jpg'),0), 'match_score':400056.0, 'name':'Preparation'},
						'Go' : {'image':cv2.imread(str(self._imagePath/'Go.jpg'),0), 'match_score':1800056.0, 'name':'Go'},
						'Victory' : {'image':cv2.imread(str(self._imagePath/'Victory.jpg'),0), 'match_score':152788.0/1.9, 'name':'Victory'},
						'Damage' : {'image':cv2.imread(str(self._imagePath/'Damage.jpg'),0), 'match_score':152788.0/1.9, 'name':'Damage'},
						'Cross-Rift' : {'image':cv2.imread(str(self._imagePath/'Cross-Rift.jpg'),0), 'match_score':152788.0/1.9, 'name':'Cross-Rift'}
					  }


		self._orderSuccess = [
							self._typeRun[self._TypeName][0],
							self._typeRun[self._TypeName][1],
							['Cross', 1, 0],
							['Rejouer', 1, 1],
							['Shop', 1, Refill],
							['+90', 1, 1],
							['Oui', 1, 1],
							['Ok', 1, 1],
							['Fermer', 1, 1],
							['Rejouer', 1, 1]
					   ]
		self._orderFail = [
							['Non', 1, 1],
							['Victory', 1, 1],
							['Preparation', 1, 1],
							['Go', 1, 1],
							['Shop', 1, Refill],
							['+90', 1, 1],
							['Oui', 1, 1],
							['Ok', 1, 1],
							['Fermer', 1, 1],
							['Rejouer', 1, 1],
							['Go', 1, 1]
					]

		self._sct = mss()
		self._learn_inf = self._path
		self._actu_pred = False
		self._prev_preds = []

		self._i = frameRateNonAnalysis
		self._c = 0
		self._forcePath = False
		self._endAction = False
		self._nbRun = 0
		self._nbVictory = 0

		self._time = {"start" : 0, "stop" : 0, "last" : 0}
		self._startedTime = False
		self._lastTime = 0

		self._checkedPred = "None"

	def verif(self, array):
		for num in array:
			for numb in array:
				if num != numb:
					return False
		return True

	def checkTemplate(self, screen, template, order):
		w, h = template['image'].shape[::-1]

		# All the 6 methods for comparison in a list
		methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
		            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

		result = cv2.matchTemplate(screen, template['image'], cv2.TM_SQDIFF)
		(minVal, maxVal, minLocation, maxLocation) = cv2.minMaxLoc(result)
		# print(minVal, template['name'])

		if (minVal <= 2*template['match_score'] and order[self._c][2] != 2):
			self._bestLocation = (int(minLocation[0] + template['image'].shape[1]/2), int(minLocation[1] + template['image'].shape[0]/2))
			# cv2.circle(self._screen, (self._bestLocation),50, (0, 255, 0), 2)
			# cv2.imshow('test', self._screen)
			# cv2.waitKey(0)
			self.clickOn(order[self._c])
			self._c += 1
		elif order[self._c][2] == 0:
			self._c += 1

		if self._c == 5:
			self._forcePathFail = False

		if self._c == len(order):
			self._c = 0
			self._endAction = True

	def clickOn(self, label):
		for self._i in range(label[1]):
			duration = r.randrange(0,self._averageMouseDeplacement)/2
			m.move(self._bestLocation[0] + self._ui_template[label[0]]['image'].shape[1] * r.randrange(-1, 1)/2, self._bestLocation[1] + self._ui_template[label[0]]['image'].shape[0] * r.randrange(-1, 1)/2, duration=duration)
			m.click(button='left')
			sleep(self._averageWaitingTime + duration)

	def main(self):
		if(self._i == self._frameRateNonAnalysis):
			sct_img = self._sct.grab(self._bounding_box)
			output = np.array(Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX"))
			pred,pred_idx,probs = self._learn_inf.predict(output)
			# print('nombre de run : ' + str(self._nbRun), end = '\r')

			if len(self._prev_preds) > 6:
				self._prev_preds.insert(0, pred)
				self._prev_preds.pop()
			else:
				self._prev_preds.append(pred)



			if pred == 'Wave' or pred == 'Boss':
				self._c = 0
				self._endAction = False
				if pred == 'Wave':
					if not(self._startedTime):
						self._startedTime = True
						self._time['start'] = time.perf_counter()

			if self.verif(self._prev_preds):
				self._checkedPred = self._prev_preds[0]
				if not(self._endAction):
					if self._prev_preds[0] == 'Fail' or self._forcePath == 'Fail':
						if self._c < len(self._orderFail):
							if self._c == 3:
								self._nbRun += 1
							self.checkTemplate(cv2.cvtColor(output,cv2.COLOR_RGB2GRAY), self._ui_template[self._orderFail[self._c][0]], self._orderFail)
							self._forcePathFail = 'Fail'
					if self._prev_preds[0] == 'Reward' or self._forcePath == 'Success':
						# for template in self.ui_template:
						# 	checkTemplate(cv2.cvtColor(output,cv2.COLOR_RGB2GRAY), self.ui_template[template])
						if self._startedTime:
							self._startedTime = False
							self._time['stop'] = time.perf_counter()
							self._lastTime = self._time['last']
							self._time['last'] = "{0:.2f}".format(self._time['stop'] - self._time['start'] - 4)

						if self._c < len(self._orderSuccess):
							if self._c == 3:
								self._nbRun += 1
								self._nbVictory += 1
							self.checkTemplate(cv2.cvtColor(output,cv2.COLOR_RGB2GRAY), self._ui_template[self._orderSuccess[self._c][0]], self._orderSuccess)
							self._forcePathFail = 'Success'
			self._i = 0
		self._i+=1