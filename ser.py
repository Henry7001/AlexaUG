from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from utils import load_data

import os
import pickle

class entrena:

	X_train, X_test, y_train, y_test = load_data(test_size=0.25)

	print("[+] Numero de muestras de entrenamiento:", X_train.shape[0])

	print("[+] Numero de muestras de prueba:", X_test.shape[0])

	print("[+] Numero de caracteristicas:", X_train.shape[1])

	model_params = {
	    'alpha': 0.01,
	    'batch_size': 256,
	    'epsilon': 1e-08, 
	    'hidden_layer_sizes': (300,), 
	    'learning_rate': 'adaptive', 
	    'max_iter': 500, 
	}

	model = MLPClassifier(**model_params)

	# Entrenando el modelo
	print("[*] Entrenando el modelo...")
	model.fit(X_train, y_train)


	y_pred = model.predict(X_test)

	# calculando la precision
	accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

	print("Precision: {:.2f}%".format(accuracy*100))

	if not os.path.isdir("result"):
	    os.mkdir("result")

	pickle.dump(model, open("result/mlp_classifier.model", "wb"))
