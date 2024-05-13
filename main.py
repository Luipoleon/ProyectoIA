import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Función para calcular la especificidad para multiclase

def specificity_score(y_true, y_pred):
    # Initialize the number of true negatives
    tn = 0

    # Initialize the number of false positives for each class
    fp = [0] * (max(max(y_true), max(y_pred)) + 1)

    # Calculate the number of true negatives and false positives
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tn += 1
        else:
            fp[pred] += 1

    # Calculate specificity for each class
    specificity = []
    for i in range(len(fp)):
        specificity.append(tn / (tn + fp[i]))

    return specificity

def main():

    # Cargar el dataset
    data = pd.read_csv('zoo2.csv')

    # One-hot encode the 'animal_name' column
    data = pd.get_dummies(data, columns=['animal_name'])

    # Dividir el dataset en características (X) y etiquetas (y)
    X = data[['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize']].values
    y = data['class_type']

    # Dividir el dataset en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Regresión logística
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    logistic_regression_predictions = logistic_regression.predict(X_test)


    # K-Vecinos Cercanos
    k_neighbors = KNeighborsClassifier()
    k_neighbors.fit(X_train, y_train)
    k_neighbors_predictions = k_neighbors.predict(X_test)
    

    # Máquinas de Vector Soporte
    support_vector_machines = SVC()
    support_vector_machines.fit(X_train, y_train)
    support_vector_machines_predictions = support_vector_machines.predict(X_test)

    # Naive Bayes
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, y_train)
    naive_bayes_predictions = naive_bayes.predict(X_test)

    # Neural Network
    neural_network = MLPClassifier(hidden_layer_sizes=(20, 20, 20, 20), max_iter=1000)
    neural_network.fit(X_train, y_train)
    neural_network_predictions = neural_network.predict(X_test)



    # Calcular la precisión, precisión, sensibilidad, especificidad y puntuación F1 para cada modelo
    logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_predictions)
    logistic_regression_precision = precision_score(y_test, logistic_regression_predictions, average='weighted', zero_division=1)
    logistic_regression_recall = recall_score(y_test, logistic_regression_predictions, average='weighted', zero_division=1)
    logistic_regression_specificity = specificity_score(y_test, logistic_regression_predictions)
    logistic_regression_f1_score = f1_score(y_test, logistic_regression_predictions, average='weighted')

    k_neighbors_accuracy = accuracy_score(y_test, k_neighbors_predictions)
    k_neighbors_precision = precision_score(y_test, k_neighbors_predictions, average='weighted', zero_division=1)
    k_neighbors_recall = recall_score(y_test, k_neighbors_predictions, average='weighted', zero_division=1)
    k_neighbors_specificity = specificity_score(y_test, k_neighbors_predictions)
    k_neighbors_f1_score = f1_score(y_test, k_neighbors_predictions, average='weighted')

    support_vector_machines_accuracy = accuracy_score(y_test, support_vector_machines_predictions)
    support_vector_machines_precision = precision_score(y_test, support_vector_machines_predictions, average='weighted', zero_division=1)
    support_vector_machines_recall = recall_score(y_test, support_vector_machines_predictions, average='weighted', zero_division=1)
    support_vector_machines_specificity = specificity_score(y_test, support_vector_machines_predictions)
    support_vector_machines_f1_score = f1_score(y_test, support_vector_machines_predictions, average='weighted')

    naive_bayes_accuracy = accuracy_score(y_test, naive_bayes_predictions)
    naive_bayes_precision = precision_score(y_test, naive_bayes_predictions, average='weighted')
    naive_bayes_recall = recall_score(y_test, naive_bayes_predictions, average='weighted')
    naive_bayes_specificity = specificity_score(y_test, naive_bayes_predictions)
    naive_bayes_f1_score = f1_score(y_test, naive_bayes_predictions, average='weighted')

    neural_network_accuracy = accuracy_score(y_test, neural_network_predictions)
    neural_network_precision = precision_score(y_test, neural_network_predictions, average='weighted')
    neural_network_recall = recall_score(y_test, neural_network_predictions, average='weighted')
    neural_network_specificity = specificity_score(y_test, neural_network_predictions)
    neural_network_f1_score = f1_score(y_test, neural_network_predictions, average='weighted')
    

  # Imprimir la precisión de cada método

    print("Resultados de Regresión Logística:")
    print("Precisión de Regresión Logística:", logistic_regression_accuracy)
    print("Precisión:", logistic_regression_precision)
    print("Sensibilidad:", logistic_regression_recall)
    print("Especificidad:", logistic_regression_specificity)
    print("Puntuación F1:", logistic_regression_f1_score)


    print("Resultados de K-Vecinos Cercanos:")
    print("Precisión de K-Vecinos Cercanos:", k_neighbors_accuracy)
    print("Precisión:", k_neighbors_precision)
    print("Sensibilidad:", k_neighbors_recall)
    print("Especificidad:", k_neighbors_specificity)
    print("Puntuación F1:", k_neighbors_f1_score)

    print("Resultados de Máquinas de Vector Soporte:")
    print("Precisión de Máquinas de Vector Soporte:", support_vector_machines_accuracy)
    print("Precisión:", support_vector_machines_precision)
    print("Sensibilidad:", support_vector_machines_recall)
    print("Especificidad:", support_vector_machines_specificity)
    print("Puntuación F1:", support_vector_machines_f1_score)

    print("Resultados de Naive Bayes:")
    print("Precisión de Naive Bayes:", naive_bayes_accuracy)
    print("Precisión:", naive_bayes_precision)
    print("Sensibilidad:", naive_bayes_recall)
    print("Especificidad:", naive_bayes_specificity)
    print("Puntuación F1:", naive_bayes_f1_score)

    print("Resultados de Neural Network:")
    print("Precisión de Neural Network:", neural_network_accuracy)
    print("Precisión:", neural_network_precision)
    print("Sensibilidad:", neural_network_recall)
    print("Especificidad:", neural_network_specificity)
    print("Puntuación F1:", neural_network_f1_score)


   
if __name__ == '__main__':
    main()