# import uvicorn
# import os
import json
# import requests
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import RedirectResponse
import opendatasets as od
from fastapi import APIRouter, HTTPException

from kaggle.api.kaggle_api_extended import KaggleApi

import pandas as pd

from fastapi import FastAPI, HTTPException

from fastapi import APIRouter
import os
import opendatasets as od
from fastapi.responses import JSONResponse

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
from google.cloud import firestore
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
from firestore import FirestoreClient

router = APIRouter()

# Define the directory to save the dataset
DATA_DIR = "src/data"

# Create a route to download the dataset
def download_dataset(dataset_name):
    JSON_FILE_PATH = "src\config\model_parameters.json"
    try:
        api = KaggleApi()
        api.authenticate()  # Authentifier l'API
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pas la bonne cle : {e}")
    if not os.path.exists(JSON_FILE_PATH):
        raise HTTPException(status_code=404, detail=f"Le fichier JSON '{JSON_FILE_PATH}' est introuvable.")
    try:
        # Charge le contenu du fichier JSON
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
            datasets = data.get("datasets", {})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail=f"Le fichier JSON est mal formé.{test}")

    # Vérifie si le dataset existe dans le fichier JSON
    if dataset_name not in datasets:
        raise HTTPException(status_code=404, detail=f"Le dataset '{dataset_name}' n'existe pas dans le fichier JSON.")
    
    # Récupère les informations du dataset
    dataset_info = datasets[dataset_name]
    dataset_url = dataset_info["url"]
    try:
        # Ensure the data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)

        # Download the dataset
        od.download(dataset_url, DATA_DIR)

        return {"message": "Dataset downloaded successfully!", "path": DATA_DIR}
    except Exception as e:
        return {"error": str(e)}
    
def add_dataset(name, url):
    JSON_FILE_PATH = "src/config/model_parameters.json"
    
    # Vérifier l'existence du fichier JSON
    if not os.path.exists(JSON_FILE_PATH):
        raise HTTPException(status_code=404, detail=f"Fichier JSON '{JSON_FILE_PATH}' introuvable.")
    
    try:
        # Charger les données existantes
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)  # Charger la structure complète (datasets + models)
            datasets = data.get("datasets", {})  # Accéder aux datasets
            model = data.get("model", {})  # Accéder aux models
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Le fichier JSON est mal formé.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la lecture du fichier JSON : {e}")
    
    # Vérifier si le dataset existe déjà
    if name in datasets:
        raise HTTPException(status_code=400, detail=f"Le dataset '{name}' existe déjà.")
    
    # Ajouter le nouveau dataset
    datasets[name] = {
        "name": name,
        "url": url
    }

    # Mettre à jour la structure complète avec les nouveaux datasets
    data["datasets"] = datasets  # Réinsérer les datasets dans la structure complète
    data["model"] = model  # Réinsérer les models (les conserver inchangés)

    try:
        # Écrire les modifications dans le fichier JSON
        with open(JSON_FILE_PATH, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)  # Sauvegarder la structure complète (datasets + models)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'écriture dans le fichier JSON : {e}")

    return {"message": f"Le dataset '{name}' a été ajouté avec succès."}
    # return data


    
def modify_dataset(old_name, old_url, new_name, new_url):
    JSON_FILE_PATH = "src\config\model_parameters.json"
    if not os.path.exists(JSON_FILE_PATH):
        raise HTTPException(status_code=404, detail=f"Fichier JSON '{JSON_FILE_PATH}' introuvable.")

    try:
        # Charger les datasets existants
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)  # Charger la structure complète (datasets + models)
            datasets = data.get("datasets", {})  # Accéder aux datasets
            model = data.get("model", {})  # Accéder aux models
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Le fichier JSON est mal formé.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la lecture du fichier JSON : {e}")

    # Vérification de l'existence du dataset à modifier
    if old_name not in datasets or datasets[old_name]["url"] != old_url:
        raise HTTPException(status_code=404, detail=f"Le dataset '{old_name}' avec l'URL fournie est introuvable.")

    # Suppression de l'ancien dataset et ajout du nouveau
    del datasets[old_name]
    datasets[new_name] = {
        "name": new_name,
        "url": new_url
    }


    # Mettre à jour la structure complète avec les nouveaux datasets
    data["datasets"] = datasets  # Réinsérer les datasets dans la structure complète
    data["model"] = model  # Réinsérer les models (les conserver inchangés)

    try:
        # Écrire les modifications dans le fichier JSON
        with open(JSON_FILE_PATH, "w") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'écriture dans le fichier JSON : {e}")

    return {"message": f"Le dataset '{old_name}' a été modifié avec succès en '{new_name}'."}

def load_iris_dataset():
    # Construction du chemin sans répétition
    base_path = os.path.join("src", "data", "iris")
    
    filename = "Iris.csv"
    path = os.path.join(base_path, filename)

    # Vérification du chemin absolu
    absolute_path = os.path.abspath(path)
    print("Chemin absolu du fichier :", absolute_path)

    # Vérifier si le fichier existe
    if not os.path.exists(absolute_path):
        raise HTTPException(status_code=404, detail=f"Le fichier Iris.csv est introuvable à cet emplacement : {absolute_path}")

    # Charger le dataset avec pandas
    try:
        iris_df = pd.read_csv(absolute_path)
        # Retourner le DataFrame
        return iris_df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement du dataset : {e}")
    

def process_iris_dataset():
    # Charger le dataset en appelant la fonction
    iris_df = load_iris_dataset()  # Assurez-vous que load_iris_dataset est une fonction qui renvoie un DataFrame

    # Vérifier s'il y a des valeurs manquantes
    if iris_df.isnull().sum().any():
        iris_df = iris_df.dropna()  # Suppression des lignes avec valeurs manquantes

    # Séparer les features (X) et la cible (y)
    X = iris_df.drop(columns=["Species"])  # Supposons que 'Species' est la colonne cible
    y = iris_df["Species"]

    # Normaliser les données (Standardisation)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Retourner les données sous forme de JSON
    processed_data = {
        "features": X_scaled.tolist(),
        "target": y.tolist()
    }
    processed_dataframe = pd.DataFrame(processed_data)

    return processed_dataframe

def split_iris_dataset(test_size) : 
    processed_iris_dataset = process_iris_dataset()
    train_df, test_df = train_test_split(processed_iris_dataset, test_size=test_size, random_state=42)

    return train_df, test_df

def train_and_save(train_test_dataset, training_model):
    train_set = train_test_dataset["dataset"]["train_set"]
    test_set = train_test_dataset["dataset"]["test_set"]
    X_train = [sample["features"] for sample in train_set]
    y_train = [sample["target"] for sample in train_set]

    # Extraire les features (X) et les cibles (y) pour l'ensemble de test
    X_test = [sample["features"] for sample in test_set]
    y_test = [sample["target"] for sample in test_set]
    
    
    knn_model = KNeighborsClassifier(**training_model)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    model_save_path = "C:/Users/Le Cornec/Desktop/EPF/5A/data_sources2/API---Webscrapping/TP2 and  3/services/epf-flower-data-science/src/models/knn_model.joblib"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    try:
        joblib.dump(knn_model, model_save_path)
    except Exception as e:
        print(f"Error saving model: {e}")

    return {
        "message": "Model trained and saved successfully.",
        "accuracy": accuracy #the result is 1 because we only have few options that can become our y_pred, and our test_set is small
    }

def get_pred(train_test_dataset, training_model):
    train_set = train_test_dataset["dataset"]["train_set"]
    test_set = train_test_dataset["dataset"]["test_set"]
    X_train = [sample["features"] for sample in train_set]
    y_train = [sample["target"] for sample in train_set]

    # Extraire les features (X) et les cibles (y) pour l'ensemble de test
    X_test = [sample["features"] for sample in test_set]
    y_test = [sample["target"] for sample in test_set]
    
    
    knn_model = KNeighborsClassifier(**training_model)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    
    return y_pred
    
def get_firestore_params(collection_name, document_id) : 
    parameters = FirestoreClient.get(collection_name, document_id) 
    return parameters

def add_firestore_params(collection_name, document_id, data) : 
    FirestoreClient.add(collection_name, document_id, data) 

def update_firestore_params(collection_name, document_id, data) : 
    FirestoreClient.update(collection_name, document_id, data) 
