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
        with open(JSON_FILE_PATH, "r") as file:
            datasets = json.load(file)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Le fichier JSON est mal formé.")

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
    JSON_FILE_PATH = "src\config\model_parameters.json"
    
    # Vérifier l'existence du fichier JSON
    if not os.path.exists(JSON_FILE_PATH):
        raise HTTPException(status_code=404, detail=f"Fichier JSON '{JSON_FILE_PATH}' introuvable.")

    try:
        # Charger les datasets existants
        with open(JSON_FILE_PATH, "r") as file:
            datasets = json.load(file)
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

    try:
        # Écrire les modifications dans le fichier JSON
        with open(JSON_FILE_PATH, "w") as file:
            json.dump(datasets, file, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'écriture dans le fichier JSON : {e}")

    return {"message": f"Le dataset '{name}' a été ajouté avec succès."}
    
def modify_dataset(old_name, old_url, new_name, new_url):
    JSON_FILE_PATH = "src\config\model_parameters.json"
    if not os.path.exists(JSON_FILE_PATH):
        raise HTTPException(status_code=404, detail=f"Fichier JSON '{JSON_FILE_PATH}' introuvable.")

    try:
        # Charger les datasets existants
        with open(JSON_FILE_PATH, "r") as file:
            datasets = json.load(file)
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

    try:
        # Écrire les modifications dans le fichier JSON
        with open(JSON_FILE_PATH, "w") as file:
            json.dump(datasets, file, indent=4)
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
