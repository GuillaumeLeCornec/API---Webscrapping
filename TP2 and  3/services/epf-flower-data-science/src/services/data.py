# import uvicorn
# import os
import json
# import requests
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import RedirectResponse
import opendatasets as od



from fastapi import FastAPI, HTTPException

from fastapi import APIRouter
import os
import opendatasets as od

router = APIRouter()

# Define the directory to save the dataset
DATA_DIR = "src/data"

# Create a route to download the dataset
def download_dataset(dataset_name):
    JSON_FILE_PATH = "src\config\model_parameters.json"

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
