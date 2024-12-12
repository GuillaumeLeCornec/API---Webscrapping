import json
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

DATA_DIR = "src/data"

def download_dataset(dataset_name):
    """
    Downloads a dataset from Kaggle based on the information in a JSON file.

    Args:
        dataset_name (str): The name of the dataset to be downloaded.

    Raises:
        HTTPException: If there is an issue with the API key, JSON file, or dataset not found.

    Returns:
        dict: A dictionary containing a success message with the download path or an error message.
    """
    JSON_FILE_PATH = "src\config\model_parameters.json"
    try:
        api = KaggleApi()
        api.authenticate() 
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pas la bonne cle : {e}")
    if not os.path.exists(JSON_FILE_PATH):
        raise HTTPException(status_code=404, detail=f"Le fichier JSON '{JSON_FILE_PATH}' est introuvable.")
    try:
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
            datasets = data.get("datasets", {})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail=f"Le fichier JSON est mal formé.{test}")

    if dataset_name not in datasets:
        raise HTTPException(status_code=404, detail=f"Le dataset '{dataset_name}' n'existe pas dans le fichier JSON.")
    
    dataset_info = datasets[dataset_name]
    dataset_url = dataset_info["url"]
    try:
        os.makedirs(DATA_DIR, exist_ok=True)

        od.download(dataset_url, DATA_DIR)

        return {"message": "Dataset downloaded successfully!", "path": DATA_DIR}
    except Exception as e:
        return {"error": str(e)}
    
def add_dataset(name, url):
    """
    Adds a new dataset to the JSON configuration file.

    Args:
        name (str): The name of the dataset to be added.
        url (str): The URL of the dataset.

    Raises:
        HTTPException: If the JSON file is not found, is malformed, or if an error occurs while reading or writing the file.
        HTTPException: If the dataset already exists in the configuration.

    Returns:
        dict: A dictionary containing a success message with the name of the added dataset.
    """
    JSON_FILE_PATH = "src/config/model_parameters.json"
    
    if not os.path.exists(JSON_FILE_PATH):
        raise HTTPException(status_code=404, detail=f"Fichier JSON '{JSON_FILE_PATH}' introuvable.")
    
    try:
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)  
            datasets = data.get("datasets", {})  
            model = data.get("model", {}) 
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Le fichier JSON est mal formé.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la lecture du fichier JSON : {e}")
    
    if name in datasets:
        raise HTTPException(status_code=400, detail=f"Le dataset '{name}' existe déjà.")
    
    datasets[name] = {
        "name": name,
        "url": url
    }

    data["datasets"] = datasets  
    data["model"] = model 

    try:
        with open(JSON_FILE_PATH, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'écriture dans le fichier JSON : {e}")

    return {"message": f"Le dataset '{name}' a été ajouté avec succès."}


    
def modify_dataset(old_name, old_url, new_name, new_url):
    """
    Modifies the information of an existing dataset in the JSON configuration file.

    Args:
        old_name (str): The current name of the dataset to be modified.
        old_url (str): The current URL of the dataset to be modified.
        new_name (str): The new name of the dataset.
        new_url (str): The new URL of the dataset.

    Raises:
        HTTPException: If the JSON file is not found, is malformed, or if an error occurs while reading or writing the file.
        HTTPException: If the dataset to be modified does not exist or the provided URL does not match.

    Returns:
        dict: A dictionary containing a success message with the name of the modified dataset.
    """
    JSON_FILE_PATH = "src\config\model_parameters.json"
    if not os.path.exists(JSON_FILE_PATH):
        raise HTTPException(status_code=404, detail=f"Fichier JSON '{JSON_FILE_PATH}' introuvable.")

    try:
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as file:
            data = json.load(file) 
            datasets = data.get("datasets", {}) 
            model = data.get("model", {})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Le fichier JSON est mal formé.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la lecture du fichier JSON : {e}")

    if old_name not in datasets or datasets[old_name]["url"] != old_url:
        raise HTTPException(status_code=404, detail=f"Le dataset '{old_name}' avec l'URL fournie est introuvable.")

    del datasets[old_name]
    datasets[new_name] = {
        "name": new_name,
        "url": new_url
    }

    data["datasets"] = datasets
    data["model"] = model

    try:
        with open(JSON_FILE_PATH, "w") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'écriture dans le fichier JSON : {e}")

    return {"message": f"Le dataset '{old_name}' a été modifié avec succès en '{new_name}'."}

def load_iris_dataset():
    """
    Loads the Iris dataset from a specified file path and returns it as a pandas DataFrame.

    The dataset is located in the 'src/data/iris' directory under the filename 'Iris.csv'.

    Raises:
        HTTPException: If the dataset file is not found or if there is an error while loading the dataset.

    Returns:
        pandas.DataFrame: A DataFrame containing the Iris dataset.
    """
    base_path = os.path.join("src", "data", "iris")
    
    filename = "Iris.csv"
    path = os.path.join(base_path, filename)

    absolute_path = os.path.abspath(path)
    print("Chemin absolu du fichier :", absolute_path)

    if not os.path.exists(absolute_path):
        raise HTTPException(status_code=404, detail=f"Le fichier Iris.csv est introuvable à cet emplacement : {absolute_path}")

    try:
        iris_df = pd.read_csv(absolute_path)
        return iris_df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement du dataset : {e}")
    

def process_iris_dataset():
    """
    Processes the Iris dataset by cleaning, splitting, and normalizing the data.

    This function performs the following steps:
    1. Loads the Iris dataset using `load_iris_dataset()`.
    2. Checks and handles missing values by dropping rows with any missing data.
    3. Splits the dataset into features (X) and target (y).
    4. Normalizes the feature data using standard scaling (mean=0, variance=1).
    5. Returns a pandas DataFrame containing the processed features and target.

    Raises:
        HTTPException: If there is an issue loading or processing the dataset.

    Returns:
        pandas.DataFrame: A DataFrame containing the processed Iris dataset with features and target.
    """

    iris_df = load_iris_dataset()  

    if iris_df.isnull().sum().any():
        iris_df = iris_df.dropna()  

    X = iris_df.drop(columns=["Species"])  
    y = iris_df["Species"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    processed_data = {
        "features": X_scaled.tolist(),
        "target": y.tolist()
    }
    processed_dataframe = pd.DataFrame(processed_data)

    return processed_dataframe

def split_iris_dataset(test_size) :
    """
    Splits the processed Iris dataset into training and testing sets.

    This function takes the processed Iris dataset and splits it into two parts:
    a training set and a testing set, based on the specified test size.

    Args:
        test_size (float): The proportion of the dataset to include in the test split
                            (between 0 and 1, where 0.2 means 20% for testing).

    Raises:
        HTTPException: If there is an issue with the dataset processing.

    Returns:
        pandas.DataFrame: The training set containing the processed Iris dataset.
        pandas.DataFrame: The testing set containing the processed Iris dataset.
    """ 
    processed_iris_dataset = process_iris_dataset()
    train_df, test_df = train_test_split(processed_iris_dataset, test_size=test_size, random_state=42)

    return train_df, test_df

def train_and_save(train_test_dataset, training_model):
    """
    Trains a K-Nearest Neighbors (KNN) model using the provided training dataset and saves the trained model.

    This function splits the dataset into training and testing sets, trains a KNN model using the training set, 
    and evaluates the model's accuracy on the testing set. The trained model is then saved to the specified path.

    Args:
        train_test_dataset (dict): A dictionary containing the training and testing sets. 
                                   It should have a structure like {"dataset": {"train_set": [...], "test_set": [...]}},
                                   where each set contains samples with "features" and "target".
        training_model (dict): A dictionary of parameters to initialize the KNeighborsClassifier model, 
                               e.g., {"n_neighbors": 5, "weights": "uniform"}.

    Raises:
        HTTPException: If there is an issue with the training or saving process.

    Returns:
        dict: A dictionary containing a message indicating successful training and saving of the model,
              and the accuracy of the model on the test set.
    """
    train_set = train_test_dataset["dataset"]["train_set"]
    test_set = train_test_dataset["dataset"]["test_set"]
    X_train = [sample["features"] for sample in train_set]
    y_train = [sample["target"] for sample in train_set]

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
    """
    Trains a K-Nearest Neighbors (KNN) model on the training dataset and returns the predicted labels for the test set.

    This function trains a KNN model using the provided training dataset, then makes predictions on the test dataset.

    Args:
        train_test_dataset (dict): A dictionary containing the training and testing sets. 
                                   It should have a structure like {"dataset": {"train_set": [...], "test_set": [...]}},
                                   where each set contains samples with "features" and "target".
        training_model (dict): A dictionary of parameters to initialize the KNeighborsClassifier model, 
                               e.g., {"n_neighbors": 5, "weights": "uniform"}.

    Returns:
        list: A list of predicted labels for the test set based on the trained KNN model.
    """
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
    """
    Retrieves parameters from a Firestore document.

    This function queries a Firestore collection using the provided collection name and document ID
    to fetch the parameters stored in that document.

    Args:
        collection_name (str): The name of the Firestore collection.
        document_id (str): The unique identifier of the document in the Firestore collection.

    Returns:
        dict: A dictionary containing the parameters retrieved from the Firestore document.

    Raises:
        HTTPException: If there is an issue with fetching data from Firestore.
    """
    parameters = FirestoreClient.get(collection_name, document_id) 
    return parameters

def add_firestore_params(collection_name, document_id, data) : 
    """
    Adds parameters to a Firestore document.

    This function adds or updates a document in the specified Firestore collection with the provided
    data. If the document exists, it will be updated; if not, a new document will be created.

    Args:
        collection_name (str): The name of the Firestore collection.
        document_id (str): The unique identifier for the document within the collection.
        data (dict): A dictionary containing the parameters to be added or updated in the Firestore document.

    Raises:
        HTTPException: If there is an issue with adding the Firestore document.
    """
    FirestoreClient.add(collection_name, document_id, data) 

def update_firestore_params(collection_name, document_id, data) : 
    """
    Updates parameters in a Firestore document.

    This function updates an existing document in the specified Firestore collection with the provided
    data. If the document does not exist, an exception will be raised.

    Args:
        collection_name (str): The name of the Firestore collection.
        document_id (str): The unique identifier of the document to be updated.
        data (dict): A dictionary containing the parameters to be updated in the Firestore document.

    Returns:
        None: This function does not return any value.

    Raises:
        HTTPException: If there is an issue with updating the Firestore document, or if the document does not exist.
    """
    FirestoreClient.update(collection_name, document_id, data) 
