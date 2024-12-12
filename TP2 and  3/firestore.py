import google.auth
from google.cloud import firestore

import requests
import subprocess
import sys 
import os
import firebase_admin
from firebase_admin import credentials
import json


# from firebase_admin import credentials, auth, firestore
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import firestore_cred_email_pwd
class FirestoreClient:
    """Wrapper around a database"""

    client: firestore.Client

    def __init__(self) -> None:
        """Init the client."""
        # cred = credentials.Certificate('firestore_cred.json')
        # self.client = firestore.Client(credentials=cred)

    @staticmethod
    def get(collection_name: str, document_id: str) -> dict:
        """Find one document by ID.
        Args:
            collection_name: The collection name
            document_id: The document id
        Return:
            Document value.
        """

        cred = credentials.Certificate('firestore_cred.json')
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)  # Initialisation de Firebase avec les credentials
        client = firestore.Client(project = "datasources-6117d")  # Création du client Firestore
        print(f"Attempting to retrieve document {document_id} from collection {collection_name}")
        # You must use a global or instance client here
        # credentials, _ = google.auth.default()
        # client = firestore.Client(credentials=credentials)
        doc = client.collection(collection_name).document(document_id).get()

        if doc.exists:
            print(f"Document found: {doc.to_dict()}")
            return doc.to_dict()
        else:
            print(f"Document not found at {collection_name} with the id {document_id}")
            raise FileExistsError(
                f"No document found at {collection_name} with the id {document_id}"
            )
    
    # @staticmethod
    def update(collection_name: str, document_id: str, data: dict) -> None:
        """Update new parameters to Firestore."""
        client = firestore.Client(project = "datasources-6117d")  # Création du client Firestore
        try:
            doc_ref = client.collection(collection_name).document(document_id)
            if isinstance(data, dict):  # Ensure that data is a dictionary
                doc_ref.set(data)
                print(f"Document {document_id} updated in collection {collection_name}")
            else : 
                # raise ValueError(f"Data   {type(data)}   must be a dictionary")
                data = json.loads(data)
                doc_ref.set(data)
        except Exception as e:
            print(f"Error updating document: {e}")
            raise


    def add(collection_name: str, document_id: str, data: dict) -> None:
        """Add a new document to Firestore, merging new data with existing document data."""
        client = firestore.Client(project="datasources-6117d")  # Création du client Firestore
        try:
            doc_ref = client.collection(collection_name).document(document_id)

            # Si data est une chaîne, tentez de la convertir en dictionnaire
            if isinstance(data, str):
                try:
                    data = json.loads(data)  # Convertir la chaîne JSON en dictionnaire
                except json.JSONDecodeError:
                    raise ValueError(f"Data {data} is a string but not a valid JSON.")

            # Assurez-vous que data est un dictionnaire
            if not isinstance(data, dict):
                raise ValueError(f"Data {data} must be a dictionary")

            # Récupérer les données existantes du document
            existing_doc = doc_ref.get()
            if existing_doc.exists:
                # Fusionner les anciennes et nouvelles données
                existing_data = existing_doc.to_dict()
                # Mettre à jour les anciens paramètres avec les nouveaux
                existing_data.update(data)  # Ajoute ou remplace les clés existantes

                # Mettre à jour le document avec la fusion des anciennes et nouvelles données
                doc_ref.set(existing_data)
                print(f"Document {document_id} updated with new data in collection {collection_name}")
            else:
                # Si le document n'existe pas, ajoutez le document avec les nouvelles données
                doc_ref.set(data)
                print(f"Document {document_id} added in collection {collection_name}")

        except Exception as e:
            print(f"Error adding or updating document: {e}")
            raise

            
    

    # def get_para(self, collection_name, document_id):


    #     # Récupérer le jeton OAuth 2.0
    #     token = subprocess.check_output(["gcloud", "auth", "application-default", "print-access-token"]).strip()

    #     # Définir l'URL
    #     project_id = "datasources-6117d"
    #     url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents/{collection_name}/{document_id}"

    #     # Effectuer la requête GET
    #     headers = {"Authorization": f"Bearer {token.decode()}"}
    #     response = requests.get(url, headers=headers)

    #     # Vérifier et afficher les résultats
    #     if response.status_code == 200:
    #         data = response.json()
    #         fields = data.get("fields", {})
    #         n_estimators = fields["n_estimators"]["integerValue"]
    #         criterion = fields["criterion"]["stringValue"]
    #         print(f"n_estimators: {n_estimators}, criterion: {criterion}")
    #     else:
    #         print(f"Error: {response.status_code}, {response.text}")

    # def test():
        

    #     # Initialiser l'application Firebase
    #     cred = credentials.Certificate('TP2 and  3/firestore_cred.json')
    #     firebase_admin.initialize_app(cred)

    #     # Authentifier un utilisateur
    #     email = "guillaume.lecornec@epfedu.fr"
    #     # password = firestore_cred_email_pwd.password

    #     try:
    #         user = auth.get_user_by_email(email)
    #         print(f"User ID: {user.uid}")
    #     except firebase_admin.auth.UserNotFoundError:
    #         print("User not found, you might need to create it.")

    #     # Accéder à Firestore avec l'utilisateur authentifié
    #     db = firestore.client()
    #     doc_ref = db.collection("parameters").document("parameters")
    #     doc = doc_ref.get()
    #     if doc.exists:
    #         print(doc.to_dict())
    #     else:
    #         print("Document not found.")
