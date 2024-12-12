from google.cloud import firestore


import firebase_admin
from firebase_admin import credentials
import json

class FirestoreClient:
    """Wrapper around a database"""

    client: firestore.Client

    def __init__(self) -> None:
        """Init the client."""

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
            firebase_admin.initialize_app(cred)  
        client = firestore.Client(project = "datasources-6117d") 
        print(f"Attempting to retrieve document {document_id} from collection {collection_name}")

        doc = client.collection(collection_name).document(document_id).get()

        if doc.exists:
            print(f"Document found: {doc.to_dict()}")
            return doc.to_dict()
        else:
            print(f"Document not found at {collection_name} with the id {document_id}")
            raise FileExistsError(
                f"No document found at {collection_name} with the id {document_id}"
            )
    
    def update(collection_name: str, document_id: str, data: dict) -> None:
        """
        Update parameters in Firestore document.

        Args:
            collection_name (str): The Firestore collection name.
            document_id (str): The document ID to update.
            data (dict): The data to update in the document.

        Returns:
            None
        """
        client = firestore.Client(project = "datasources-6117d")  
        try:
            doc_ref = client.collection(collection_name).document(document_id)
            if isinstance(data, dict):  
                doc_ref.set(data)
                print(f"Document {document_id} updated in collection {collection_name}")
            else : 
                data = json.loads(data)
                doc_ref.set(data)
        except Exception as e:
            print(f"Error updating document: {e}")
            raise


    def add(collection_name: str, document_id: str, data: dict) -> None:
        """
        Add or update a document in Firestore.

        Args:
            collection_name (str): The Firestore collection name.
            document_id (str): The document ID to add or update.
            data (dict): The data to add or update in the document.

        Returns:
            None
        """
        client = firestore.Client(project="datasources-6117d")  
        try:
            doc_ref = client.collection(collection_name).document(document_id)

            if isinstance(data, str):
                try:
                    data = json.loads(data)  
                except json.JSONDecodeError:
                    raise ValueError(f"Data {data} is a string but not a valid JSON.")

            if not isinstance(data, dict):
                raise ValueError(f"Data {data} must be a dictionary")

            existing_doc = doc_ref.get()
            if existing_doc.exists:
                existing_data = existing_doc.to_dict()
                existing_data.update(data)  

                doc_ref.set(existing_data)
                print(f"Document {document_id} updated with new data in collection {collection_name}")
            else:
                doc_ref.set(data)
                print(f"Document {document_id} added in collection {collection_name}")

        except Exception as e:
            print(f"Error adding or updating document: {e}")
            raise