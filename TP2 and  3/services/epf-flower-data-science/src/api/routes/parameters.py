from google.cloud import firestore

def create_firestore_parameters():
    # Initialiser le client Firestore
    db = firestore.Client()

    # Créer la collection "parameters" et insérer le document "parameters"
    parameters_data = {
        "n_estimators": 100,
        "criterion": "gini"
    }

    # Ajouter le document dans la collection
    db.collection("parameters").document("parameters").set(parameters_data)
    print("Parameters document created successfully.")

if __name__ == "__main__":
    create_firestore_parameters()
