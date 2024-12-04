import firebase_admin
from firebase_admin import credentials

cred = credentials.Certificate("ServiceKey.json")
firebase_admin.initialize_app(cred)
