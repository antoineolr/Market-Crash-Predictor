import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM 
import joblib


def create_hmm_model(n_states=3,covariance_type="full",n_iter=200,random_state=42):
    model=GaussianHMM(n_components=n_states,covariance_type=covariance_type,n_iter=n_iter,random_state=random_state,tol=0.01)
    model.startprob_= np.array([0.70,0.25,0.05])
    model.state_names={
        0: "Stable",
        1: "Volatile",
        2: "Crise"
    }
    return model

def train_hmm(data,model):
    try:
        model.fit(data)
        return model
    except Exception as e:
        print(f"Erreur entrainement HMM; :{e}")
        return None

def predict_states(data,model):
    try:
        states=model.predict(data)
        return states
    except Exception as e:
        print(f"Erreur prediction :{e}")
        return None
    
def predict_proba(data,model):
    try:
        probabilities=model.predict_proba(data)
        return probabilities
    except Exception as e:
        print(f"Erreur de prediction :{e}")
        return None

def save_model(model,filepath):
    try:
        joblib.dump(model,filepath)
        print(f'Saving reussi dans {filepath}')
    except Exception as e:
        print(f"Erreur de saving :{e}")
        return None

def load_model(filepath):
    try:
        model=joblib.load(filepath)
        return model
    except Exception as e:
        print(f"Erreur de loading :{e}")
        return None
    
def get_transition_matrix(model):
    return model.transmat_