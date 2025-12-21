import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM 
import joblib
from sklearn.preprocessing import StandardScaler


def create_hmm_model(n_states=3,covariance_type="full",n_iter=200,random_state=42):
    model=GaussianHMM(n_components=n_states,covariance_type=covariance_type,n_iter=n_iter,random_state=random_state,tol=0.01,init_params='tmc')
    model.startprob_= np.array([0.70,0.25,0.05])
    model.state_names={
        0: "Stable",
        1: "Volatile",
        2: "Crise"
    }
    return model

def train_hmm(data,model):
    try:
        features = ['Rolling_Vol', 'GARCH_Vol', 'Momentum', 'RSI', 'Volume_Standardized', 'VIX_Vol_Ratio']
        X = data[features].values
        
        # Remove any inf/nan values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize features for better convergence
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model.fit(X_scaled)
        model.scaler = scaler  # Save scaler for later use
        return model
    except Exception as e:
        print(f"Erreur entrainement HMM; :{e}")
        return None

def predict_states(data,model):
    try:
        features = ['Rolling_Vol', 'GARCH_Vol', 'Momentum', 'RSI', 'Volume_Standardized', 'VIX_Vol_Ratio']
        X = data[features].values
        
        # Clean and scale using saved scaler
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = model.scaler.transform(X)
        
        states=model.predict(X_scaled)
        return states
    except Exception as e:
        print(f"Erreur prediction :{e}")
        return None
    
def predict_proba(data,model):
    try:
        features = ['Rolling_Vol', 'GARCH_Vol', 'Momentum', 'RSI', 'Volume_Standardized', 'VIX_Vol_Ratio']
        X = data[features].values
        
        # Clean and scale using saved scaler
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = model.scaler.transform(X)
        
        probabilities=model.predict_proba(X_scaled)
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