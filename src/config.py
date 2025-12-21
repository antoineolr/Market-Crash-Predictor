import numpy as np

ticker_SP500="^GSPC"
ticker_VIX="^VIX"
start_date="2000-01-01"
end_date="2024-12-31"
split_date="2018-12-31" 


window_vol = 20 ### jours
rsi_period = 14 ###jours
momentum_window = 20 ###jours
volume_window = 60 ###jours
p=1
q=1 ###GARCH(1,1))

n_states = 3 ###Stable, Volatile, Crise
covariance_type = "full"
n_iter = 200
tol = 0.01 
random_state = 42 
startprob = [0.70, 0.25, 0.05]
state_names = {0: "Stable", 1: "Volatile", 2: "Crise"}

threshold = 0.3 ###pour P(Crise)
upper_threshold = 0.3
lower_threshold = 0.15
smoothing_window = 5 ###jours
tax = 0.001 ### 0.1% / transactions
initial_capital = 10000 ### $ ou €
risk_free_rate = 0.02  ### 2% annuel

thresholds = np.arange(0.1, 0.6, 0.05)  # Test lower thresholds: 0.1, 0.15, 0.2, ..., 0.55 

output_dir = "data/"
models_dir = "models/"
plots_dir = "plots/"