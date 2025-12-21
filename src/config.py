ticker_SP500="^GSPC"
ticker_VIX="^VIX"
start_date="2000-01-01"
end_date="2024-12-31"
split_date="2018-12-31" 


window_vol = 20 (jours)
rsi_period = 14
momentum_window = 20
volume_window = 60
p=1, q=1 ###GARCH(1,1))

n_states = 3 ###Stable, Volatile, Crise
covariance_type = "full"
n_iter = 200
tol = 0.01
random_state = 42 
startprob = [0.70, 0.25, 0.05]
state_names = {0: "Stable", 1: "Volatile", 2: "Crise"}

threshold = 0.7 ###pour P(Crise)
upper_threshold = 0.7, lower_threshold = 0.4
smoothing_window = 5 
tax = 0.001 
initial_capital = 10000 
risk_free_rate = 0.02 

thresholds = np.arange(0.5, 0.95, 0.05) 

output_dir = "data/"
models_dir = "models/"
plots_dir = "plots/"