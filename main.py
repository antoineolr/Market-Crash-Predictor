from src.config import (ticker_SP500,ticker_VIX,start_date,end_date,split_date,
                        window_vol,rsi_period,momentum_window,volume_window,p,q,
                        n_states,covariance_type,n_iter,tol,random_state,startprob,state_names,
                        threshold,upper_threshold,lower_threshold,smoothing_window,tax,initial_capital,risk_free_rate,
                        thresholds)
from src.data_handler import (load_market_data,load_vix_data,compute_log_returns,align_datasets,preprocess_data)
from src.feature_engineering import (compute_rolling_volatility,fit_garch_model,extract_garch_volatility,compute_momentum,compute_rsi,compute_volume_standardized,compute_vix_vol_ratio,add_features)
from src.core_model import (create_hmm_model,train_hmm,predict_states,predict_proba,save_model,load_model)
from src.trading_signals import (generate_signals,generate_signals_hysteresis,apply_smoothing,add_transaction_costs,count_transactions,backtest_threshold)
from src.backtesting import (calculate_portfolio_value,calculate_metrics,backtest_buy_and_hold,backtest_hmm_strategy,compare_strategies)
from src.visualization import (graph_hidden_states,graph_crisis_prob,graph_trading_signals,graph_comparison,graph_drawdown,hist_metrics)
import numpy as np

SP500=load_market_data(ticker_SP500,start_date,end_date)
VIX=load_vix_data(start_date,end_date)

data=align_datasets(SP500,VIX)
compute_log_returns(data)

data=add_features(data)

train_data=data[:split_date]
test_data=data[split_date:]

model=create_hmm_model(n_states,covariance_type,n_iter,random_state)
model=train_hmm(train_data,model)

states=predict_states(test_data,model)

probabilities_test=predict_proba(test_data,model)
probabilities_train=predict_proba(train_data,model)

optimal_threshold=backtest_threshold(train_data,probabilities_train,thresholds,tax,initial_capital)

signals=generate_signals(probabilities_test,optimal_threshold)
signals=apply_smoothing(signals,smoothing_window)

metrics_bh, portfolio_bh=backtest_buy_and_hold(test_data,initial_capital)
metrics_hmm, portfolio_hmm=backtest_hmm_strategy(test_data,signals,tax,initial_capital)

comparison=compare_strategies(metrics_bh,metrics_hmm)
print(comparison)

running_max_bh = np.maximum.accumulate(portfolio_bh)
dd_bh = (portfolio_bh - running_max_bh) / running_max_bh

running_max_hmm = np.maximum.accumulate(portfolio_hmm)
dd_hmm = (portfolio_hmm - running_max_hmm) / running_max_hmm

graph_hidden_states(test_data,states)
graph_crisis_prob(test_data,probabilities_test,optimal_threshold)
graph_trading_signals(test_data,signals)
graph_comparison(test_data,portfolio_bh,portfolio_hmm)
graph_drawdown(test_data,dd_bh,dd_hmm)

print("Optimal threshold :", optimal_threshold)
print("Nombres de transactions :", count_transactions(signals))

