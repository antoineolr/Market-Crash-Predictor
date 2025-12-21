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

SP500=load_market_data(ticker_SP500,start_date,end_date)
VIX=load_vix_data(ticker_VIX,end_date)

data=align_datasets(SP500,VIX)
log_return=compute_log_returns(data)

data=add_features(data,window_vol,momentum_window,rsi_period,volume_window)

train_data=data[split_date:]
test_data=data[:split_date]

model=create_hmm_model(n_states,covariance_type,n_iter,random_state)
model=train_hmm(train_data,model)

states=predict_states(test_data,model)
probabilities=predict_proba(test_data,model)

optimal_threshold=backtest_threshold(train_data,probabilities,thresholds,tax,initial_capital)

signals=generate_signals(probabilities,optimal_threshold)
signals=apply_smoothing(signals,smoothing_window)

metrics_bh=backtest_buy_and_hold(test_data,initial_capital)
metrics_hmm=backtest_hmm_strategy(test_data,signals,tax,initial_capital)

comparison=compare_strategies(metrics_bh,metrics_hmm)
print(comparison)



graph_hidden_states(data,states)
graph_crisis_prob(data,probabilities,optimal_threshold)
graph_trading_signals(data,signals)
graph_comparison(data,)
graph_drawdown(data,)