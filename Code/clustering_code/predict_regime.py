from hmmlearn.hmm import GaussianHMM
import pandas as pd
from datetime import datetime, timedelta
import pickle
import matplotlib.pyplot as plt

backtest_start = datetime.strptime('2022-01-01', '%Y-%m-%d')
backtest_end = datetime.strptime('2023-01-01', '%Y-%m-%d')

train_start = datetime.strptime('2013-01-01', '%Y-%m-%d')
train_end = datetime.strptime('2021-12-31', '%Y-%m-%d')

def get_data(company, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_data = company[(pd.to_datetime(company['Date']) >= start_date) & (pd.to_datetime(company['Date']) <= end_date)].copy()
    return filtered_data

nifty50index = pd.read_csv("Dataset/nifty50Index.csv")
nifty50index = nifty50index.dropna()

hmm_model = GaussianHMM(n_components=2, n_iter=1000)
df = get_data(nifty50index, train_start, train_end)
X = df['Weekly returns'].values.reshape(-1, 1)
hmm_model.fit(X)
df.loc[:, 'Regime'] = hmm_model.predict(X)

# def get_regimes_and_retrain(hmm_model, data, n_iter=100):
    # new_hmm_model = GaussianHMM(n_components=hmm_model.n_components, covariance_type=hmm_model.covariance_type, n_iter=n_iter)
    # new_hmm_model.set_params()

    # X = data['Weekly returns'].values.reshape(-1, 1)
    # data['Regime'] = hmm_model.predict(X)  
    # new_hmm_model.fit(X)

    # return data, new_hmm_model

curr_window_start = backtest_start
curr_window_end = curr_window_start + timedelta(days=7)
merged_data = df  # Initialize an empty DataFrame to store merged data

while curr_window_end <= backtest_end:
    nifty_data = get_data(nifty50index, curr_window_start, curr_window_end)
    # nifty_with_regimes, hmm_model = get_regimes_and_retrain(hmm_model, nifty_data)
    predictions = hmm_model.predict(nifty_data['Weekly returns'].values.reshape(-1,1))
    nifty_data.loc[:,'Regime'] = predictions
    merged_data = pd.concat([merged_data, nifty_data], ignore_index=True)
    
    curr_window_start += timedelta(days=7)
    curr_window_end += timedelta(days=7)

merged_data['Date']= pd.to_datetime(merged_data['Date'])
merged_data.to_csv("Dataset/Regime_prediction.csv")

with open("model/hmm_model.pkl", "wb") as f:
    pickle.dump(hmm_model, f)

plt.figure(figsize=(10, 6))
merged_data_subset = merged_data[(merged_data['Date'] >= backtest_start) & (merged_data['Date'] <= backtest_end)]

plt.figure(figsize=(10, 6))
plt.plot(merged_data_subset['Date'], merged_data_subset['Weekly returns'], color='black', label='Weekly Returns')
for regime in merged_data_subset['Regime'].unique():
    regime_data = merged_data_subset[merged_data_subset['Regime'] == regime]
    plt.scatter(regime_data['Date'], regime_data['Weekly returns'], label=f'Regime {regime}', alpha=0.5)
plt.title('Weekly Returns and Regimes Over Time (2019-2022)')
plt.xlabel('Date')
plt.ylabel('Weekly Returns')
plt.legend()
plt.grid(True)
plt.savefig('Pictures/regimes_precitions_plot.png')
plt.show()
