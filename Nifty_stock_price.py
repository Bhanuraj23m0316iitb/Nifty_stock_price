#!/usr/bin/env python
# coding: utf-8

# # **Project: Stock Market Forecasting**

# ## **Importing libraries**

# In[120]:


import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ## **Data Collection**

# In[121]:


df = yf.download(tickers= '^NSEI', start= '2013-03-01', end="2023-02-28")
df


# ## **EDA**

# In[122]:


df.columns                      #shows the name of the coulmns/variables present in data


# In[123]:


df.shape                        #shape of the stock data


# In[124]:


df.dtypes                       #data types of Stock data


# In[125]:


df.describe()                   # describe the data in terms of total no of values(counts), means of the variables(mean), std, min, max etc.


# In[126]:


df.info()                        #provides information related to total rows(enteries) & columns,index of columns, total non-null count and data types.  


# #### **Open vs year**

# In[129]:


plt.figure(figsize = (20,10))
plt.plot(df['Open'])
plt.title('Open Price')
plt.xlabel('Year')
plt.ylabel('Price')
plt.show()


# #### **Open and Close**

# In[130]:


plt.figure(figsize = (30,10))
plt.plot(df['Open'])
plt.plot(df['Close'])
plt.title('Nifty50 Open and Close Prices')
plt.xlabel('Year')
plt.ylabel('Price')
plt.legend(['Open', 'Close'])
plt.show()


# #### **High and Low**

# In[131]:


plt.figure(figsize = (20,10))
plt.plot(df['High'])
plt.plot(df['Low'])
plt.title('Nifty50 High and Low Prices')
plt.xlabel('Year')
plt.ylabel('Price')
plt.legend(['High', 'Low'])
plt.show()


# #### **Pair Plot**

# In[132]:


sns.pairplot(df, palette='Blues')


# ## **Technical Indicators as features**

# In[133]:


df["SMA_20"] = df['Open'].rolling(window=20).mean()              #Simple Moving Average with a window of 20 days
df["SMA_50"] = df['Open'].rolling(window=50).mean()              #Simple Moving Avearge with a window of 50 days
df['EMA_5'] = df['Open'].ewm(span=20, adjust=False).mean()       #Exponential Moving Average with a span of 20 days


# In[134]:


df['Returns'] = df['Open'].pct_change()                           #Calculates the daily returns by taking the percentage change of 'Adj Close' column       


# In[135]:


df['Volatility'] = df['Returns'].rolling(window=20).std()               #Calculates the 20-day rolling standard deviation of 'Returns'.


# In[136]:


import talib


# In[137]:


df['RSI'] = talib.RSI(df['Open'], timeperiod=14)                   #Calculates the Relative Strength Index (RSI) using the 'talib' library with a time period of 14

#Calculates the Slow Stochastic Oscillator using the 'talib' library with a SlowK period of 14 and a SlowD period of 3.
df['Stoch_slowk'], df['Stoch_slowd'] = talib.STOCH(df['High'], df['Low'], df['Open'], slowk_period=14, slowd_period=3)

#Calculates the Moving Average Convergence Divergence (MACD) using the 'talib' library with a fast period of 12, a slow period of 26, and a signal period of 9.
df['Macd'], df['Macd_signal'], df['Macd_hist'] = talib.MACD(df['Open'], fastperiod=12, slowperiod=26, signalperiod=9)


# In[138]:


df


# In[139]:


df =df.drop('Close', axis=1)


# In[140]:


df


# #### **Correlation Matrix**

# In[141]:


corr_matrix = df.corr()       #corr() function find the correlation between each pairs of protein data variables
corr_matrix


# #### **Correlation Heat Map**

# In[142]:


import seaborn as sns              
plt.figure(figsize=(30,10))
plt.figure(figsize=(8,6))
sns.set_context('paper', font_scale=1.4)
sns.heatmap(corr_matrix,cmap='YlGnBu')          #heatmap of corr_matrix having yellow(low corr.),green(moderate) and blue(high) colour
plt.show()      


# In[143]:


cor_matrix = corr_matrix.abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))

# Find features with high correlation and drop them
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
print(to_drop)

df.drop(to_drop, axis=1, inplace=True)


# In[144]:


df


# In[145]:


df.columns


# #### **Imputing Missing Variables**

# In[146]:


from sklearn.experimental import enable_iterative_imputer  #import libraries
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=50, random_state=0)   
df_imputed = imputer.fit_transform(df)                    #estimate the missing values
df = pd.DataFrame(df_imputed, columns=df.columns, index=df.index) #frame a dataset including estimates of missing values
df  


# ## **Scaling and Normalization**

# In[147]:


scaler = StandardScaler()
df_std = scaler.fit_transform(df)

# # Normalize the data using min-max scaling method
min_max_scaler = MinMaxScaler()
df = pd.DataFrame(min_max_scaler.fit_transform(df_std), columns=df.columns, index=df.index)

# # Print the preprocessed data
df


# #### **Split the data in fetaures and target** 

# In[148]:


features = df.drop(['Open'], axis=1)                         
target = df['Open']                                  


# In[149]:


features


# In[150]:


target


# ## **Ridge Regression**

# #### **Without GridSearch**

# In[151]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)


alpha = 0.03

# Training a ridge regression model with the specified hyperparameters
ridge = Ridge(alpha=alpha)
ridge.fit(X_train, y_train)

# Predictions on the test set
y_pred = ridge.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
nrmse = rmse / (y_test.max() - y_test.min())
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("MAE:", mae)
print("RMSE:", rmse)
print("NRMSE:", nrmse)
print("R2 score:", r2)


# #### **y_test vs y_pred**

# In[152]:


import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True Values vs. Predictions")
plt.show()


# #### **With GridSearch**

# In[154]:


from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# Range of hyperparameters to search
alpha = np.logspace(-6, 6, 100)

# Cross-validation strategy
cv = TimeSeriesSplit(n_splits=5)

# Perform grid search to find the best hyperparameters
ridge_cv = RidgeCV(alphas=alpha, cv=cv)
ridge_cv.fit(X_train, y_train)
best_alpha = ridge_cv.alpha_

# Train a ridge regression model with the best hyperparameters
ridge_best = Ridge(alpha=best_alpha)
ridge_best.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ridge_best.predict(X_test)

 # Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
nrmse = rmse / (y_test.max() - y_test.min())
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("MAE:", mae)
print("RMSE:", rmse)
print("NRMSE:", nrmse)
print("R2 score:", r2)


# In[155]:


import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True Values vs. Predictions")
plt.show()


# In[156]:


import matplotlib.pyplot as plt

# Plot the actual and predicted values of the target variable against time
plt.figure(figsize=(30, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Ridge Regression Model')
plt.legend()
plt.show()


# In[157]:


import matplotlib.pyplot as plt

# Plot the actual and predicted values of the target variable against time
plt.figure(figsize=(10, 6))
plt.plot(target.index, target, label='Actual')
plt.plot(target.index, ridge_best.predict(features), label='Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Ridge Regression Model')
plt.legend()
plt.show()


# ## **Support Vector Regression**

# #### **Without GridSearch**

# In[158]:


from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# Without grid search
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)

y_pred = svr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
nrmse = rmse / (y_test.max() - y_test.min())
r2 = r2_score(y_test, y_pred)
print("SVR without grid search:")
print("MAE:", mae)
print("RMSE:", rmse)
print("NRMSE:", nrmse)
print("R2 Score:", r2)


# In[159]:


import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True Values vs. Predictions")
plt.show()


# #### **With GridSearch**

# In[160]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# Define parameter grid for grid search
param_grid = {'C': [0.1, 1, 10, 100], 
              'gamma': [0.1, 1, 10, 100], 
              'epsilon': [0.1, 0.5, 1]}

# Define SVR model with RBF kernel
svr = SVR(kernel='rbf')

# Define grid search over hyperparameters
grid_search = GridSearchCV(svr, param_grid=param_grid, cv=5, n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and corresponding score
print("Best hyperparameters for SVR:", grid_search.best_params_)
print("Best score for SVR:", grid_search.best_score_)

# Use the best model to make predictions on the test data
best_svr = grid_search.best_estimator_
y_pred = best_svr.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
nrmse = rmse / (y_test.max() - y_test.min())
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("MAE:", mae)
print("RMSE:", rmse)
print("NRMSE:", nrmse)
print("R2 score:", r2)


# In[161]:


import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True Values vs. Predictions")
plt.show()


# #### **Feature Importance of SVR**

# In[162]:


from sklearn.inspection import permutation_importance

# Create an SVR model with the desired hyperparameters
svr = SVR(kernel='rbf', C=10, epsilon=0.1,gamma=0.1)

# Fit the model to the training data
svr.fit(X_train, y_train)

# Get the feature importance using permutation feature importance
result = permutation_importance(svr, X_test, y_test, n_repeats=10, random_state=42)

# Plot the feature importance
importance = result.importances_mean
indices = importance.argsort()[::-1]

fig=plt.figure(figsize=(20,10))
plt.bar(range(X_test.shape[1]), importance[indices])
plt.xticks(range(X_test.shape[1]), X_test.columns[indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()


# In[163]:


import matplotlib.pyplot as plt

# Plot the actual and predicted values of the target variable against time
plt.figure(figsize=(30, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Support Vector Regression Model')
plt.legend()
plt.show()


# In[165]:


import matplotlib.pyplot as plt

# Plot the actual and predicted values of the target variable against time
plt.figure(figsize=(10, 6))
plt.plot(target.index, target, label='Actual')
plt.plot(target.index, best_svr.predict(features), label='Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('SVR Model')
plt.legend()
plt.show()


# ## **Random Forest Regressor**

# #### **Without GridSearch**

# In[166]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# Initialize a random forest regression model
rf = RandomForestRegressor(n_estimators=100, max_depth=10, max_features='sqrt')

# Fit the model to the training data
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
nrmse = rmse / (y_test.max() - y_test.min())
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Normalized Root Mean Squared Error: {nrmse:.2f}")
print(f"R^2 Score: {r2:.2f}")


# In[167]:


import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True Values vs. Predictions")
plt.show()


# #### **With GridSearch**

# In[168]:


#import randomforestclassifier from sklearn.ensemble
from sklearn.ensemble import RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

#parameter grid to search over
param_grid = {'max_depth': [2, 6, 8, 10, 20], 'max_features': ['sqrt', 'log2', None]}

cv = TimeSeriesSplit(n_splits=5)

#the random forest model
rf = RandomForestRegressor(n_estimators=100)


# use GridSearchCV to find the best hyperparameters
rf_grid_search = GridSearchCV(rf, param_grid, cv=cv, n_jobs=-1)

rf_grid_search.fit(X_train, y_train)


# print the best hyperparameters
print("Best hyperparameters for random forest genotype: ", rf_grid_search.best_params_)


#Print best score corresponding to hyperparameters
print("Best score for random forest:", rf_grid_search.best_score_)


# Fit the model with the best hyperparameters
rf_model = rf_grid_search.best_estimator_
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
nrmse = rmse / (y_test.max() - y_test.min())
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Normalized Root Mean Squared Error: {nrmse:.2f}")
print(f"R^2 Score: {r2:.2f}")


# In[170]:


import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True Values vs. Predictions")
plt.show()


# #### **Feature Importance of RF**

# In[171]:


from sklearn.ensemble import RandomForestRegressor

#random forest classifier
rf = RandomForestRegressor(n_estimators=100, max_depth=20, max_features=None)
rf.fit(X_train, y_train)

# Sort the feature importances in descending order
sort_indices = rf.feature_importances_.argsort()[::-1]

#plot the feature importances
fig=plt.figure(figsize=(20,20))
plt.bar(range(X_train.shape[1]), rf.feature_importances_[sort_indices])
plt.xticks(range(X_train.shape[1]), X_train.columns[sort_indices], rotation=90)
plt.xlabel("Feature Importance")
plt.ylabel("Importance Score")
plt.show()


# #### **Actual (y_test) vs Predicted (y_pred)**

# In[172]:


import matplotlib.pyplot as plt
y_pred = rf_model.predict(X_test)

# Plot the actual and predicted values of the target variable against time
plt.figure(figsize=(30, 10))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Random Forest Regression Model')
plt.legend()
plt.show()


# #### Actual vs Predicted 

# In[173]:


import matplotlib.pyplot as plt

# Plot the actual and predicted values of the target variable against time
plt.figure(figsize=(30, 10))
plt.plot(target.index, target, label='Actual')
plt.plot(target.index, best_svr.predict(features), label='Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('RFR Model')
plt.legend()
plt.show()


# ## **LSTM Model**

# In[174]:


import tensorflow.keras as keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Creating feature and target datasets for LSTM
X, y = [], []
for i in range(30, features.shape[0]):
    X.append(features.iloc[i-30:i, :])  # Selecting columns from 'Volume' to 'Macd_hist'
    y.append(target.iloc[i])        # Selecting the 'adj close' column

X, y = np.array(X), np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()

# 1st LSTM layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

# 2nd LSTM layer
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

# 3rd LSTM layer
model.add(LSTM(units=50))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(units=1))

optimiser = Adam(learning_rate=0.001)

model.compile(optimizer=optimiser, loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=64)
model.summary()

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
nrmse = rmse / (y_test.max() - y_test.min())
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("MAE:", mae)
print("RMSE:", rmse)
print("NRMSE:", nrmse)
print("R-squared:", r2)


# In[175]:


#### **y_test vs y_pred**


# In[176]:


import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True Values vs. Predictions")
plt.show()


# In[177]:


import matplotlib.pyplot as plt

# Plot the actual and predicted values of the target variable against time
plt.figure(figsize=(30, 10))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Support Vector Regression Model')
plt.legend()
plt.show()


# ## **References**

# Ray han 2023, How to drop out highly correlated features in Python? [source code]:https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python
# 
# Pedregosa, F. et al., 2011. Scikit-learn: Machine learning in Python. Journal of machine learning research.
# 
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# 

# In[ ]:





# In[ ]:




