# --------------
# Import Libraries
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Code starts here
df = pd.read_csv(path)
print(df.head)
df.columns = df.columns.str.lower().str.replace(" ", "_")
df= df.fillna(np.nan)
# Code ends here


# --------------
from sklearn.model_selection import train_test_split
df.set_index(keys='serial_number',inplace=True,drop=True)


# Code starts
df.iloc[:, 2:4] = df.iloc[:, 2:4].apply(pd.to_datetime, errors='coerce')

X = df.iloc[:, 0:len(df.columns) -1]
y = df["2016_deposits"]

X_train,X_val,y_train,y_val = train_test_split(X, y, test_size = 0.25, random_state = 3)
# Code ends here


# --------------
# time_col = X_train.select_dtypes(exclude=[np.number,'O']).columns
time_col = ['established_date', 'acquired_date']

# Code starts here
new_col_name = ["since_"+x for x in time_col ]
for i in range(0, len(time_col)):
    X_train[new_col_name[i]] = (pd.datetime.now() - X_train[time_col[i]])/np.timedelta64(1,'Y')
    X_val[new_col_name[i]] = (pd.datetime.now() - X_val[time_col[i]])/np.timedelta64(1,'Y')

    X_train.drop(time_col[i], axis=1, inplace=True)
    X_val.drop(time_col[i], axis=1, inplace=True)


# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
cat = X_train.select_dtypes(include='O').columns.tolist()

# Code starts here
X_train.fillna(0, inplace=True)
X_val.fillna(0, inplace=True)

le = LabelEncoder()
for i in cat:
    X_train[i] = le.fit_transform(X_train[i])
    X_val[i] = le.fit_transform(X_val[i])

X_train_temp = pd.get_dummies(data=X_train, columns=cat)
X_val_temp = pd.get_dummies(data=X_val, columns=cat)
# Code ends here


# --------------
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Code starts here
# create a regressor object 
dt = DecisionTreeRegressor(random_state = 5)

# fill y_train null values with zero
y_train.fillna(0, inplace=True)
# fit the regressor with X data 
dt.fit(X_train, y_train)
accuracy = dt.score(X_val, y_val)

y_pred = dt.predict(X_val)
rmse = np.sqrt( mean_squared_error(y_val, y_pred))



# --------------
from xgboost import XGBRegressor


# Code starts here
xgb = XGBRegressor(max_depth=50, learning_rate=0.83, n_estimators=100)

# fit the regressor with X data 
xgb.fit(X_train, y_train)
accuracy = xgb.score(X_val, y_val)

y_pred = xgb.predict(X_val)
rmse = np.sqrt( mean_squared_error(y_val, y_pred))
# Code ends here


