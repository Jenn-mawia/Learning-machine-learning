import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing,svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]#features
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0


df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

#compare forecast price to adjusted close price
forecast_out = int(math.ceil(0.1*len(df)))
#print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)#label for forecasted price


X = np.array(df.drop(['label'], 1))#features
X = preprocessing.scale(X)#scaling X(scale new values alongside all other values)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]


df.dropna(inplace=True)
y = np.array(df['label'])#labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# clf = LinearRegression(n_jobs=-1)
# clf.fit(X_train, y_train)
# with open('linearregression.pickle', 'wb') as f: #save the classifier(clf)
#     pickle.dump(clf, f)

#pickling. no need to train data everytime.
pickle_in = open('linearregression.pickle', 'rb') #use classifier
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
#print(accuracy)
forecast_set = clf.predict(X_lately)#forecast for the next 30 days
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

#populate the data frame with dates
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]#.doc references the index for the df.
                                                                        # next date is a timestamp
#np.nan for _ in range(len(df.columns)-1)] is a list of values that are np.nan +[i] adds forecast col.
print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)#in the 4th location
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

