import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pandas as pd
from dataprep.eda import create_report
from dataprep.datasets import load_dataset
from dataprep.eda import plot
import webbrowser

data = 'C:/Users/YK/PycharmProjects/Py/nvidia stock price/nvidia.csv'

#__dataset__
dataset = pd.read_csv('C:/Users/YK/PycharmProjects/Py/nvidia stock price/nvidia.csv')
#print(dataset)
df = pd.DataFrame(dataset)
print(df.count())

# using autoMl algorithm to create a eda report   

report = create_report(df)
report.save('C:/Users/YK/PycharmProjects/Py/nvidia stock price/report.html') 
webbrowser.open_new_tab('C:/Users/YK/PycharmProjects/Py/nvidia stock price/report.html')


# Manual EDA
df = df.drop(columns=['Stock Splits','Dividends','Volume'])

print(df.tail())
date,open,high,low,close = np.array(df['Date']),np.array(df['Open']),np.array(df['High']),np.array(df['Low']),np.array(df['Close'])
date = date[5200:5700]
open = open[5200:5700]
high = high[5200:5700]
low = low[5200:5700]
close = close[5200:5700]

d = {'date':date,'open':open,'high':high,'low':low,'close':close}
df1 = pd.DataFrame(d)
print(df1)



x = open
y = close


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.50,random_state=17,shuffle=True)



x_train = x_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)
y_test = y_test.reshape(-1,1)


model = linear_model.LinearRegression()

model.fit(x_train,y_train)
pred = model.predict(x_test)
#print(pred)


#To retrieve the intercept:
print("Intercept_value",model.intercept_)

#For retrieving the slope:
print("coefficient",model.coef_)


df2 = pd.DataFrame({'Actual': x_test.flatten(), 'Predicted': pred.flatten()})
print(df2)

#_Error_Calculation_
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


acc = model.score(y_test,pred)
print("Model Accuracy:%.2f"%(acc*100),"%")



plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,pred,color='blue')
plt.show()

df3 = df2.head(25)
df3.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.25', color='red')
plt.grid(which='minor', linestyle=':', linewidth='0.25', color='blue')
plt.show()

#Testing
print("Testing_data:")
e1 = [6.22],[54.22],[88.44],[78.21],[99.0],[445.00],[74.11]
pred2 = model.predict(e1)
print(pred2)





