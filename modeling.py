import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler 
import numpy as np 
import joblib
data = pd.read_csv("cleaned_data.csv" , index_col=False)
data = data.sample(frac=1 , random_state=42).reset_index(drop=True)

print(data.corr()['price'])
numerical_columns = []
for column in data.columns :
    if len(data[column].unique() ) > 3 :
        numerical_columns.append(column)

scaler_x = StandardScaler()
scaler_y = StandardScaler()



x , y = np.array(data.drop(columns=['price'])) , np.array(data[['price']])
x_train , x_test , y_train , y_test = train_test_split(x ,y , train_size=0.8 , test_size=0.2 , shuffle=True , random_state=42 )

y_train_log = np.log(y_train)
y_test_log = np.log(y_test)

#x_train = scaler_x.fit_transform(x_train)
#y_train = scaler_y.fit_transform(y_train)
  

print(x.shape , y.shape)
print(x_train.shape , y_train.shape , x_test.shape ,y_test.shape )


lr = LinearRegression(fit_intercept=True)
lr_log = LinearRegression(fit_intercept=True)

lr.fit(x_train , y_train)
lr_log.fit(x_train , y_train_log)


columns = np.array(data.drop(columns=['price']).columns)
print(columns)
weights = lr.coef_
for _ , column in enumerate(columns) :
    print(f"{column} : {weights[0][_]} {_}")

print(f"the intercept is {lr.intercept_}")

#save the model 


joblib.dump(lr , 'house_price_lr.pkl')
joblib.dump(lr_log , 'house_price_log.pkl')
joblib.dump(scaler_x , 'scaler_for_x.pkl')
joblib.dump(scaler_y , 'scaler_for_y.pkl')
np.savez("test_data" , 
         x_test = x_test , 
         y_test = y_test , 
         y_test_log = y_test_log, 
         columns = columns)
print("Model and Scaler saved successfully")