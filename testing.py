import joblib 
import numpy as np 
from sklearn.metrics import mean_squared_error , median_absolute_error , root_mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt 
lr_model = joblib.load('house_price_lr.pkl')
lr_model_log = joblib.load('house_price_log.pkl')


scaler_x = joblib.load('scaler_for_x.pkl')
scaler_y = joblib.load('scaler_for_y.pkl')


test_data = np.load("test_data.npz")


x_test = test_data["x_test"]
y_test = test_data["y_test"]
y_test_log = test_data["y_test_log"]
print(x_test.shape , y_test.shape)

#x_test_norm = scaler_x.transform(x_test)

prediction = lr_model_log.predict(x_test)
prediction_log = np.exp(prediction)


prediction_normal = lr_model.predict(x_test)


print(prediction.shape , prediction[0:5])

MAE = median_absolute_error(y_true=y_test , y_pred=prediction)
MSE = mean_squared_error(y_true=y_test , y_pred=prediction)
RMSE = root_mean_squared_error(y_true=y_test , y_pred=prediction)
R_square = r2_score(y_pred=prediction , y_true=y_test)
print(f"the MAE = {MAE}")
print(f"the rmse = {RMSE}")
print(f"the mse = {MSE}")
print(f"the r square = {R_square}")


residuals = y_test - prediction 


