# import
from skimage.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# เก็บตัวแปล load_diabetes()
diabete = load_diabetes()
# แปลง
X = diabete.data
y = diabete.target

# หา features
print(X.shape)
print(y.shape)

# แบ่ง test train
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.99)
print(X_train.shape, X_test.shape)

# เริ่ม LinearRegression
print("This is LinearRegression")
model = LinearRegression()
model.fit(X_train,y_train) # เรียนรู้
predict = model.predict(X_test) # ทำข้อสอบ
mse = mean_squared_error(y_test, predict) # มาดูว่าทำข้อสอบผิดมากน้อยแค่ไหน
print(mse)

# เริ่ม Ridge
print("This is Ridge")
model = Ridge()
model.fit(X_train,y_train) # เรียนรู้
predict = model.predict(X_test) # ทำข้อสอบ
mse = mean_squared_error(y_test, predict) # มาดูว่าทำข้อสอบผิดมากน้อยแค่ไหน
print(mse)

# เริ่ม Lasso
print("This is Lasso")
model = Lasso()
model.fit(X_train,y_train) # เรียนรู้
predict = model.predict(X_test) # ทำข้อสอบ
mse = mean_squared_error(y_test, predict) # มาดูว่าทำข้อสอบผิดมากน้อยแค่ไหน
print(mse)

# เริ่ม ElasticNet
print("This is ElasticNet")
model = ElasticNet()
model.fit(X_train,y_train) # เรียนรู้
predict = model.predict(X_test) # ทำข้อสอบ
mse = mean_squared_error(y_test, predict) # มาดูว่าทำข้อสอบผิดมากน้อยแค่ไหน
print(mse)