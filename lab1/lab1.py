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
# ใช้คำสั่ง Print และคัดลอกชุดข้อมูล diabete ที่เป็น data ทั้งหมดแล้ววางลงในช่องนี้
print(X)
#ใช้คำสั่ง Print และคัดลอกชุดข้อมูล diabete ที่เป็น Target ทั้งหมดแล้ววางลงในช่องนี้
print(y)

# หา features
print(X.shape)
print(y.shape)

# แบ่ง test train
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.80)
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
model = Ridge(alpha = 2.0)
model.fit(X_train,y_train) # เรียนรู้
predict = model.predict(X_test) # ทำข้อสอบ
mse = mean_squared_error(y_test, predict) # มาดูว่าทำข้อสอบผิดมากน้อยแค่ไหน
print(mse)

