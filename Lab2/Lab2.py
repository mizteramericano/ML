from nltk import accuracy
from sklearn import svm # svc = svm.SVC()
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# datasets
df = load_breast_cancer()
X = df.data # features
y = df.target # target

# print(y)
# print(df.feature_names)
# print(df.target_names)


# a. แยกข้อมูล Train 80% และ Test 20% (ใช้ train_test_split)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# b. Fit model โดยใช้ Kernel rbf
clf = SVC() # ถ้า default เป็น  kernel='rbf' อยู่แล้ว ก็เป็น clf = svm.SVC() ก็ได้
clf.fit(X_train,y_train) # เรียนรู้
predict = clf.predict(X_test) #ทำข้อสอบ

# print(y_test) # Ground Truth (Target)
# print(predict) # SVM ที่ใช้ rbf เป็น kernel

for i in range(len(predict)):
    print(i+1,'\t' , y_test[i] ,'\t',predict[i])

acc = accuracy_score(y_test,predict)
pre = precision_score(y_test,predict)
recall = recall_score(y_test,predict)
f1 = f1_score(y_test,predict)
print("acc : ", acc)
print("precision : " , pre)
print("recall : ", recall)
print("f1 : " , f1)

# Confusion matrix
con = confusion_matrix(y_test,predict)
print(con)
