
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# dataset
# 1.ใช้ Neural Network (load_breast_cancer)
df = load_breast_cancer()
X = df.data # feature
y = df.target # answer

# a.แยกข้อมูล Train 80% และ Test 20%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
# fit ข้อมูล
clf = MLPClassifier(max_iter=1000, hidden_layer_sizes=(32,32,32,32,32,32,)) # 32 คือจำนวน node ใน hidden layer ในแต่ละ layer
clf.fit(X_train,y_train) # เรียนรู้

predict = clf.predict(X_test) #ทำข้อสอบ


# c.สร้าง Confusion matrix / Accuracy / F1-score
# Accuracy
acc = accuracy_score(y_test,predict)
print("acc : " , acc)
# F1 score
f1 = f1_score(y_test,predict,average="macro")
print("F1 score : " , f1)
# Confusion matrix
con = confusion_matrix(y_test,predict)
print(con)
