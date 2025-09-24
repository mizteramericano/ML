from sklearn.datasets import load_wine
from sklearn.metrics import  f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = load_wine()
X = df.data # features
y = df.target # targer

# print(X)
# print(y)

# ********** train_test_split ***********
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
# ********** train_test_split ***********


# ********** DecisionTree **********
#dst = DecisionTreeClassifier(max_depth=5)
#dst.fit(X_train,y_train) # เริ่มทำแบบฝึกหัด
#pmd = dst.predict(X_test) # เริ่มทำข้อสอบ

# แสดงผลของ tree
#print("DecisionTreeClassifier")
#acc_dct = accuracy_score(y_test,pmd) # acc
#print("acc ของ DecisionTree: " , acc_dct)
#f1_dct = f1_score(y_test,pmd,average="macro") # f1
#print("f1 ของ DecisionTree : " , f1_dct)

# ********** RandomForest **********
rf = RandomForestClassifier(max_depth=5, n_estimators = 1000)
rf.fit(X_train,y_train) # เริ่มทำแบบฝึกหัด
prf = rf.predict(X_test) # เริ่มทำข้อสอบ

# แสดงผลของ RandomForest
print("RandomForestClassifierr")
acc_rf = accuracy_score(y_test,prf) # acc
print("acc ของ RandomForest: " , acc_rf)
f1_rf = f1_score(y_test,prf,average="macro") # f1
print("f1 ของ RandomForest : " , f1_rf)
