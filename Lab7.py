from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline



data = load_breast_cancer()
X = data.data
y = data.target
f = data.feature_names

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# ***************************** DecisionTreeClassifier ********************
dst = DecisionTreeClassifier()
dst.fit(X_train,y_train) # เริ่มทำแบบฝึกหัด
pdst = dst.predict(X_test) # เริ่มทำข้อสอบ

# print("DecisionTreeClassifier : ")
acc_dst = accuracy_score(y_test,pdst) # acc
# print("acc ของ DecisionTreeClassifier: " , acc_dst)
pcs_dst = precision_score(y_test,pdst) # pcs
# print("precision ของ DecisionTreeClassifier: " , pcs_dst)
rc_dst = recall_score(y_test,pdst) # recall
# print("recall ของ DecisionTreeClassifier: " , rc_dst)


# ***************************** MLP ************************************
mlp = MLPClassifier(hidden_layer_sizes=(50,))
mlp.fit(X_train,y_train) # เริ่มทำแบบฝึกหัด
p_mlp = mlp.predict(X_test) # เริ่มทำข้อสอบ
# print("MLPClassifier: ")
acc_mlp = accuracy_score(y_test,p_mlp) # acc
# print("acc ของ MLPClassifier: " , acc_mlp)
pcs_mlp = precision_score(y_test,p_mlp) # pcs
# print("precision ของ MLPClassifier: " , pcs_mlp)
rc_mlp = recall_score(y_test,p_mlp) # recall
# print("recall_score ของ MLPClassifier: " , rc_mlp)

# ***************************** MLP + ANOVA *************************************
anv = SelectKBest(f_classif , k = 15)
anv_sum = make_pipeline(anv , mlp)
anv_sum.fit(X_train,y_train) # เริ่มทำแบบฝึกหัด
p_anv = anv_sum.predict(X_test) # เริ่มทำข้อสอบ
print("MLPC +  ANOVA: ")
acc_anv = accuracy_score(y_test,p_anv) # acc
print("acc ของ MLPC +  ANOVA: " , acc_anv)
pcs_anv = precision_score(y_test,p_anv) # pcs
print("precision ของ MLPC +  ANOVA: " , pcs_anv )
rc_anv = recall_score(y_test,p_anv) # recall
print("recall ของ MLPC +  ANOVA: " , rc_anv)

n = 0
for i in anv.get_support():
    if i :
        print(f[n])
    n+=1