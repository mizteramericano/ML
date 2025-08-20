from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
df = load_wine()
X = df.data # features
y = df.target # targer

# print(df.feature_names)
# print(df.target_names)
# print(X)
# print(y)
# print(X.shape)
# แยกข้อมูล Train 80% และ Test 20%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# หาค่าความสูงจำกัดของต้นไม้ที่เหมาะสม โดยบันทึกผลลงในใบงานส่วนที่ 2
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train,y_train)
# Train โดยใช้ข้อมูลที่แยกไว้สำหรับ Train และทำการพยากรณ์ข้อมูลโดยใช้ข้อมูลที่แยกไว้สำหรับ Test
predict = model.predict(X_test) #ทำข้อสอบ

# แสดงผลต้นไม้ บันทึก และวางภาพลงในใบงานส่วนที่ 3
print(classification_report(y_test, predict))

# ใบงานส่วนที่ 4
con = confusion_matrix(y_test,predict)
print(con)

# ใบงานส่วนที่ 3
plt.figure(figsize=(12,12))
plot_tree(model , fontsize=10)
plt.show()
