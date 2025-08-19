import os

from kagglehub import KaggleDatasetAdapter
from sklearn import datasets
import kagglehub
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

path = kagglehub.dataset_download("aryashah2k/credit-card-customer-data")

# path นี้ยู่ใน....
print("Path to dataset files:", path)

# df = pd.read_csv("C:\Users\ADMIN\.cache\kagglehub\datasets\aryashah2k\credit-card-customer-data\versions\2")
# หาไฟล์ CSV ในโฟลเดอร์
for file in os.listdir(path):
    if file.endswith(".csv"):
        csv_file = os.path.join(path, file)
        print("CSV file found:", csv_file)

        # โหลด CSV เข้า DataFrame
        df = pd.read_csv(csv_file)
        print(df.head())  # แสดง 5 แถวแรก

print(df.columns)

# train model
