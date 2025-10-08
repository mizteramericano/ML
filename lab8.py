from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

iris = datasets.load_iris()
X = iris.data
y = iris.target
tn = iris.target_names

print("target")
print(y)

ag = AgglomerativeClustering(n_clusters=5)
ag_pred = ag.fit_predict(X)
print("Hierarchical" , ag_pred)
print(tn)

plt.title("agnes Clustering")
dendrogram(linkage(X,'ward'))
plt.show()

