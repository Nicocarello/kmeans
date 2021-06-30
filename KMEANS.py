import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('D:/Anaconda/datasets/wine/redwine.csv',sep=';')

# plt.hist(data['quality'])

#MIRO LOS PROMEDIOS DE LOS VINOS SEGUN CADA CALIDAD
print(data.groupby('quality').mean())

#NORMALIZO LOS DATOS

df_norm = (data-data.min())/(data.max()-data.min())

#CLUSTERING JERARQUICO

from sklearn.cluster import AgglomerativeClustering

clust = AgglomerativeClustering(n_clusters = 6,linkage='ward').fit(df_norm)

md=pd.Series(clust.labels_)
print(md)
plt.hist(md)


from scipy.cluster.hierarchy import dendrogram,linkage

z=linkage(df_norm,'ward')
plt.figure(figsize=(25,10))
dendrogram(z,leaf_rotation=90.,leaf_font_size=8)
plt.show()

from sklearn.cluster import KMeans
from sklearn import datasets

model = KMeans(n_clusters = 6)
model.fit(df_norm)
prediccion = model.predict(df_norm)
print(model.labels_)
md_k = pd.Series(model.labels_)

df_norm['calidad_h'] = md
df_norm['calidad_k'] =md_k