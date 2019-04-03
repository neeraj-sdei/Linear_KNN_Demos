import pickle
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
dataset = pd.read_csv('Mall_Customers.csv')
X= dataset.iloc[:, [3,4]].values


classifier_f= open("model_kmeans.pickle", "rb")
kmeansmodel = pickle.load(classifier_f)
classifier_f.close()

classifier_f= open("clusters_kmeans.pickle", "rb")
kmeans = pickle.load(classifier_f)
classifier_f.close()

y_kmeans= kmeansmodel.fit_predict(X)

class KaggleCustomerBasket:
    def getGraph(self):
        plt.cla()
        plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
        plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
        plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
        plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
        plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
        plt.title('Clusters of customers')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        img = io.BytesIO()
        plt.savefig(img, format='png')
        graph_url = base64.b64encode(img.getvalue()).decode()

        #plt.show()
        return 'data:image/png;base64,{}'.format(graph_url)
        #plt.show()




