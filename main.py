from io import BytesIO

from flask import Flask, request, make_response
from stemming.porter2 import stem
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import numpy as np

app = Flask(__name__)


def cleanse_text(text):
    if text:
        # remove whitespace
        clean = ' '.join(text.split())
        # stemming
        red_text = [stem(i.lower()) for i in clean.split()]
        return ' '.join(red_text)

    else:
        return text


@app.route('/cluster', methods=["POST"])
def cluster():
    data = pd.read_csv(request.files['dataset'])
    print(data.head())
    unstructure = 'text'
    if 'col' in request.args:
        unstructure = request.args.get('col')
    no_of_clusters = 2
    if no_of_clusters in request.args:
        no_of_clusters = request.args.get('no_of_clusters')
    data['clean_sum'] = data[unstructure].apply(cleanse_text)
    print(data['clean_sum'])
    vect = CountVectorizer()
    counts = vect.fit_transform(data['clean_sum'])
    kmeans = KMeans(n_clusters=no_of_clusters)
    data['cluster_num'] = kmeans.fit_predict(counts)
    data = data.drop(['clean_sum'], axis=1)
    print(data)
    data.to_excel("file.xlsx",sheet_name='cluster',encoding='utf-8',index=False)

    cluster=[]
    for i in range(np.shape(kmeans.cluster_centers_)[0]):
        data_center=pd.concat([pd.Series(vect.get_feature_names()),pd.DataFrame(kmeans.cluster_centers_[i])],axis=1)
        data_center.columns=['keywords','weights']
        data_center=data_center.sort_values(by='weights',ascending=False)
        data_clust=data_center.head(10)['keywords']
        cluster.append(data_clust)
    pd.DataFrame(cluster).to_excel("clust_cen.xlsx",sheet_name='cluster',encoding='utf-8',index=False)
    return 'this works'


if __name__ == "__main__":
    app.run(host='0.0.0.0')
