import timeit
import sklearn.datasets as skl
import pandas as pd

###################################
n_clusters = '3'
max_iter = '300'
blob_samples = '500'
bloc_n_features = '3'
blob_cluster_std = '0.6'  # default is 1
blob_random_state = '0'  # default is false
###################################


setup = """
import sklearn.datasets as skl
blobs = skl.make_blobs(n_samples=""" + blob_samples + """, centers= """ + bloc_n_features + """, cluster_std= """ + blob_cluster_std + """, random_state= """ + blob_random_state + """)[0]
print('blobs ', blobs)"""

stmt = '''
from kmeans_objects import k_means_serial
kmeans = k_means_serial(n_clusters =''' + n_clusters + ''', max_iter = ''' + max_iter + ''')
kmeans.fit(blobs)
'''


k_means_run_serial = timeit.timeit(stmt=stmt, setup=setup, number=300) / 300

result = pd.DataFrame(k_means_run_serial)
print('\nresult ', result)
