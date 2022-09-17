import timeit
import sklearn.datasets as skl
import pandas as pd

###################################
N = 100 #numero di exe timeit
n_clusters = '3'
max_iter = '300'
blob_samples = '500'
bloc_n_features = '3'
blob_cluster_std = '0.6'  # default is 1
blob_random_state = '0'  # default is false
###################################


def setupp(blob_samples):
    setup = """import sklearn.datasets as skl; blobs = skl.make_blobs(n_samples=""" + blob_samples + """, centers= """ + bloc_n_features + """, cluster_std= """ + blob_cluster_std + """, random_state= """ + blob_random_state + """)[0]; print('blobss', """+ blob_samples + """)"""
    return setup


stmt = '''
from kmeans_objects import k_means_serial
kmeans = k_means_serial(n_clusters =''' + n_clusters + ''', max_iter = ''' + max_iter + ''')
kmeans.fit(blobs)
'''

k_means_run_serial_t = []
for i in range(1,6):
    blob_samples = str(int(blob_samples) + 500)
    print('\nblob qui ', blob_samples)
    setup = setupp(blob_samples)
    k_means_run_serial_t.append(timeit.timeit(stmt=stmt, setup=setup, number=N) / N)

result = pd.DataFrame(k_means_run_serial_t)
print('\nresult ', result)  # tempo medio del set di esecuzioni timeit (N) per ogni round
