import time
import timeit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SERIAL = 1
PARALLEL = 1
yerr = np.linspace(0.05, 0.2, 10)


def setupp(blob_samples):
    setup = """import sklearn.datasets as skl; blobs = skl.make_blobs(n_samples=""" + blob_samples + """, centers= """ + bloc_n_features + """, cluster_std= """ + blob_cluster_std + """, random_state= """ + blob_random_state + """)[0]; print('blobss', """+ blob_samples + """)"""
    return setup


def stmtt(n_cores):
    stmt = '''from kmeans_objects import k_means_parallel; kmeans = k_means_parallel(n_clusters =''' + n_clusters + ''', max_iter = ''' + max_iter + ''', num_of_cores = ''' + n_cores + '''); kmeans.fit(blobs)'''
    return stmt


if SERIAL:
    ###################################
    N = 50 # numero di exe timeit
    M = 3
    STEP = 30000
    n_clusters = '3'
    max_iter = '8'
    blob_samples = '0'
    bloc_n_features = '3'
    blob_cluster_std = '0.6'  # default is 1
    blob_random_state = '0'  # default is false
    ###################################


    print('\nSerial starting ')
    time.sleep(1)


    stmt = '''from kmeans_objects import k_means_serial; kmeans = k_means_serial(n_clusters =''' + n_clusters + ''', max_iter = ''' + max_iter + '''); kmeans.fit(blobs)'''

    k_means_run_serial_t = [0.0]
    x_ax = [0]
    for i in range(1, M+1):
        if i > 2:
            print('\nlast loop')
            STEP = 40000
        x_ax.append(i*STEP)
        blob_samples = str(int(blob_samples) + STEP)
        #print('\nblob qui ', blob_samples)
        setup = setupp(blob_samples)
        k_means_run_serial_t.append(timeit.timeit(stmt=stmt, setup=setup, number=N) / N)
        plt.plot(i*STEP, k_means_run_serial_t[i], 'o', color='blue')
    #print('\nresult ', pd.DataFrame(k_means_run_serial_t))  # tempo medio del set di esecuzioni timeit (N) per ogni round

    plt.plot(x_ax, k_means_run_serial_t, '--c', label='Serial execution', color='cyan')

    plt.legend(loc="upper left")


if PARALLEL:
    print('\nParallel starting ')
    time.sleep(5)
    ###################################
    N = 50     # numero di exe timeit
    M = 3
    FIRST_STEP = 30000
    n_clusters = '3'
    max_iter = '8'
    blob_samples = '0'
    bloc_n_features = '3'
    blob_cluster_std = '0.6'  # default is 1
    blob_random_state = '0'  # default is false
    # num_of_cores = '2'
    ###################################

    # stmt = '''from kmeans_objects import k_means_parallel; kmeans = k_means_parallel(n_clusters =''' + n_clusters + ''', max_iter = ''' + max_iter + ''', num_of_cores = ''' + num_of_cores + '''); kmeans.fit(blobs)'''

    k_means_parallels = []
    k_means_run_parallel_t_2_cores = [0.0]
    k_means_run_parallel_t_4_cores = [0.0]
    k_means_run_parallel_t_8_cores = [0.0]
    k_means_parallels.append(k_means_run_parallel_t_2_cores)
    k_means_parallels.append(k_means_run_parallel_t_4_cores)
    k_means_parallels.append(k_means_run_parallel_t_8_cores)
    z = 0
    x_ax = [0]
    fill_a_ax = True
    for cores_number in (2 ** p for p in range(1, 4)):
        blob_samples = '0'
        STEP = FIRST_STEP
        stmt = stmtt(str(cores_number))
        for i in range(1, M+1):
            if i > 2:
                print('\nlast loop')
                STEP = 40000
            if fill_a_ax:
                x_ax.append(i * STEP)
            blob_samples = str(int(blob_samples) + STEP)
            setup = setupp(blob_samples)
            #print('z ', z, ' i ', i)

            k_means_parallels[z].append(timeit.timeit(stmt=stmt, setup=setup, number=N) / N)
            #print('k_means_parallels ', k_means_parallels)
            if cores_number == 2:
                plt.plot(i * STEP, k_means_run_parallel_t_2_cores[i], 'o', color='orange')
            elif cores_number == 4:
                plt.plot(i * STEP, k_means_run_parallel_t_4_cores[i], 'o', color='green')
            elif cores_number == 8:
                plt.plot(i * STEP, k_means_run_parallel_t_8_cores[i], 'o', color='black')
        fill_a_ax = False
        print('\n __ fine for interno ')
        z += 1


    #print('\nresult_2 ', pd.DataFrame(k_means_run_parallel_t_2_cores))  # tempo medio del set di esecuzioni timeit (N) per ogni round
    #print('\nresult_4 ', pd.DataFrame(k_means_run_parallel_t_4_cores))  # tempo medio del set di esecuzioni timeit (N) per ogni round
    #print('\nresult_6 ', pd.DataFrame(k_means_run_parallel_t_6_cores))  # tempo medio del set di esecuzioni timeit (N) per ogni round
    #print('\nresult_8 ', pd.DataFrame(k_means_run_parallel_t_8_cores))  # tempo medio del set di esecuzioni timeit (N) per ogni round

    plt.plot(x_ax, k_means_run_parallel_t_2_cores, '--c', label='2_Core Parallel execution', color='orange')
    plt.plot(x_ax, k_means_run_parallel_t_4_cores, '--c', label='4_Core Parallel execution', color='green')
    plt.plot(x_ax, k_means_run_parallel_t_8_cores, '--c', label='8_Core Parallel execution', color='black')
    plt.legend(loc="upper left")

plt.ylabel('Tempi di esecuzione')
plt.show()

