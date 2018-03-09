#!/usr/bin/env python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
import sys
import numpy as np
import scipy
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
#import autosklearn.classification
#import autosklearn.regression
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.ensemble import GradientBoostingRegressor
import functools
import pandas as pd
import sys
import timeit


# ***************************************************************************
#  Benchmark Runner
# ***************************************************************************

def benchmark_runner(tests, python_type, power = 3, numberexec =1000):
    result = []
    for test in tests:
        sizes = [10**(i+1) for i in range(power)]
        for size in sizes:
            row = dict()
            row['type'] = python_type
            row['function'] = test.__name__
            row['size'] = size
            a =  np.random.random((size,1))
            b =  np.random.random((size,1))
            c = np.random.random((1,size))
            try:
                try:
                    time = min(timeit.Timer(functools.partial(test, size)).repeat(repeat=3, number=numberexec))
                    print("- {}(size = {}): {} seconds".format(test.__name__,size, time))
                except:
                    try:
                        time = min(timeit.Timer(functools.partial(test,a,b)).repeat(repeat=3, number=numberexec))
                        print("- {}(a and b size = {}): {} seconds".format(test.__name__,size, time))
                    except:
                        try:
                            time = min(timeit.Timer(functools.partial(test,a,c)).repeat(repeat=3, number=numberexec))
                            print("- {}(a and c size = {}): {} seconds".format(test.__name__,size, time))
                        except:
                            time = min(timeit.Timer(functools.partial(test,a)).repeat(repeat=3, number=numberexec))
                            print("- {}(a size = {}): {} seconds".format(test.__name__,size, time))
                row['time'] = time
            except:
                row['time'] = -1
            result.append(row)
    return result

# ***************************************************************************
#  Numpy Test
# ***************************************************************************
def numpy_zeros_test(size):

    # Create an array of all zeros
    a = np.zeros((size,1))

def numpy_ones_test(size):
    # Create an array of all ones
    a = np.ones((size,1))


def numpy_full_test(size):
    # Create a constant array
    a = np.full((size,1), 7)

def numpy_random_test(size):
    # Create an array filled with random values
    a = np.random.random((size,1))

def numpy_sum_test(a,b):
    c = a+b

def numpy_subs_test(a,b):
    c = a-b

def numpy_mul_test(a,b):
    c = a*b

def numpy_div_test(a,b):
    c = a/b

def numpy_sum_scalar_test(a):
    c = a+2

def numpy_subs_scalar_test(a):
    c = a-2

def numpy_mul_scalar_test(a):
    c = a*2

def numpy_div_scalar_test(a):
    c = a/2

def numpy_sqrt_test(a):
    np.sqrt(a)

def numpy_transpose_test(a):
    np.transpose(a)

def numpy_test_log(a):
    np.log10(a)

def numpy_exp_test(a):
    np.exp(a)

def numpy_dot_test(a,b):
    np.dot(a,b)

# ***************************************************************************
# Scipy Test
# ***************************************************************************

def scipy_erf_test(a):
    scipy.special.erf(a)

# ***************************************************************************
# scikitlearn Test
# ***************************************************************************
def scikit_dtc_test(size):
    X, y = datasets.make_classification(n_samples=1000, n_features=size, n_informative=2, n_redundant=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)


def scikit_gbc_test(size):
    X, y = datasets.make_classification(n_samples=1000, n_features=size, n_informative=2, n_redundant=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model =  GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

def scikit_knnc_test(size):
    X, y = datasets.make_classification(n_samples=1000, n_features=size, n_informative=2, n_redundant=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = KNeighborsClassifier(n_jobs=-1)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

def scikit_logr_test(size):
    X, y = datasets.make_classification(n_samples=1000, n_features=size, n_informative=2, n_redundant=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = LogisticRegression(random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

def scikit_lsvc_test(size):
    X, y = datasets.make_classification(n_samples=1000, n_features=size, n_informative=2, n_redundant=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = LinearSVC(random_state=42)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

def scikit_rfc_test(size):
    X, y = datasets.make_classification(n_samples=1000, n_features=size, n_informative=2, n_redundant=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

def scikit_ridgec_test(size):
    X, y = datasets.make_classification(n_samples=1000, n_features=size, n_informative=2, n_redundant=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RidgeClassifier(random_state=42)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

def scikit_dtr_test(size):
    X, y = datasets.make_regression(n_samples=1000, n_features=size, random_state=0, noise=4.0,bias=100.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

def scikit_knnr_test(size):
    X, y = datasets.make_regression(n_samples=1000, n_features=size, random_state=0, noise=4.0,bias=100.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = KNeighborsRegressor(n_jobs=-1)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

def scikit_lsvr_test(size):
    X, y = datasets.make_regression(n_samples=1000, n_features=size, random_state=0, noise=4.0,bias=100.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = LinearSVR(random_state=42)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

def scikit_lasso_test(size):
    X, y = datasets.make_regression(n_samples=1000, n_features=size, random_state=0, noise=4.0,bias=100.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = Lasso(random_state=42)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

def scikit_rfr_test(size):
    X, y = datasets.make_regression(n_samples=1000, n_features=size, random_state=0, noise=4.0,bias=100.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

def scikit_ridger_test(size):
    X, y = datasets.make_regression(n_samples=1000, n_features=size, random_state=0, noise=4.0,bias=100.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = Ridge(random_state=42)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

# ***************************************************************************
# autosklearn Test
# ***************************************************************************
"""
def autosklearn_clf_test(size):
    X, y = datasets.make_classification(n_samples=1000, n_features=size, n_informative=2, n_redundant=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    automl = autosklearn.classification.AutoSklearnClassifier(ensemble_size=1, \
    time_left_for_this_task=60,per_run_time_limit =30,\
    initial_configurations_via_metalearning=0)
    automl.fit(X_train, y_train)

def autosklearn_reg_test(size):
    X, y = datasets.make_regression(n_samples=1000, n_features=size, random_state=0, noise=4.0,bias=100.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    automl = autosklearn.regression.AutoSklearnRegressor(ensemble_size=1, \
    time_left_for_this_task=60,per_run_time_limit =30,\
    initial_configurations_via_metalearning=0)
    automl.fit(X_train, y_train)"""

if __name__ == "__main__":
    python_type = ("Intel" if 'Intel' in sys.version else "Regular")
    print(python_type)
    print("Starting Benchmarks...")
    testNumpy = True
    testScipy =  True
    testScikit = True
    testAuto = False

    results = []
    # ******************************************************************
    # Numpy Benchmarking
    # ******************************************************************
    if testNumpy:
        numpy_tests = [numpy_zeros_test,numpy_ones_test,numpy_full_test,\
        numpy_random_test,numpy_sum_test,numpy_subs_test,\
        numpy_mul_test,numpy_div_test,numpy_sqrt_test, \
        numpy_transpose_test, numpy_sum_scalar_test,numpy_subs_scalar_test,\
        numpy_mul_scalar_test,numpy_div_scalar_test,  numpy_test_log, numpy_exp_test]

        print("********************Numpy Benchmarks************************")
        results += benchmark_runner(numpy_tests, python_type, power=2)
        results += benchmark_runner([numpy_dot_test], python_type)
    # ******************************************************************
    # Scipy Benchmarking
    # ******************************************************************
    if testScipy:
        scipy_tests =  [scipy_erf_test]
        print("********************Scipy Benchmarks************************")
        results += benchmark_runner(scipy_tests, python_type,power=7)

    # ******************************************************************
    # Scikit Learn Benchmarking
    # ******************************************************************
    if testScikit:
        scikit_tests = [scikit_dtc_test, scikit_gbc_test,\
         scikit_knnc_test,scikit_logr_test,scikit_lsvc_test, scikit_rfc_test, \
         scikit_ridgec_test,scikit_dtr_test, scikit_knnr_test, scikit_lsvr_test,\
          scikit_lasso_test, scikit_rfr_test, scikit_ridger_test]
        print("********************Scikit Benchmarks************************")
        results += benchmark_runner(scikit_tests, python_type, power=7)
    # ******************************************************************
    # AutoSklearn Benchmarking
    # ******************************************************************
    """    if testAuto:
            auto_tests = [autosklearn_reg_test, autosklearn_clf_test]
            print("********************Autosklearn Benchmarks************************")
            results += benchmark_runner(auto_tests, python_type, power=7)
    """
    result_pd = pd.DataFrame(results)
    result_pd.to_csv("python_benchmarks.csv", sep='\t', mode="a+", index=False)
    print("End of  Benchmarks")
