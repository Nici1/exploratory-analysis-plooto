import sys, os
sys.path.insert(0, './')
sys.path.append(os.path.join(os.path.dirname(__file__), "../models"))

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


from datetime import datetime
import numpy as np
import pandas as pd
import json
import shutil

import unittest
import tensorflow as tf

# Algorithm imports
from anomalyDetection import AnomalyDetectionAbstract
from borderCheck import BorderCheck
from welford import Welford
from EMA import EMA
from filtering import Filtering
from isolationForest import IsolationForest
from GAN import GAN
from PCA import PCA
from hampel import Hampel
from MACD import MACD
from clustering import Clustering
from combination import Combination, AND, OR

# Normalization imports
from normalization import LastNAverage, PeriodicLastNAverage


def create_model_instance(algorithm_str, configuration, save = False):
        model =  eval(algorithm_str)
        model.configure(configuration)
        if (save):
            #save config to temporary file for retrain purposes
            if not os.path.isdir("configuration"):
                os.makedirs("configuration")
            
            filepath = "configuration/Test_config.txt"

            with open(filepath, 'w') as data_file:
                full_conf = {
                    "anomaly_detection_alg": [algorithm_str],
                    "anomaly_detection_conf":[configuration]
                }
                json.dump(full_conf, data_file)

            model.configure(configuration, "Test_config.txt", algorithm_indx = 0)
        else:
            model.configure(configuration, algorithm_indx = 0)

        return model

def create_message(timestamp, value):
    message = {
        "timestamp" : timestamp,
        "ftr_vector" : value
    }
    return message

def create_testing_file(filepath, withzero = False, FV_length = None):
    timestamps = [1459926000 + 3600*x for x in range(100)]

    values = [1.0]*100
    if(withzero):
        values[-1] = 0.0

    vals = []
    timest = []
    if (FV_length is not None):
        for i in range(FV_length, len(values)):
            vals.append(values[i-FV_length+1:i+1])
            timest.append(timestamps[i])
        values = vals
        timestamps = timest
    else:
        values = [[1.0]]*100
        if(withzero):
            values[-1] = [0.0]

    
    df = pd.DataFrame({'timestamp': timestamps, 'ftr_vector': values})
    df.to_csv(filepath, index = False)

    return filepath

def create_clustering_testing_file(filepath):
    timestamps = [1459926000 + 3600*x for x in range(14)]

    data = [
        [10.3, 10.44],
        [9.8, 11.3],
        [15.433, 16.4],
        [0, 0.2],
        [0.2, 0.234],
        [0.3, 0.12],
        [0.11, 0.0456],
        [0.01, 0.07996],
        [1.3, 0.211],
        [1, 1.65],
        [1.2, 1.22],
        [1.332, 1.03],
        [1.222, 1.01],
        [1.554, 1.44]
    ]
    
    df = pd.DataFrame({'timestamp': timestamps, 'ftr_vector': data})
    df.to_csv(filepath, index = False)

    return filepath

def create_testing_file_feature_construction(filepath):
    timestamps = [1459926000 + 3600*x for x in range(20)]
    values = [[x, x+100] for x in range(20)]
    data = {
        'timestamp': timestamps,
        'ftr_vector': values
    }
    testset = pd.DataFrame(data = data)
    testset.to_csv(filepath, index = False)

    return filepath


class BCTestCase(unittest.TestCase):

    def setUp(self):
        configuration = {
        "input_vector_size": 1,
        "warning_stages": [0.7, 0.9],
        "UL": 4,
        "LL": 2,
        "output": [],
        "output_conf": [{}]
        }
        self.model = create_model_instance("BorderCheck()", configuration)

    def tearDown(self) -> None:
        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")

        return super().tearDown()


class BCTestClassPropperties(BCTestCase):
    #Check propperties setup.
    def test_UL(self):
        self.assertEqual(self.model.UL, 4)
        self.assertEqual(self.model.LL, 2)
        self.assertEqual(self.model.warning_stages, [0.7, 0.9])


class BCTestFunctionality(BCTestCase):

    def test_OK(self):
        #Test a value at the center of the range. Should give OK status.
        message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                 [3])
        self.model.message_insert(message)
        self.assertEqual(self.model.status_code, 1)

    def test_outliers(self):
        #Above UL. Should give Error (-1 status code).
        message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                 [5])
        self.model.message_insert(message)
        self.assertEqual(self.model.status_code, -1)

        #Below LL. Should give Error.
        message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                 [1])
        self.model.message_insert(message)
        self.assertEqual(self.model.status_code, -1)

        #Close to LL. Should give warning (0 status code)
        message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                 [2.1])
        self.model.message_insert(message)
        self.assertEqual(self.model.status_code, 0)


class WelfordDefinedNTestCase(unittest.TestCase):

    def setUp(self):
        configuration = {
        "input_vector_size": 1,
        "warning_stages": [0.7, 0.9],
        "N": 4,
        "X": 2,
        "output": [],
        "output_conf": [
            {}
        ],
        }
        self.model = create_model_instance("Welford()", configuration)
    
    def tearDown(self) -> None:
        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")

        return super().tearDown()


class WelfordDefinedNTestClassPropperties(WelfordDefinedNTestCase):
    #Check propperties setup.
    def test_Propperties(self):
        self.assertEqual(self.model.N, 4)
        self.assertEqual(self.model.X, 2)
        self.assertEqual(self.model.warning_stages, [0.7, 0.9])


class WelfordDefinedNTestFunctionality(WelfordDefinedNTestCase):
    def test_OK(self):
        test_data = [1, 2, 3, 4, 1, 2]

        for data_indx in range(len(test_data)):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     [test_data[data_indx]])
            self.model.message_insert(message)

            if(data_indx < 4):
                self.assertEqual(self.model.status_code, 2)
            else:
                self.assertEqual(self.model.status_code, 1)

    def test_error(self):
        test_data = [1, 2, 3, 4, -0.1, 5.73]

        for data_indx in range(len(test_data)):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     [test_data[data_indx]])
            self.model.message_insert(message)

            if(data_indx < 4):
                self.assertEqual(self.model.status_code, 2)
            else:
                self.assertEqual(self.model.status_code, -1)


class WelfordUndefinedNTestCase(unittest.TestCase):

    def setUp(self):
        configuration = {
        "input_vector_size": 1,
        "X": 2,
        "warning_stages": [],
        "output": [],
        "output_conf": [
            {}
        ],
        }
        self.model = create_model_instance("Welford()", configuration)
    
    def tearDown(self) -> None:
        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")

        return super().tearDown()


class WelfordUndefinedNTestClassPropperties(WelfordUndefinedNTestCase):
    #Check propperties setup.
    def test_Propperties(self):
        self.assertEqual(self.model.X, 2)


class WelfordUndefinedNTestFunctionality(WelfordUndefinedNTestCase):
    def test_OK(self):
        test_data = [1, 2, 2.4, 2.6, 1, 3.1]

        for data_indx in range(len(test_data)):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     [test_data[data_indx]])
            self.model.message_insert(message)
            
            # Check memory length
            self.assertEqual(self.model.count, data_indx+1)

            if(data_indx < 2):
                self.assertEqual(self.model.status_code, 2)
            else:
                self.assertEqual(self.model.status_code, 1)

    def test_error(self):
        test_data = [1, 2, 3, -1, 5, -2.5]

        for data_indx in range(len(test_data)):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     [test_data[data_indx]])
            self.model.message_insert(message)

            # Check memory length
            self.assertEqual(self.model.count, data_indx+1)

            if(data_indx < 2):
                self.assertEqual(self.model.status_code, 2)
            else:
                self.assertEqual(self.model.status_code, -1)


class EMATestCase(unittest.TestCase):

    def setUp(self):
        configuration = {
        "input_vector_size": 1,
        "warning_stages": [0.7, 0.9],
        "UL": 4,
        "LL": 2,
        "N": 5,
        "output": [],
        "output_conf": [{}],
        }
        self.model = create_model_instance("EMA()", configuration)

    def tearDown(self) -> None:
        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")

        return super().tearDown()


class EMATestClassPropperties(EMATestCase):
    #Check propperties setup.
    def test_Propperties(self):
        self.assertEqual(self.model.UL, 4)
        self.assertEqual(self.model.LL, 2)
        self.assertEqual(self.model.N, 5)
        self.assertEqual(self.model.warning_stages, [0.7, 0.9])


class EMATestFunctionality(EMATestCase):
    def test_OK(self):
        #Insert values in the middle of the range. All should have no error.
        test_array = [3, 3, 3]
        for i in test_array:
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     [i])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, 1)

    def test_Error(self):
        #Check values which drift out of the range. Should transition from OK -> warning -> error
        test_array = [3, 4, 4, 4, 4, 5, 5, 5]
        expected_status = [1, 1, 1, 0, 0, -1, -1, -1]
        for i in range(len(test_array)):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     [test_array[i]])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_status[i])

     
class Filtering1TestCase(unittest.TestCase):
    #Test case for filtering - mode 1
    def setUp(self):
        configuration = {
        "input_vector_size": 1,
        "mode": 1,
        "LL": 0,
        "UL": 1,
        "filter_order": 3,
        "cutoff_frequency":0.4,
        "warning_stages": [0.7, 0.9], 
        "output": [],
        "output_conf": [{}],
        }
        self.model = create_model_instance("Filtering()", configuration)

    def tearDown(self) -> None:
        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")

        return super().tearDown()


class Filtering0TestCase(unittest.TestCase):
    #Test case for filtering - mode 0
    def setUp(self):
        configuration = {
        "input_vector_size": 1,
        "mode": 0,
        "LL": 0,
        "UL": 1,
        "filter_order": 3,
        "cutoff_frequency":0.4,
        "warning_stages": [0.7, 0.9], 
        "output": [],
        "output_conf": [{}],
        }
        self.model = create_model_instance("Filtering()", configuration)

    def tearDown(self) -> None:
        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")

        return super().tearDown()


class Filtering1TestClassPropperties(Filtering1TestCase):
    #Check propperties setup.
    def test_Propperties(self):
        self.assertEqual(self.model.UL, 1)
        self.assertEqual(self.model.LL, 0)
        self.assertEqual(self.model.filter_order, 3)
        self.assertEqual(self.model.cutoff_frequency, 0.4)
        self.assertEqual(self.model.warning_stages, [0.7, 0.9])
        self.assertEqual(self.model.mode, 1)

    def test_Kernel(self):
        #Test kernel coefficients
        self.assertAlmostEqual(self.model.a[0], 1, 8)
        self.assertAlmostEqual(self.model.a[1], -0.57724052, 8)
        self.assertAlmostEqual(self.model.a[2], 0.42178705, 8)
        self.assertAlmostEqual(self.model.a[3], -0.05629724, 8)

        self.assertAlmostEqual(self.model.b[0], 0.09853116, 8)
        self.assertAlmostEqual(self.model.b[1], 0.29559348, 8)
        self.assertAlmostEqual(self.model.b[2], 0.29559348, 8)
        self.assertAlmostEqual(self.model.b[3], 0.09853116, 8)

        self.assertAlmostEqual(self.model.z[0], 0.90146884, 8)
        self.assertAlmostEqual(self.model.z[1], 0.02863483, 8)
        self.assertAlmostEqual(self.model.z[2], 0.1548284, 8)


class Filtering1TestFunctionality(Filtering1TestCase):
    def test_Constant(self):
        #Test constant datastream.
        test_array = np.ones(10)
        for i in test_array:
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(), [i])
            self.model.message_insert(message)
            self.assertAlmostEqual(self.model.filtered, 1, 8)
            self.assertAlmostEqual(self.model.result, 0, 8)

    def test_Errors(self):
        #Test drifting datastream.
        test_array = [0, 0, 0, 1, 2, 2, 2]
        expected_status = [0, 1, 1, -1, -1, 1, 1]
        for i in range(len(test_array)):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(), [test_array[i]])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_status[i])


class Filtering0TestFunctionality(Filtering0TestCase):
    def test_Constant(self):
        #Test constant datastream.
        test_array = np.ones(10)
        for i in test_array:
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(), [i])
            self.model.message_insert(message)
            self.assertAlmostEqual(self.model.filtered, 1, 8)
            self.assertAlmostEqual(self.model.result, 1, 8)

    def test_Errors(self):
        #Test drifting datastream.
        test_array = [0.5, 0.5, 0.5, 1, 1, 1, 2, 2, 2]
        expected_status = [0, 1, 1, 1, 1, 0, -1, -1, -1]
        for i in range(len(test_array)):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(), [test_array[i]])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_status[i])


class IsolForestTestCase(unittest.TestCase):
    def setUp(self):
        # Set random seed so results are reproducable
        np.random.seed(0)

        if not os.path.isdir("unittest"):
            os.makedirs("unittest")

        create_testing_file("./unittest/IsolForestTestData.csv",
                            withzero=True)

        configuration = {
        "train_data": "./unittest/IsolForestTestData.csv",
        "train_conf": {
            "max_features": 7,
            "max_samples": 5,
            "contamination": "0.1",
            "model_name": "IsolForestTestModel"
        },
        "retrain_file": "./unittest/IsolationForestRetrainData.csv",
        "retrain_interval": 10,
        "samples_for_retrain": 100,
        "input_vector_size": 1,
        "shifts": [[1,2,3,4]],
        "averages": [[1,2]],
        "output": [],
        "output_conf": [{}]
        }
        self.f = "models"

        #Create a temporary /models folder.
        if not os.path.isdir(self.f):
            os.makedirs(self.f)
        self.model = create_model_instance("IsolationForest()", configuration, save = True)
    
    def tearDown(self):
        if os.path.isdir(self.f):
            shutil.rmtree(self.f)

        # Delete unittest folder
        shutil.rmtree("unittest")

        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")


class IsolForestTestClassPropperties(IsolForestTestCase):
    #Check propperties setup.
    def test_Propperties(self):
        self.assertEqual(self.model.max_features, 7)
        self.assertEqual(self.model.max_samples, 5)
        self.assertEqual(self.model.retrain_interval, 10)
        self.assertEqual(self.model.samples_for_retrain, 100)


class IsolForestTestFunctionality(IsolForestTestCase):
    def test_OK(self):
        #Insert same values as in train set (status should be 1).
        test_array = [1.0]*15
        expected_status = [2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        for i in range(len(test_array)):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     [test_array[i]])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_status[i])
        self.assertEqual(self.model.retrain_counter, 1)

    def test_errors(self):
        #insert different values as in train set (status should be -1).
        test_array = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        expected_status = [2, 2, 2, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        for i in range(len(test_array)):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     [test_array[i]])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_status[i])
        self.assertEqual(self.model.retrain_counter, 1)


class GANTestCase(unittest.TestCase):
    def setUp(self):
        # Set random seed so results are reproducable
        np.random.seed(0)
        tf.random.set_seed(3241)

        # Make unittest directory
        if not os.path.isdir("unittest"):
            os.makedirs("unittest")

        create_testing_file("./unittest/GANTestData.csv", withzero = True, FV_length=10)

        configuration = {
        "train_data": "./unittest/GANTestData.csv",
        "train_conf":{
            "model_name": "GAN_Test",
            "N_shifts": 9,
            "N_latent": 3,
            "K": 1.5,
            "len_window": 1000
        },
        "retrain_file": "./unittest/GANRetrainData.csv",
        "retrain_interval": 3,
        "samples_for_retrain": 90,
        "input_vector_size": 10,
        "output": [],
        "output_conf": [{}]
        }
        self.f = "models"

        #Create a temporary /models folder.
        if not os.path.isdir(self.f):
            os.makedirs(self.f)
        self.model = create_model_instance("GAN()", configuration, save = True)


    def tearDown(self):
        if os.path.isdir(self.f):
            shutil.rmtree(self.f)
        
        # Delete unittest folder
        shutil.rmtree("unittest")

        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")


class GANTestClassPropperties(GANTestCase):
    #Check propperties setup.
    def test_Propperties(self):
        self.assertEqual(self.model.N_shifts, 9)
        self.assertEqual(self.model.N_latent, 3)
        self.assertEqual(self.model.retrain_interval, 3)
        self.assertEqual(self.model.samples_for_retrain, 90)


class GANTestFunctionality(GANTestCase):
    def test_OK(self):
        #Insert same values as in train set (status should be 1).
        test_array = [1]*10
        for i in range(3):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     test_array)
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, 1)
        self.assertEqual(self.model.retrain_counter, 1)

    def test_errors(self):
        #Insert same values as in train set (status should be 1).
        test_array = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        for i in range(3):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     test_array)
            self.model.message_insert(message)

        test_array = [1, 2, 4, 100, 1, 1, 5, 1, 1, 1]

        for i in range(3):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     test_array)
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, -1)
        
        test_array1 = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        for i in range(1):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     test_array)
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, -1)
        self.assertEqual(self.model.retrain_counter, 2)


class PCATestCase(unittest.TestCase):
    def setUp(self):
        # Set random seed so results are reproducable
        np.random.seed(0)

        # Make unittest directory
        if not os.path.isdir("unittest"):
            os.makedirs("unittest")
            
        create_testing_file("./unittest/PCATestData.csv", withzero = True, FV_length = 10)

        configuration = {
        "train_data": "./unittest/PCATestData.csv",
        "train_conf":{
            "max_features": 5,
            "max_samples": 25,
            "contamination": "0.01",
            "model_name": "PCA_Test",
            "N_components": 5
        },
        "retrain_file": "./unittest/PCARetrainData.csv",
        "retrain_interval": 10,
        "samples_for_retrain": 90,
        "input_vector_size": 10, 
        "output": [],
        "output_conf": [{}]
        }
        self.f = "models"

        #Create a temporary /models folder.
        if not os.path.isdir(self.f):
            os.makedirs(self.f)
        self.model = create_model_instance("PCA()", configuration, save = True)

    def tearDown(self):
        if os.path.isdir(self.f):
            shutil.rmtree(self.f)

        # Delete unittest folder
        shutil.rmtree("unittest")

        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")


class PCATestClassPropperties(PCATestCase):
    def test_Propperties(self):
        self.assertEqual(self.model.max_features, 5)
        self.assertEqual(self.model.max_samples, 25)
        self.assertEqual(self.model.retrain_interval, 10)
        self.assertEqual(self.model.samples_for_retrain, 90)


class PCATestFunctionality(PCATestCase):
    def test_OK(self):
        #Insert same values as in train set (status should be 1).
        test_array = [1]*10
        for i in range(15):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     test_array)
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, 1)
        self.assertEqual(self.model.retrain_counter, 1)

    def test_errors(self):
        #Insert same values as in train set (status should be 1).
        test_array = [0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0, -0.5, -0-5]
        for i in range(15):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     test_array)
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, -1)
        self.assertEqual(self.model.retrain_counter, 1)


class MACDTestCase(unittest.TestCase):
    def setUp(self):
        # Set random seed so results are reproducable
        np.random.seed(0)

        configuration = {
        "input_vector_size": 1,
        "warning_stages": [0.5],
        "period1": 10,
        "period2": 30,
        "UL": 1.0,
        "LL": -1.0,
        "output": [],
        "output_conf": [{}]
    }
        self.model = create_model_instance("MACD()", configuration, save = True)


class MACDTestPropperties(MACDTestCase):
    def test_Propperties(self):
        self.assertEqual(self.model.warning_stages, [0.5])
        self.assertEqual(self.model.period1, 10)
        self.assertEqual(self.model.period2, 30)
        self.assertEqual(self.model.UL, 1)
        self.assertEqual(self.model.LL, -1)


class MACDTestFunctionality(MACDTestCase):
    def  test_OK(self):
        for i in range(30):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     [1])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, 1)
    
    def test_NOK(self):
        for i in range(30):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     [1])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, 1)

        expected_statusses = [1,1,1,0,0,0,-1,-1,-1,-1]
        for i in range(10):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     [-0.4*i])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_statusses[i])


class ClusteringTestCase(unittest.TestCase):
    def setUp(self):
        # Set random seed so results are reproducable
        np.random.seed(0)

        if not os.path.isdir("unittest"):
            os.makedirs("unittest")

        create_clustering_testing_file("./unittest/ClusteringTestData.csv")

        configuration = {
        "train_data": "./unittest/ClusteringTestData.csv",
        "retrain_file": "./unittest/ClusteringRetrainData.csv",
        "eps": 0.98,
        "min_samples": 3,
        "treshold": 1.5,
        "retrain_interval": 10,
        "samples_for_retrain": 10,
        "input_vector_size": 2,
        "output": [],
        "output_conf": [{}]
        }
        self.f = "models"

        #Create a temporary /models folder.
        if not os.path.isdir(self.f):
            os.makedirs(self.f)
        self.model = create_model_instance("Clustering()", configuration, save=True)
    
    def tearDown(self):
        if os.path.isdir(self.f):
            shutil.rmtree(self.f)

        # Delete unittest folder
        shutil.rmtree("unittest")

        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")


class ClusteringTestClassPropperties(ClusteringTestCase):
    #Check propperties setup.
    def test_Propperties(self):
        self.assertEqual(self.model.eps, 0.98)
        self.assertEqual(self.model.min_samples, 3)
        self.assertEqual(self.model.treshold, 1.5)
        self.assertEqual(self.model.retrain_interval, 10)
        self.assertEqual(self.model.samples_for_retrain, 10)


class ClusteringTestFunctionality(ClusteringTestCase):
    def test_OK(self):
        # Insert similar values as in train set (status should be 1).
        test_array = [[1.0, 0.9], [0.4, 0.0], [2.554, 2.44]]
        expected_status = [1, 1, 1]
        for i in range(len(test_array)):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     test_array[i])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_status[i])
            

    def test_errors(self):
        #insert different values as in train set (status should be -1).
        test_array = [[3.054, 2.96], [10.0, 11.0], [-5.0, -1.0]]
        expected_status = [-1, -1, -1]
        for i in range(len(test_array)):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     test_array[i])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_status[i])
    
    def test_retrain(self):
        #insert different values as in train set (status should be -1).
        test_array = [
            [10, 20.96],
            [10.0, 1.0],
            [10.4, 21.1],
            [0.2, 0.9],
            [10.4, 20.098],
            [9.99, 20.44],
            [9.988, 20.656],
            [10.443, 21],
            [10.454, 20.546],
            [9.995, 20.99],
            [10.005, 20.3425],
            [10.1295, 20.456],
            [1.0, 0.9]     
            ]
        expected_status = [-1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1]
        for i in range(len(test_array)):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     test_array[i])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_status[i])
        self.assertEqual(self.model.retrain_counter, 1)


class ProphetTestCase(unittest.TestCase):
    def setUp(self):
        # Set random seed so results are reproducable
        np.random.seed(0)

        if not os.path.isdir("unittest"):
            os.makedirs("unittest")

        create_clustering_testing_file("./unittest/ClusteringTestData.csv")

        configuration = {
        "train_data": "./unittest/ClusteringTestData.csv",
        "retrain_file": "./unittest/ClusteringRetrainData.csv",
        "eps": 0.98,
        "min_samples": 3,
        "treshold": 1.5,
        "retrain_interval": 10,
        "samples_for_retrain": 10,
        "input_vector_size": 2,
        "output": [],
        "output_conf": [{}]
        }
        self.f = "models"

        #Create a temporary /models folder.
        if not os.path.isdir(self.f):
            os.makedirs(self.f)
        self.model = create_model_instance("Clustering()", configuration, save=True)
    
    def tearDown(self):
        if os.path.isdir(self.f):
            shutil.rmtree(self.f)

        # Delete unittest folder
        shutil.rmtree("unittest")

        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")


class ProphetTestClassPropperties(ProphetTestCase):
    #Check propperties setup.
    def test_Propperties(self):
        # TODO
        pass


class ProphetTestFunctionality(ProphetTestCase):
    # TODO everything
    
    def test_OK(self):
        pass
        """# Insert similar values as in train set (status should be 1).
        test_array = [[1.0, 0.9], [0.4, 0.0], [2.554, 2.44]]
        expected_status = [1, 1, 1]
        for i in range(len(test_array)):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     test_array[i])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_status[i])"""
            

    def test_errors(self):
        pass
        """#insert different values as in train set (status should be -1).
        test_array = [[3.054, 2.96], [10.0, 11.0], [-5.0, -1.0]]
        expected_status = [-1, -1, -1]
        for i in range(len(test_array)):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     test_array[i])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_status[i])"""
    
    def test_retrain(self):
        pass
        """#insert different values as in train set (status should be -1).
        test_array = [
            [10, 20.96],
            [10.0, 1.0],
            [10.4, 21.1],
            [0.2, 0.9],
            [10.4, 20.098],
            [9.99, 20.44],
            [9.988, 20.656],
            [10.443, 21],
            [10.454, 20.546],
            [9.995, 20.99],
            [10.005, 20.3425],
            [10.1295, 20.456],
            [1.0, 0.9]     
            ]
        expected_status = [-1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1]
        for i in range(len(test_array)):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(),
                                     test_array[i])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_status[i])
        self.assertEqual(self.model.retrain_counter, 1)"""


class CombinationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.configuration = {
            "anomaly_detection_alg": ["Combination()"],
            "status_determiner": "AND()",
            "anomaly_algorithms": ["BorderCheck()", "BorderCheck()"],
            "output": [],
            "output_conf": [{}],
            "input_vector_size": 1,
            "anomaly_algorithms_configurations":[
            {
            "input_vector_size": 1,
            "warning_stages": [0.9],
            "UL": 0.5,
            "LL": 0,
            "output": [],
            "output_conf": [{}]
            },
            {
                "input_vector_size": 1,
                "warning_stages": [0.9],
                "UL": 1,
                "LL": 0,
                "output": [],
                "output_conf": [{}]
            }
            ]
        }
        self.f = "models"

        #self.model = create_model_instance("Combination()", configuration, save = True)
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()


class CombinationTestClassPropperties(CombinationTestCase):
    def test_ANDPropperties(self):
        #check algorithms and determiner setup
        self.configuration["status_determiner"] = "AND()"
        self.model = create_model_instance("Combination()", self.configuration)
        self.assertIsInstance(self.model.anomaly_algorithms[0], BorderCheck)
        self.assertIsInstance(self.model.anomaly_algorithms[1], BorderCheck)
        self.assertIsInstance(self.model.status_determiner,AND)
    
    def test_ORPropperties(self):
        self.configuration["status_determiner"] = "OR()"
        self.model = create_model_instance("Combination()", self.configuration)
        self.assertIsInstance(self.model.anomaly_algorithms[0], BorderCheck)
        self.assertIsInstance(self.model.anomaly_algorithms[1], BorderCheck)
        self.assertIsInstance(self.model.status_determiner,OR)


class CombinationTestFunctionality(CombinationTestCase):
    def test_AND(self):
        self.configuration["status_determiner"] = "AND()"
        self.model = create_model_instance("Combination()", self.configuration)
        test_array = [0.2, 1, 1.5]
        expected_status = [1, 0, -1]
        for i in range(3):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(), [test_array[i]])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_status[i])

    def test_OR(self):
        self.configuration["status_determiner"] = "OR()"
        self.model = create_model_instance("Combination()", self.configuration)
        test_array = [0.2, 0.5, 1.5]
        expected_status = [1, 0, -1]
        for i in range(3):
            message = create_message((datetime.now()-datetime(1970,1,1)).total_seconds(), [test_array[i]])
            self.model.message_insert(message)
            self.assertEqual(self.model.status_code, expected_status[i])


class FeatureConstructionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # Border check does not use feature construction (or vector size > 1) 
        # but will be used because of simplicity
        configuration = {
            "input_vector_size": 2,
            "averages": [[2, 3], [2]],
            "periodic_averages": [[[2, [3]], [3, [2]]], []],
            "shifts": [[1, 2, 3, 4], []],
            "time_features": ["day", "month", "weekday", "hour", "minute"],
            "warning_stages": [0.7, 0.9],
            "UL": 4,
            "LL": 2,
            "output": [],
            "output_conf": [{}]
        }
        self.model = create_model_instance("BorderCheck()", configuration)

        self.f = "models"

        #Create a temporary /models folder.
        if not os.path.isdir(self.f):
            os.makedirs(self.f)
        self.model = create_model_instance("BorderCheck()", configuration)

        # Execute feature constructions (FV-s are saved and will be checked in
        # following tests)
        test_data = [[x, x+101] for x in range(10)]
        # timestamps are 1 day and 1 hour and 1 minute apart
        timestamps = timestamps = [1459926000 + (24*3600+60+3600)*x for x in range(20)]
        self.feature_vectors = []
        for sample_indx in range(10):
            feature_vector = self.model.feature_construction(value=test_data[sample_indx],
                                            timestamp=timestamps[sample_indx])
            self.feature_vectors.append(feature_vector)
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.isdir(self.f):
            shutil.rmtree(self.f)

        if os.path.isdir("configuration"):
            shutil.rmtree("configuration")

        return super().tearDown()
    
    def test_shifts(self):
        # First 4 feature vecotrs are undefined since memory does not contain
        # enough samples to construct all features
        for x in self.feature_vectors[:4]:
            self.assertFalse(x)
        
        # Extract shifted features
        shifts = [fv[7:11] for fv in self.feature_vectors[4:]]
        
        # Test shifted features
        self.assertListEqual(shifts[0], [3, 2, 1, 0])

        self.assertListEqual(shifts[1], [4, 3, 2, 1])

        self.assertListEqual(shifts[2], [5, 4, 3, 2])

        self.assertListEqual(shifts[3], [6, 5, 4, 3])

        self.assertListEqual(shifts[4], [7, 6, 5, 4])

        self.assertListEqual(shifts[5], [8, 7, 6, 5])

    def test_averages(self):
        # First 4 feature vecotrs are undefined since memory does not contain
        # enough samples to construct all features
        for x in self.feature_vectors[:4]:
            self.assertFalse(x)
        
        # Extract average features
        averages = [fv[2:5] for fv in self.feature_vectors[4:]]
        
        # Test average features
        self.assertListEqual(averages[0], [3.5, 3, 104.5])

        self.assertListEqual(averages[1], [4.5, 4, 105.5])

        self.assertListEqual(averages[2], [5.5, 5, 106.5])

        self.assertListEqual(averages[3], [6.5, 6, 107.5])

        self.assertListEqual(averages[4], [7.5, 7, 108.5])

        self.assertListEqual(averages[5], [8.5, 8, 109.5])

    def test_periodic_averages(self):
        # First 4 feature vecotrs are undefined since memory does not contain
        # enough samples to construct all features
        for x in self.feature_vectors[:4]:
            self.assertFalse(x)
        
        # Extract periodic average features
        periodic_averages = [fv[5:7] for fv in self.feature_vectors[4:]]
        
        # Test periodic average features
        self.assertListEqual(periodic_averages[0], [2, 2.5])

        self.assertListEqual(periodic_averages[1], [3, 3.5])

        self.assertListEqual(periodic_averages[2], [4, 4.5])

        self.assertListEqual(periodic_averages[3], [5, 5.5])

        self.assertListEqual(periodic_averages[4], [6, 6.5])

        self.assertListEqual(periodic_averages[5], [7, 7.5])

    def test_time_features(self):
        # First 4 feature vecotrs are undefined since memory does not contain
        # enough samples to construct all features
        for x in self.feature_vectors[:4]:
            self.assertFalse(x)
        
        # Extract time features
        time_features = [fv[11:] for fv in self.feature_vectors[4:]]
        
        # Test time features
        self.assertListEqual(time_features[0], [4, 10, 6, 11, 4])

        self.assertListEqual(time_features[1], [4, 11, 0, 12, 5])

        self.assertListEqual(time_features[2], [4, 12, 1, 13, 6])

        self.assertListEqual(time_features[3], [4, 13, 2, 14, 7])

        self.assertListEqual(time_features[4], [4, 14, 3, 15, 8])

        self.assertListEqual(time_features[5], [4, 15, 4, 16, 9])


class AverageNormalizationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        configuration = {
            "N": 4
        }
        self.normalization = LastNAverage()
        self.normalization.configure(conf=configuration)
        return super().setUp()


class AverageNormalizationClassProperties(AverageNormalizationTestCase):
    def test_properties(self):
        # Test setup from configuration
        self.assertEqual(self.normalization.N, 4)


class AverageNormalizationFunctionality(AverageNormalizationTestCase):
    def test_normalization(self):
        # Testing data and expected results
        test_data = [[x, 11+2*x] for x in range(9)]
        result_data = [
            [1, 10],
            [1, 12.25],
            [1.25, 12.5625],
            [1.3125, 12.453125],
            [1.1406, 11.81640625],
            [1.17578, 12.27050781]
        ]

        # Test add_value function
        self.normalization.add_value([1, 1])

        normalized_data = []
        for data in test_data:
            response = self.normalization.get_normalized(data)
            normalized_data.append(response)

        for fail in normalized_data[:3]:
            # First 3 entries must be False
            self.assertFalse(fail)

        # Test expected results
        for response_indx in range(6):
            self.assertAlmostEqual(normalized_data[3+response_indx][0], result_data[response_indx][0], 4)
            self.assertAlmostEqual(normalized_data[3+response_indx][1], result_data[response_indx][1], 4)


class PeriodicAverageNormalizationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        configuration = {
            "N": 4,
            "period": 2
        }
        self.normalization = PeriodicLastNAverage()
        self.normalization.configure(conf=configuration)
        return super().setUp()


class PeriodicAverageNormalizationClassProperties(PeriodicAverageNormalizationTestCase):
    def test_properties(self):
        # Test setup from configuration
        self.assertEqual(self.normalization.N, 4)
        self.assertEqual(self.normalization.period, 2)
        self.assertEqual(self.normalization.memory_len, 7)


class PeriodicAverageNormalizationFunctionality(PeriodicAverageNormalizationTestCase):
    def test_normalization(self):
        # Testing data and expected results
        test_data = [[x, 11+2*x] for x in range(11)]
        result_data = [
            [2.5, 13],
            [2.125, 14.5],
            [2.78125, 16.375],
            [2.8203125, 15.84375],
            [3.236328125, 17.0859375]
        ]

        # Test add_value function
        self.normalization.add_value([1, 1])

        normalized_data = []
        for data in test_data:
            response = self.normalization.get_normalized(data)
            normalized_data.append(response)

        for fail in normalized_data[:6]:
            # First 3 entries must be False
            self.assertFalse(fail)

        # Test expected results
        for response_indx in range(5):
            self.assertAlmostEqual(normalized_data[6+response_indx][0], result_data[response_indx][0], 4)
            self.assertAlmostEqual(normalized_data[6+response_indx][1], result_data[response_indx][1], 4)


class MsgCheckTestFunctionality(BCTestCase):
    def test_OK(self):
        message = {"timestamp" : (datetime.now()-datetime(1970,1,1)).total_seconds(),
                    "ftr_vector" : [1]}

        self.model.message_insert(message)
        self.assertEqual(self.model.check_ftr_vector(message), True)
    
    def test_NOK(self):
        #wrong name
        message = {"timestap" : (datetime.now()-datetime(1970,1,1)).total_seconds(),
                    "ftr_vector" : [1]}
        self.model.message_insert(message)
        self.assertEqual(self.model.check_ftr_vector(message), False)

        #print("a")
        message = {"timestamp" : (datetime.now()-datetime(1970,1,1)).total_seconds(),
                    "ftr_vect" : [1]}
        self.model.message_insert(message)
        self.assertEqual(self.model.check_ftr_vector(message), False)
        #print("a")

        #wrong ftr_vector length
        message = {"timestamp" : (datetime.now()-datetime(1970,1,1)).total_seconds(),
                    "ftr_vector" : [1, 1]}
        self.model.message_insert(message)
        self.assertEqual(self.model.check_ftr_vector(message), False)

        #wrong ftr_vector type
        message = {"timestamp" : (datetime.now()-datetime(1970,1,1)).total_seconds(),
                    "ftr_vector" : ["Im a string"]}
        self.model.message_insert(message)
        self.assertEqual(self.model.check_ftr_vector(message), False)

        #None in ftr_vector
        message = {"timestamp" : (datetime.now()-datetime(1970,1,1)).total_seconds(),
                    "ftr_vector" : [None]}
        self.model.message_insert(message)
        self.assertEqual(self.model.check_ftr_vector(message), False)

        #wrong timestamp
        message = {"timestamp" : datetime.now(),
                    "ftr_vector" : [1]}
        self.model.message_insert(message)
        self.assertEqual(self.model.check_ftr_vector(message), False)


if __name__ == '__main__':
    unittest.main(verbosity=2)