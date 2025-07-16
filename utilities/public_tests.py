import numpy as np
import math

def sigmoid_test(target):
    assert np.isclose(target(3.0), 0.9525741268224334), "Failed for scalar input"
    assert np.allclose(target(np.array([2.5, 0])), [0.92414182, 0.5]), "Failed for 1D array"
    assert np.allclose(target(np.array([[2.5, -2.5], [0, 1]])), 
                       [[0.92414182, 0.07585818], [0.5, 0.73105858]]), "Failed for 2D array"
    print('\033[92mAll tests passed!')
    
def compute_cost_test(target):
    X = np.array([[0, 0, 0, 0]]).T
    y = np.array([0, 0, 0, 0])
    w = np.array([0])
    b = 1
    result = target(X, y, w, b)
    if math.isinf(result):
        raise ValueError("Did you get the sigmoid of z_wb?")
    
    np.random.seed(17)  
    X = np.random.randn(5, 2)
    y = np.array([1, 0, 0, 1, 1])
    w = np.random.randn(2)
    b = 0
    result = target(X, y, w, b)
    assert np.isclose(result, 2.15510667), f"Wrong output. Expected: {2.15510667} got: {result}"
    
    X = np.random.randn(4, 3)
    y = np.array([1, 1, 0, 0])
    w = np.random.randn(3)
    b = 0
    
    result = target(X, y, w, b)
    assert np.isclose(result, 0.80709376), f"Wrong output. Expected: {0.80709376} got: {result}"

    X = np.random.randn(4, 3)
    y = np.array([1, 0,1, 0])
    w = np.random.randn(3)
    b = 3
    result = target(X, y, w, b)
    assert np.isclose(result, 0.4529660647), f"Wrong output. Expected: {0.4529660647} got: {result}. Did you inizialized z_wb = b?"
    
    print('\033[92mAll tests passed!')
    
def compute_gradient_test(target):
    np.random.seed(1)
    X = np.random.randn(7, 3)
    y = np.array([1, 0, 1, 0, 1, 1, 0])
    test_w = np.array([1, 0.5, -0.35])
    test_b = 1.7
    dj_db, dj_dw  = target(X, y, test_w, test_b)
    
    assert np.isclose(dj_db, 0.28936094), f"Wrong value for dj_db. Expected: {0.28936094} got: {dj_db}" 
    assert dj_dw.shape == test_w.shape, f"Wrong shape for dj_dw. Expected: {test_w.shape} got: {dj_dw.shape}" 
    assert np.allclose(dj_dw, [-0.11999166, 0.41498775, -0.71968405]), f"Wrong values for dj_dw. Got: {dj_dw}"

    print('\033[92mAll tests passed!') 
    
def predict_test(target):
    np.random.seed(5)
    b = 0.5    
    w = np.random.randn(3)
    X = np.random.randn(8, 3)
    
    result = target(X, w, b)
    wrong_1 = [1., 1., 0., 0., 1., 0., 0., 1.]
    expected_1 = [1., 1., 1., 0., 1., 0., 0., 1.]
    if np.allclose(result, wrong_1):
        raise ValueError("Did you apply the sigmoid before applying the threshold?")
    assert result.shape == (len(X),), f"Wrong length. Expected : {(len(X),)} got: {result.shape}"
    assert np.allclose(result, expected_1), f"Wrong output: Expected : {expected_1} got: {result}"
    
    b = -1.7    
    w = np.random.randn(4) + 0.6
    X = np.random.randn(6, 4)
    
    result = target(X, w, b)
    expected_2 = [0., 0., 0., 1., 1., 0.]
    assert result.shape == (len(X),), f"Wrong length. Expected : {(len(X),)} got: {result.shape}"
    assert np.allclose(result,expected_2), f"Wrong output: Expected : {expected_2} got: {result}"

    print('\033[92mAll tests passed!')
    
def compute_cost_reg_test(target):
    np.random.seed(1)
    w = np.random.randn(3)
    b = 0.4
    X = np.random.randn(6, 3)
    y = np.array([0, 1, 1, 0, 1, 1])
    lambda_ = 0.1
    expected_output = target(X, y, w, b, lambda_)
    
    assert np.isclose(expected_output, 0.5469746792761936), f"Wrong output. Expected: {0.5469746792761936} got:{expected_output}"
    
    w = np.random.randn(5)
    b = -0.6
    X = np.random.randn(8, 5)
    y = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    lambda_ = 0.01
    output = target(X, y, w, b, lambda_)
    assert np.isclose(output, 1.2608591964119995), f"Wrong output. Expected: {1.2608591964119995} got:{output}"
    
    w = np.array([2, 2, 2, 2, 2])
    b = 0
    X = np.zeros((8, 5))
    y = np.array([0.5] * 8)
    lambda_ = 3
    output = target(X, y, w, b, lambda_)
    expected = -np.log(0.5) + 3. / (2. * 8.) * 20.
    assert np.isclose(output, expected), f"Wrong output. Expected: {expected} got:{output}"
    
    print('\033[92mAll tests passed!') 
    
def compute_gradient_reg_test(target):
    np.random.seed(1)
    w = np.random.randn(5)
    b = 0.2
    X = np.random.randn(7, 5)
    y = np.array([0, 1, 1, 0, 1, 1, 0])
    lambda_ = 0.1
    expected1 = (-0.1506447567869257, np.array([ 0.19530838, -0.00632206,  0.19687367,  0.15741161,  0.02791437]))
    dj_db, dj_dw = target(X, y, w, b, lambda_)
    
    assert np.isclose(dj_db, expected1[0]), f"Wrong dj_db. Expected: {expected1[0]} got: {dj_db}"
    assert np.allclose(dj_dw, expected1[1]), f"Wrong dj_dw. Expected: {expected1[1]} got: {dj_dw}"

    
    w = np.random.randn(7)
    b = 0
    X = np.random.randn(7, 7)
    y = np.array([1, 0, 0, 0, 1, 1, 0])
    lambda_ = 0
    expected2 = (0.02660329857573818, np.array([ 0.23567643, -0.06921029, -0.19705212, -0.0002884 ,  0.06490588,
        0.26948175,  0.10777992]))
    dj_db, dj_dw = target(X, y, w, b, lambda_)
    assert np.isclose(dj_db, expected2[0]), f"Wrong dj_db. Expected: {expected2[0]} got: {dj_db}"
    assert np.allclose(dj_dw, expected2[1]), f"Wrong dj_dw. Expected: {expected2[1]} got: {dj_dw}"
    
    print('\033[92mAll tests passed!') 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

import numpy as np

def test_eval_mse(target):
    y_hat = np.array([2.4, 4.2])
    y_tmp = np.array([2.3, 4.1])
    result = target(y_hat, y_tmp)
    
    assert np.isclose(result, 0.005, atol=1e-6), f"Wrong value. Expected 0.005, got {result}"
    
    y_hat = np.array([3.] * 10)
    y_tmp = np.array([3.] * 10)
    result = target(y_hat, y_tmp)
    assert np.isclose(result, 0.), f"Wrong value. Expected 0.0 when y_hat == t_tmp, but got {result}"
    
    y_hat = np.array([3.])
    y_tmp = np.array([0.])
    result = target(y_hat, y_tmp)
    assert np.isclose(result, 4.5), f"Wrong value. Expected 4.5, but got {result}. Remember the square termn"
    
    y_hat = np.array([3.] * 5)
    y_tmp = np.array([2.] * 5)
    result = target(y_hat, y_tmp)
    assert np.isclose(result, 0.5), f"Wrong value. Expected 0.5, but got {result}. Remember to divide by (2*m)"
    
    print("\033[92m All tests passed.")
    
def test_eval_cat_err(target):
    y_hat = np.array([1, 0, 1, 1, 1, 0])
    y_tmp = np.array([0, 1, 0, 0, 0, 1])
    result = target(y_hat, y_tmp)
    assert not np.isclose(result, 6.), f"Wrong value. Expected 1, but got {result}. Did you divided by m?"
    
    y_hat = np.array([1, 2, 0])
    y_tmp = np.array([1, 2, 3])
    result = target(y_hat, y_tmp)
    assert np.isclose(result, 1./3., atol=1e-6), f"Wrong value. Expected 0.333, but got {result}"
    
    y_hat = np.array([1, 0, 1, 1, 1, 0])
    y_tmp = np.array([1, 1, 1, 0, 0, 0])
    result = target(y_hat, y_tmp)
    assert np.isclose(result, 3./6., atol=1e-6), f"Wrong value. Expected 0.5, but got {result}"
    
    y_hat = np.array([[1], [2], [0], [3]])
    y_tmp = np.array([[1], [2], [1], [3]])
    res_tmp =  target(y_hat, y_tmp)
    assert type(res_tmp) != np.ndarray, f"The output must be an scalar but got {type(res_tmp)}"
    
    print("\033[92m All tests passed.")
    
def model_test(target, classes, input_size):
    target.build(input_shape=(None,input_size))
    expected_lr = 0.01
    
    assert len(target.layers) == 3, \
        f"Wrong number of layers. Expected 3 but got {len(target.layers)}"
    assert target.input.shape.as_list() == [None, input_size], \
        f"Wrong input shape. Expected [None,  {input_size}] but got {target.input.shape.as_list()}"
    i = 0
    expected = [[Dense, [None, 120], relu],
                [Dense, [None, 40], relu],
                [Dense, [None, classes], linear]]

    for layer in target.layers:
        assert type(layer) == expected[i][0], \
            f"Wrong type in layer {i}. Expected {expected[i][0]} but got {type(layer)}"
        assert layer.output.shape.as_list() == expected[i][1], \
            f"Wrong number of units in layer {i}. Expected {expected[i][1]} but got {layer.output.shape.as_list()}"
        assert layer.activation == expected[i][2], \
            f"Wrong activation in layer {i}. Expected {expected[i][2]} but got {layer.activation}"
        assert layer.kernel_regularizer == None, "You must not specify any regularizer for any layer"
        i = i + 1
        
    assert type(target.loss)==SparseCategoricalCrossentropy, f"Wrong loss function. Expected {SparseCategoricalCrossentropy}, but got {target.loss}"
    assert type(target.optimizer)==Adam, f"Wrong loss function. Expected {Adam}, but got {target.optimizer}"
    lr = target.optimizer.learning_rate.numpy()
    assert np.isclose(lr, expected_lr, atol=1e-8), f"Wrong learning rate. Expected {expected_lr}, but got {lr}"
    assert target.loss.get_config()['from_logits'], f"Set from_logits=True in loss function"

    print("\033[92mAll tests passed!")
    
def model_s_test(target, classes, input_size):
    target.build(input_shape=(None,input_size))
    expected_lr = 0.01
    
    assert len(target.layers) == 2, \
        f"Wrong number of layers. Expected 3 but got {len(target.layers)}"
    assert target.input.shape.as_list() == [None, input_size], \
        f"Wrong input shape. Expected [None,  {input_size}] but got {target.input.shape.as_list()}"
    i = 0
    expected = [[Dense, [None, 6], relu],
                [Dense, [None, classes], linear]]

    for layer in target.layers:
        assert type(layer) == expected[i][0], \
            f"Wrong type in layer {i}. Expected {expected[i][0]} but got {type(layer)}"
        assert layer.output.shape.as_list() == expected[i][1], \
            f"Wrong number of units in layer {i}. Expected {expected[i][1]} but got {layer.output.shape.as_list()}"
        assert layer.activation == expected[i][2], \
            f"Wrong activation in layer {i}. Expected {expected[i][2]} but got {layer.activation}"
        assert layer.kernel_regularizer == None, "You must not specify any regularizer any layer"
        i = i + 1
        
    assert type(target.loss)==SparseCategoricalCrossentropy, f"Wrong loss function. Expected {SparseCategoricalCrossentropy}, but got {target.loss}"
    assert type(target.optimizer)==Adam, f"Wrong loss function. Expected {Adam}, but got {target.optimizer}"
    lr = target.optimizer.learning_rate.numpy()
    assert np.isclose(lr, expected_lr, atol=1e-8), f"Wrong learning rate. Expected {expected_lr}, but got {lr}"
    assert target.loss.get_config()['from_logits'], f"Set from_logits=True in loss function"

    print("\033[92mAll tests passed!")
    
def model_r_test(target, classes, input_size):
    target.build(input_shape=(None,input_size))
    expected_lr = 0.01
    print("ddd")
    assert len(target.layers) == 3, \
        f"Wrong number of layers. Expected 3 but got {len(target.layers)}"
    assert target.input.shape.as_list() == [None, input_size], \
        f"Wrong input shape. Expected [None,  {input_size}] but got {target.input.shape.as_list()}"
    i = 0
    expected = [[Dense, [None, 120], relu, (tf.keras.regularizers.l2, 0.1)],
                [Dense, [None, 40], relu, (tf.keras.regularizers.l2, 0.1)],
                [Dense, [None, classes], linear, None]]

    for layer in target.layers:
        assert type(layer) == expected[i][0], \
            f"Wrong type in layer {i}. Expected {expected[i][0]} but got {type(layer)}"
        assert layer.output.shape.as_list() == expected[i][1], \
            f"Wrong number of units in layer {i}. Expected {expected[i][1]} but got {layer.output.shape.as_list()}"
        assert layer.activation == expected[i][2], \
            f"Wrong activation in layer {i}. Expected {expected[i][2]} but got {layer.activation}"
        if not (expected[i][3] == None):
            assert type(layer.kernel_regularizer) == expected[i][3][0], f"Wrong regularizer. Expected L2 regularizer but got {type(layer.kernel_regularizer)}"
            assert np.isclose(layer.kernel_regularizer.l2,  expected[i][3][1]), f"Wrong regularization factor. Expected {expected[i][3][1]}, but got {layer.kernel_regularizer.l2}"
        else:
            assert layer.kernel_regularizer == None, "You must not specify any regularizer for the 3th layer"
        i = i + 1
        
    assert type(target.loss)==SparseCategoricalCrossentropy, f"Wrong loss function. Expected {SparseCategoricalCrossentropy}, but got {target.loss}"
    assert type(target.optimizer)==Adam, f"Wrong loss function. Expected {Adam}, but got {target.optimizer}"
    lr = target.optimizer.learning_rate.numpy()
    assert np.isclose(lr, expected_lr, atol=1e-8), f"Wrong learning rate. Expected {expected_lr}, but got {lr}"
    assert target.loss.get_config()['from_logits'], f"Set from_logits=True in loss function"

    print("\033[92mAll tests passed!")