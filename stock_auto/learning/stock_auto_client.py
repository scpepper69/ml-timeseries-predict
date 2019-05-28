import numpy
import matplotlib.pyplot as plt
import pandas
import math
import tensorflow as tf
from tensorflow.python.framework import graph_io
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from graphpipe import remote

forecast_num=5

for x in range(forecast_num):

    #-------------------------------------------------
    # Read data for prediction
    validate_data = pandas.read_csv('./stock_auto_test.csv', usecols=[1,2,3,4,5], engine='python', names=('nissan','toyota','mazda','honda','subaru'))
    validate_data = validate_data.tail(30)

    dataset = validate_data.values
    dataset = validate_data.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    dataset = dataset.astype('float32')

    #-------------------------------------------------
    # convert an array of values into a dataset matrix
    def create_test_dataset(dataset, look_back=30):
        dataX = []
        for i in range(len(dataset)+1-look_back):
            xset = []
            for j in range(dataset.shape[1]):
                a = dataset[i:(i+look_back), j]
                xset.append(a)
            dataX.append(xset)
        return numpy.array(dataX)

    # reshape into X=t and Y=t+1
    look_back = 30
    testX = create_test_dataset(dataset, look_back)
    print(testX.shape)
    print(testX[0])

    # reshape input to be [samples, time steps(number of variables), features] *convert time series into column
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

    #-------------------------------------------------
    # Execute Prediction to model server (graphpipe)
    predictions, = remote.execute_multi("http://127.0.0.1:9041", [testX], ['lstm_input'], ['dense/BiasAdd'])
    testPredict = scaler.inverse_transform(predictions)

    #-------------------------------------------------
    # Export predict data
    result=testPredict[-1]
    result=result.astype('int16')

    out="N+"+str(x+1)
    for i in range(len(result)):
        out=out+","+str(result[i])

    with open('./stock_auto_test.csv', 'a') as f:
        print(out, file=f)

# Create result graph image
result = pandas.read_csv('./stock_auto_test.csv', usecols=[1,2,3,4,5], engine='python', names=('nissan','toyota','mazda','honda','subaru'))
result.plot()
result_set = result.values
plt.vlines(30, 0, np.max(result_set), "blue", linestyles='dashed')
plt.legend(loc='upper left')
plt.savefig('./result.png')
plt.close()

