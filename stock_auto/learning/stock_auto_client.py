import numpy
import matplotlib.pyplot as plt
import pandas
import math
import tensorflow as tf
from tensorflow.python.framework import graph_io
import tf_graph_util as graph_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from graphpipe import remote

forecast_num=5

for x in range(forecast_num):

    # Read data for prediction
    validate_data = pandas.read_csv('./stock_auto_test.csv', usecols=[1,2,3,4,5], engine='python', names=('nissan','toyota','mazda','honda','subaru'))
    validate_data = validate_data.tail(10)

    print(validate_data.head())

    #-------------------------------------------------
    dataset = validate_data.values
    dataset = validate_data.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    dataset = dataset.astype('float32')

    #-------------------------------------------------

    # convert an array of values into a dataset matrix
    def create_test_dataset(dataset, look_back=10):
        dataX = []
        for i in range(len(dataset)+1-look_back):
            xset = []
            for j in range(dataset.shape[1]):
                a = dataset[i:(i+look_back), j]
                xset.append(a)
            dataX.append(xset)
        return numpy.array(dataX)

    # reshape into X=t and Y=t+1
    look_back = 10
    testX = create_test_dataset(dataset, look_back)
    print(testX.shape)
    print(testX[0])

    # reshape input to be [samples, time steps(number of variables), features] *convert time series into column
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

    #-------------------------------------------------

    #-------------------------------------------------
    # Execute Prediction to model server (graphpipe)
    predictions, = remote.execute_multi("http://127.0.0.1:9041", [testX], ['lstm_input'], ['dense/BiasAdd'])

    # make predictions
    #trainPredict = model.predict(trainX)
    #testPredict = model.predict(testX)
    #pad_col = numpy.zeros(dataset.shape[1]-1)

    # invert predictions
    #def pad_array(val):
    #    return numpy.array([numpy.insert(pad_col, 0, x) for x in val])
    
    #trainPredict = scaler.inverse_transform(pad_array(trainPredict))
    #trainY = scaler.inverse_transform(pad_array(trainY))
    #testPredict = scaler.inverse_transform(pad_array(testPredict))
    #testY = scaler.inverse_transform(pad_array(testY))

    #trainPredict = scaler.inverse_transform(trainPredict)
    #trainY = scaler.inverse_transform(trainY)
    #testPredict = scaler.inverse_transform(testPredict)
    #testY = scaler.inverse_transform(testY)

    testPredict = scaler.inverse_transform(predictions)

    #-------------------------------------------------
    print(testPredict)

    result=testPredict[-1]
    result=result.astype('int16')

    out="N+"+str(x+1)
    for i in range(len(result)):
        out=out+","+str(result[i])

    with open('./stock_auto_test.csv', 'a') as f:
        print(out, file=f)










# shift train predictions for plotting
#trainPredictPlot = numpy.empty_like(dataset)
#trainPredictPlot[:, :] = numpy.nan
#trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
#testPredictPlot = numpy.empty_like(dataset)
#testPredictPlot[:, :] = numpy.nan
#testPredictPlot[len(trainPredict)+(look_back*2):len(dataset), :] = testPredict
# plot baseline and predictions
#plt.plot(scaler.inverse_transform(dataset))
#plt.plot(trainPredictPlot)
#plt.plot(testPredictPlot)
#plt.show()

#plt.close()


#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------
