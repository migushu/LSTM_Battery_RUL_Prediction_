from keras.models import Sequential,model_from_json
from keras.layers import Dropout,LSTM,Dense,Activation
import numpy as np

class lstm_network():

    def build_network(self, time_step, features_num =1, dropout_prob=0.2,dense_units = 1,lstm_units = 50,optimizer='rmsprop'):
        # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
        model = Sequential()
        #每个时间步经过第一个LSTM后，得到的中间隐向量是50维，一个20个时间步的样本数据进去得到（20×50）的数据
        model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, features_num)))
        print(model.layers)
        model.add(Dropout(float(dropout_prob)))
        model.add(LSTM(units=100))
        model.add(Dropout(float(dropout_prob)))
        model.add(Dense(units=dense_units))
        model.add(Activation('linear', name='LSTMActivation'))

        model.compile(loss='mse', optimizer=optimizer)
        model.summary()
        return model

