import numpy as np
import matplotlib.pyplot as plt
import lstm_network
import lstm_model
import load_data
import loss
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from keras.layers import LSTM,Dense,Dropout
from keras.models import Sequential
import math

def main():
    #TODO:
    time_step = 10
    feat_num = 1
    split = 0.5
    epochs = 100
    cell_num = 50
    # data_all = load_data.load_data("multi_battery/cells_multi_input.csv", time_step, usecols=[0,1,7]).data_all
    data_all = load_data.load_data('data/2017_06_30_cell0_data.csv',time_step,usecols=[3,9]).data_all
    print(data_all.shape)
    # plt.plot(data_all[:,0],data_all[:,1])
    # plt.show()

    #split to train and test.
    num_train = int(split * len(data_all))
    train = data_all[:num_train,1:]
    test = data_all[num_train:,1:]
    print(train.shape)

    #converting dataset to train_x and train_y
    # Here we only scale the train dataset, and not the entire dataset to prevent information leak
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(np.array(train).reshape(-1,1))
    print("scaler.mean= {0}, scaler.var= {1}".format(scaler.mean_,scaler.var_))

    #split into train_x and train_y
    x_train_scaled,y_train_scaled = get_supervised_x_y(train_scaled,time_step)
    print('y_train_scaled.shape= {0}'.format(y_train_scaled.shape))

    #scale test and split to test_x and test_y
    test_x_scaled, test_y, test_mu_list, test_std_list = get_x_scaled_y(test,time_step)
    print('test_x_scaled.shape= {0}'.format(test_x_scaled.shape))

    rmse,mape,pre  = train_pred_eval_model(x_train_scaled,y_train_scaled,test_x_scaled,
                                           test_y,test_mu_list,test_std_list,
                                           lstm_units=cell_num,epochs=epochs)
    print("rmse= {0}, mape={1}".format(rmse,mape))
    plt.plot(np.array(pre).reshape(-1,1),'r')
    plt.plot(np.array(test_y).reshape(-1,1))
    plt.show()


    # data_train = np.concatenate((data_all[:num_train,:1],scale.fit_transform(data_all[:num_train,1:])),axis=1)
    # data_test = np.concatenate((data_all[num_train:,:1],scale.transform(data_all[num_train:,1:])),axis=1)
    # train_x,train_y = get_supervised_x_y(data_train, time_step)
    # test_x,test_y = get_supervised_x_y(data_test, time_step)
    # print(len(train_x),len(train_y),len(test_x),len(test_y))


def get_supervised_x_y(data, time_step):
    """
       Split data into x (features) and y (target)
    """
    data_x, data_y = [], []
    for i in range(0,len(data)-time_step*2 ):
        data_x.append(data[i: i + time_step])
        data_y.append(data[i+time_step:i+time_step*2])
    data_x = np.array(data_x)
    data_y = np.array(data_y).reshape(-1,time_step)
    return data_x,data_y

def get_x_scaled_y(data, time_step):
    """
          Split data into x (features) and y (target)
          We scale x to have mean 0 and std dev 1, and return this.
          We do not scale y here.
          Inputs
              data     : pandas series to extract x and y
              time_step
          Outputs
              x_scaled : features used to predict y. Scaled such that each element has mean 0 and std dev 1
              y        : target values. Not scaled
              mu_list  : list of the means. Same length as x_scaled and y
              std_list : list of the std devs. Same length as x_scaled and y
          """
    x_scaled, y_noscaled, mu_list, std_list = [], [], [], []
    for i in range(0,len(data)-2*time_step,time_step):#减去2倍timestep,y最后一个长度才能保证为timestep
        index = int(i/time_step)
        mu_list.append(np.mean(data[i:i+time_step]))
        std_list.append(np.std(data[i:i+time_step]))
        x_scaled.append((data[i:i+time_step] - mu_list[index]) / std_list[index])
        y_noscaled.append(data[i+time_step:i+2*time_step,:])
    x_scaled = np.array(x_scaled)
    print(x_scaled.shape)
    y_noscaled = np.array(y_noscaled).reshape(x_scaled.shape[0],x_scaled.shape[1])
    print(y_noscaled.shape)


    return x_scaled, y_noscaled, mu_list, std_list

def train_pred_eval_model(x_train_scaled, y_train_scaled, x_test_scaled, y_test, mu_teste_list, std_test_list, lstm_units=10, \
                          dropout_prob=0.5, optimizer='rmsprop', epochs=50, batch_size=64):
    '''
           Train model, do prediction, scale back to original range and do evaluation
           Use LSTM here.
           Returns rmse, mape and predicted values
           Inputs
               x_train_scaled  : e.g. x_train_scaled.shape=(451, 9, 1). Here we are using the past 9 values to predict the next value
               y_train_scaled  : e.g. y_train_scaled.shape=(451, 1)
               x_cv_scaled     : use this to do predictions
               y_cv            : actual value of the predictions
               mu_cv_list      : list of the means. Same length as x_scaled and y
               std_cv_list     : list of the std devs. Same length as x_scaled and y
               lstm_units      : lstm param
               dropout_prob    : lstm param
               optimizer       : lstm param
               epochs          : lstm param
               batch_size      : lstm param
           Outputs
               rmse            : root mean square error
               mape            : mean absolute percentage error
               est             : predictions
           '''
    # Create the LSTM network
    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(x_train_scaled.shape[1], x_train_scaled.shape[2])))
    model.add(Dropout(dropout_prob))  # Add dropout with a probability of 0.5
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(dropout_prob))  # Add dropout with a probability of 0.5
    model.add(Dense(y_train_scaled.shape[1]))

    # Compile and fit the LSTM network
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.fit(x_train_scaled, y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=1)

    # Do prediction
    est_scaled = []
    test_x = x_test_scaled[0:1,:,:] #(1,10,1)
    for i in range(len(x_test_scaled)):
        pre = model.predict(test_x) #(1,10)
        est_scaled.append(pre)
        pre= np.reshape(pre,(1,x_test_scaled.shape[1],x_test_scaled.shape[2]))
        test_x = pre

    est_scaled = np.array(est_scaled).reshape(x_test_scaled.shape[0],y_train_scaled.shape[1])
    est = (est_scaled * np.array(std_test_list).reshape(-1, 1)) + np.array(mu_teste_list).reshape(-1, 1)
    rmse = math.sqrt(mean_squared_error(y_test, est))
    mape = loss.get_mape(y_test, est)

    return rmse, mape, est



    # for i in range(31):
    #     data = [row for row in data_all if row[0] == i]
    #     # print('{0}: {1}'.format(i,rows))
    #     print('{}: {}'.format(i,len(data)))

    # data_x,data_y = [],[]
    # for i in range(0,len(data_all)-time_step*2):
    #     data_x.append(data_all[i:i+time_step])
    #     data_y.append(data_all[i+time_step:i+time_step*2])
    # data_x = np.array(data_x).astype(float)
    # data_y = np.array(data_y).astype(float)
    # print(len(data_x))
    # boundary = int(split *len(data_x))
    # train_x = data_x[:boundary]
    # train_y = data_y[:boundary]
    # test_x = data_x[boundary:]
    # test_y = data_y[boundary:]
    #
    # train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], feat_num))
    # test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], feat_num))
    # train_y = np.reshape(train_y,(train_y.shape[0],train_y.shape[1]))
    # test_y = np.reshape(test_y,(test_y.shape[0],test_y.shape[1]))
    #
    #
    # model = lstm_network.lstm_network().build_network(time_step, feat_num, dropout_prob=0.2, dense_units=time_step)
    # lstm = lstm_model.lstm()
    # model = lstm.train_model(model, train_x, train_y, batch_size=32, epochs=epochs)
    # pre = lstm.predict(model, test_x[0:1], pre_way=0)
    # pre = np.reshape(pre,(-1,time_step))
    # pre= scale_y.inverse_transform(pre)
    # rmse = loss.get_rmse(test_y,pre)
    # print(rmse)
    #
    # test_plt_y = test_y[:,0:1]
    # for i in test_y[-1:,1:]:
    #     test_plt_y = np.append(test_plt_y, i)
    #
    # for i in range(1):
    #     plt.plot(pre[i,:],'r')
    #
    # plt.plot(data_all)
    # plt.plot(range(time_step,time_step+len(train_y[:,0:1])),train_y[:,0:1])
    # plt.plot(range(time_step+len(train_y[:,0:1]),time_step+len(train_y[:,0:1])+len(test_plt_y)),test_plt_y)
    # plt.show()

if __name__ == '__main__':
    main()