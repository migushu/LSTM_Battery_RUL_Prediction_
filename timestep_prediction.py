import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
import load_data
import loss
import lstm_model
import lstm_network
import utils

'''
循环预测：
用前N timestep 步来预测接下来的N步，将预测的N步用作下一个N步预测的输入
用前N timestep 步来预测接下来的M步，将预测的M步添加到测试集中，用作下一个预测的部分输入
'''

def main():
    #DONE:实现输入和预测长度不一样的情况
    time_step = 30
    pre_step = 10
    split = 0.6
    epochs = 100
    cell_num = 50
    dropout_prob = 0.5
    batch_size = 64

    #预训练
    import run
    filename = "multi_battery/cells_multi_input.csv"
    dataloader = load_data.load_data(filename, time_step, split, usecols=[0,7])
    train_x, train_y,scale = run.get_base_train_data(dataloader, time_step,pre_step,is_timestep_y=True)
    model = None
    lstm = lstm_model.lstm()
    #训练出一个基本模型
    # model = lstm_network.lstm_network().build_network(time_step=train_x.shape[1],
    #                            features_num=train_x.shape[2],
    #                            dropout_prob=dropout_prob,
    #                            dense_units=pre_step,
    #                            lstm_units=cell_num,
    #                            optimizer='rmsprop'
    #                            )
    # model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)
    save_path = "saved_model/timestep_pre_timestep:{0}_prestep:{1}_batchsize:{2}".format(time_step,pre_step,batch_size)
    # lstm.save_model(model,save_path,scale,scale)
    #
    model = lstm.load_model(save_path+".json",save_path+".h5")

    #针对具体cell预测。
    data_all = load_data.load_data('data/2017_06_30_cell30_data.csv',time_step,usecols=[3,9]).data_all
    #split to train and test.
    num_train = int(split * len(data_all))
    train = data_all[:num_train,1:]
    test = data_all[num_train-time_step:,1:]
    print(train.shape)

    #converting dataset to train_x and train_y
    # Here we only scale the train dataset, and not the entire dataset to prevent information leak
    # scaler = StandardScaler()
    train_scaled = scale.transform(np.array(train).reshape(-1,1))
    # print("scaler.min= {0}".format(scale.min_))

    #split into train_x and train_y
    x_train_scaled,y_train_scaled = get_supervised_timestep_x_y(train_scaled, time_step,pre_step)
    print('y_train_scaled.shape= {0}'.format(y_train_scaled.shape))

    #scale test and split to test_x and test_y
    test_x_scaled, test_y, test_mu_list, test_std_list = get_x_scaled_y(test,time_step,pre_step,scale)
    print('test_x_scaled.shape= {0}'.format(test_x_scaled.shape))

    rmse,mape,pre,model  = train_pred_eval_model(x_train_scaled,y_train_scaled,test_x_scaled,
                                           test_y,test_mu_list,test_std_list,scale,
                                           lstm_units=cell_num,epochs=epochs,model=model)
    print("rmse= {0}, mape={1}".format(rmse,mape))
    lstm = lstm_model.lstm()
    save_path = save_path+"_predict"
    lstm.save_model(model,save_path,scale,scale)

    fig = plt.figure(2)
    plt.plot(train,'b--')
    plt_pre = np.array(pre).reshape(-1,1)
    plt_test = np.array(test_y).reshape(-1,1)
    plt.plot(range(len(train),len(train)+len(plt_pre)),plt_pre,'r')
    plt.plot(range(len(train),len(train)+len(plt_test)),plt_test,'y')
    plt.legend(["train","predict","test"])

    name = "timestep:{0}_{1}.png".format(time_step,utils.get_time())
    plt.savefig("result/multi_pre/" + name)
    plt.show()
    plt.close()


    # data_train = np.concatenate((data_all[:num_train,:1],scale.fit_transform(data_all[:num_train,1:])),axis=1)
    # data_test = np.concatenate((data_all[num_train:,:1],scale.transform(data_all[num_train:,1:])),axis=1)
    # train_x,train_y = get_supervised_x_y(data_train, time_step)
    # test_x,test_y = get_supervised_x_y(data_test, time_step)
    # print(len(train_x),len(train_y),len(test_x),len(test_y))

def save_model(model,save_path,scale_x,scale_y):
    model_json = model.to_json()
    with open(save_path + ".json", "w") as f:  # .json
        f.write(model_json)
    joblib.dump(scale_x, save_path + "x.scale")
    joblib.dump(scale_y, save_path + "y.scale")
    model.save_weights(save_path + ".h5")  # .h5


def get_supervised_timestep_x_y(data, time_step,pre_step):
    """
       Split data into x (features) and y (target)
    """
    data_x, data_y,data_all = [], [], []
    for i in range(0,len(data)-time_step-pre_step ):
        data_all.append(data[i:i+time_step+pre_step])
        # data_x.append(data[i: i + time_step])
        # data_y.append(data[i+time_step:i+time_step+pre_step])
    # data_x = np.array(data_x)
    # data_y = np.array(data_y).reshape(-1,pre_step,1)
    # data = np.concatenate((data_x,data_y),axis=2)
    data_all = np.array(data_all)
    # random.shuffle(data)#DONE:搞混了x和y
    # np.random.shuffle(data)
    np.random.shuffle(data_all)
    data_x = data_all[:,:time_step]
    data_y = np.array(data_all[:,time_step:])
    data_y = data_y.reshape(data_y.shape[0],data_y.shape[1])
    # data_x = data[:,:,0:1]
    # data_y = data[:,:,1]
    return data_x,data_y

def get_x_scaled_y(data, time_step,pre_step,scale):
    """
          Split data into x (features) and y (target)
          """
    x_scaled, y_unscaled, mu_list, std_list = [], [], [], []
    for i in range(0,len(data)-time_step-pre_step,pre_step):#减去2倍timestep,y最后一个长度才能保证为timestep
        mu_list.append(np.mean(data[i:i+time_step]))
        std_list.append(np.std(data[i:i+time_step]))
        # x_scaled.append((data[i:i+time_step] - mu_list[index]) / std_list[index])
        x_scaled.append(scale.transform(data[i:i+time_step,:]))
        y_unscaled.append(data[i+time_step:i+time_step+pre_step,:])
    x_scaled = np.array(x_scaled)
    print(x_scaled.shape)
    y_unscaled = np.array(y_unscaled).reshape(-1,pre_step)
    print(y_unscaled.shape)


    return x_scaled, y_unscaled, mu_list, std_list

def train_pred_eval_model(x_train_scaled, y_train_scaled, x_test_scaled, y_test, mu_teste_list, std_test_list, scale,lstm_units=10, \
                          dropout_prob=0.5, optimizer='rmsprop', epochs=50, batch_size=64,model = None):
    '''
           Train model, do prediction, scale back to original range and do evaluation
           Use LSTM here.
           Returns rmse, mape and predicted values
           Outputs
               rmse            : root mean square error
               mape            : mean absolute percentage error
               est             : predictions
    '''
    if model == None:
        model = lstm_network.lstm_network().build_network(time_step=x_train_scaled.shape[1],
                                                          features_num=x_train_scaled.shape[2],
                                                          dropout_prob=dropout_prob,
                                                          dense_units=y_train_scaled.shape[1],
                                                          lstm_units=lstm_units,
                                                          optimizer=optimizer)

    # Compile and fit the LSTM network
    model.fit(x_train_scaled, y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=1)

    # Do prediction
    timesteps = x_test_scaled.shape[1]
    est_scaled = []
    pre_num = len(y_test)
    predict_y = []
    pre_xlist = []#预测的输入列表
    pre_xlist.extend(x_test_scaled[0,:,0].tolist())
    # test_x = x_test_scaled[0:1,:,:] #(1,10,1)
    for i in range(pre_num):
        predictx = np.array(pre_xlist[-timesteps:])
        predictx = np.reshape(predictx,(1,timesteps,1))
        pre_y = model.predict(predictx)
        pre_xlist.extend(pre_y[0])
        est_scaled.extend(pre_y)

        # pre = model.predict(test_x) #(1,10)
        # est_scaled.append(pre)
        # pre= np.reshape(pre,(1,x_test_scaled.shape[1],x_test_scaled.shape[2]))
        # test_x = pre

    est_scaled = np.array(est_scaled).reshape(-1,1)
    # est = (est_scaled * np.array(std_test_list).reshape(-1, 1)) + np.array(mu_teste_list).reshape(-1, 1)
    est = scale.inverse_transform(est_scaled)
    # est = np.array(est).reshape(x_test_scaled.shape[0],x_test_scaled.shape[1])
    y_test = np.reshape(y_test,(-1,1))
    rmse = math.sqrt(mean_squared_error(y_test, est))
    mape = loss.get_mape(y_test, est)

    return rmse, mape, est, model


if __name__ == '__main__':
    main()
