import numpy as np
import matplotlib.pyplot as plt
import lstm_network
import lstm_model
import load_data
import loss
from sklearn.preprocessing import MinMaxScaler
def main():
    time_step = 50
    feat_num = 1
    split = 0.5
    epochs = 1
    data_all = load_data.load_data("multi_battery/cells_multi_input.csv", time_step, usecols=[0,1,7]).data_all

    # rows  = [row for row in data_all if row[0]== 0]
    # print(rows)

    print(data_all.shape)
    scale = MinMaxScaler()
    data_all = np.concatenate((data_all[:,:2],scale.fit_transform(data_all[:,2:])),axis=1)

    for i in range(31):
        data = [row for row in data_all if row[0] == i]
        # print('{0}: {1}'.format(i,rows))
        print('{}: {}'.format(i,len(data)))




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