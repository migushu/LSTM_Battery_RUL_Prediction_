import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.externals import joblib
import random

class load_data():
    def __init__(self,filename,seq_len=50,split=0.3,usecols =[ 9, 10]):
        self.file_name = filename
        self.sequence_length = seq_len
        self.split = split
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.df = pd.read_csv(self.file_name, sep=',', usecols=usecols)
        # self.df = pd.read_csv(self.file_name, sep=',')
        self.data_all = np.array(self.df).astype(float)

    def get_x_y(self):
        num_train =int(len(self.data_all)*self.split)
        num_test =int(len(self.data_all) - num_train)
        train = self.data_all[:num_train,:]
        test = self.data_all[num_train-self.sequence_length+1:,:]
        train_x,train_y =self.get_train_x_y(train)
        test_x,test_y =self.get_test_x_y(test)
        return train_x,train_y,test_x,test_y

    def get_train_x_y(self, data_train,scale = 1,is_shuffle = False):
        data_scalered = data_train
        if scale ==1:
            data_scalered = self.scale_train_data(data_train)
        data = []
        for i in range(len(data_scalered) - self.sequence_length + 1):
            # data.append(data_scalered[i: i + self.sequence_length - 1])
            data.append(data_scalered[i: i + self.sequence_length ])
        reshaped_data = np.array(data).astype('float64')

        if is_shuffle:
            np.random.shuffle(reshaped_data)
            # random.shuffle(reshaped_data)

        train_x = reshaped_data[:, :, :-1]
        train_y = reshaped_data[:, len(reshaped_data[1]) - 1, -1:]##-2:-1
        return train_x, train_y

    def get_test_x_y(self,data_test):
        data_scalered = self.scale_test_data(data_test)
        data = []
        for i in range(len(data_scalered) - self.sequence_length + 1):
            # data.append(data_scalered[i: i + self.sequence_length - 1])
            data.append(data_scalered[i: i + self.sequence_length ])
        reshaped_data = np.array(data).astype('float64')

        test_x = reshaped_data[:, :, :-1]
        test_y = reshaped_data[:, len(reshaped_data[1]) - 1, -1:]
        return test_x, test_y

    def scale_train_data(self, data):
        data_x = self.scaler_x.fit_transform(data[:, :-1])
        data_y = self.scaler_y.fit_transform(data[:, -1:]) #label不归一化
        # data_x = self.scaler_x.transform(data[:, :-1])
        # data_y = self.scaler_y.transform(data[:, -1:])
        # data_y =data[:,-1:]
        data_all = np.concatenate((data_x, data_y), axis=1)
        return data_all

    def scale_test_data(self, data_test):
        data_x = self.scaler_x.transform(data_test[:,:-1])
        data_y = self.scaler_y.transform(data_test[:,-1:])
        # data_y = data_test[:, -1:]
        data_all = np.concatenate((data_x,data_y),axis=1)
        return data_all

    def get_scaler_x_y(self):
        return self.scaler_x,self.scaler_y

    def get_all_y(self):
        return  self.data_all[:,-1:]

    def load_scaler(self,file_path=''):
        scalerx = joblib.load("saved_model/base_seqlen20_batchsize128_epoch1_features1x.scale")
        scalery = joblib.load("saved_model/base_seqlen20_batchsize128_epoch1_features1y.scale")
        self.scaler_x = scalerx
        self.scaler_y = scalery
        return scalerx, scalery




def main():
    dataloader =load_data("2017_06_30_cell0_data.csv",20,0.5)
    train_x,train_y,test_x,test_y = dataloader.get_x_y()
    scaler_x,scaler_y = dataloader.get_scaler_x_y()

    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 7))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 7))
    predict_y =lstm.train_model(train_x,train_y,test_x,batch_size=16,epochs=50,pre_way=0)
    # mse = lstm.get_mse(test_y, predict_y)
    # mape = lstm.get_mape(test_y, predict_y)
    # print("scaled_mse= %0.3f" % mse)
    # print("scaled_mape= %0.3f%%"% mape)

    # test_y = scaler_y.inverse_transform(test_y)
    # predict_y = scaler_y.inverse_transform(predict_y)
    mse = lstm.get_rmse(test_y, predict_y)
    mape = lstm.get_mape(test_y, predict_y)
    print("mse= %0.3f" % mse)
    print("mape= %0.3f%%" % mape)
    print("len train_x: " + str(len(train_x)))
    print("len test_x: "+str(len(test_x)))

if __name__ == '__main__':
    main()