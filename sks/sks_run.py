import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np
import lstm_network
import loss
import utils
import argparse


data_path = "../data2/"
cell_filename = "Statistics_1-046"
suffix = '.csv'
cell_num = ['043','046']
#Cycle_Index,Test_Time(s),Date_Time,Current(A),Voltage(V),
# Charge_Capacity(Ah),Discharge_Capacity(Ah),Charge_Energy(Wh),Discharge_Energy(Wh),Internal_Resistance(Ohm),
# AC_Impedance(Ohm),ACI_Phase_Angle(Deg),Charge_Time(s),DisCharge_Time(s),Vmax_On_Cycle(V)
cycle_cap_header = ['Cycle_Index', 'Discharge_Capacity(Ah)']
usecols = [0,6]
multi_usecols = [0,1,3,4,5,6,7,8,12,13,14]
feature_num =len(multi_usecols)-1
col_index  = 5 # 1 除去第0列的cycle。
# split = 0.5
# timestep = 30
# pre_step = 1
# dense_units = pre_step
# batchsize = 64
# epochs = 100
#TODO:数据用pd表示

def error_2_csv(eol_cap,start_pre_cycle,eol_cycle,rul,pre_rul,rul_error,failure = 0.7):
    #header:cell,timestep,epochs,batchsize,split,eol_cap,start_pre_cycle,eol_cycle,rul,pre_rul,rul_error
    error_file = '../result/sks_error/error.csv'
    error_str = '\n{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}'.format(cell_filename,failure,timestep,
                                                                 epochs,batchsize,split,
                                                                 eol_cap,start_pre_cycle,
                                                                 eol_cycle,rul,
                                                                 pre_rul,rul_error)
    with open(error_file,'a+') as f:
        f.writelines(error_str)

def get_rul(data,split,col_index = 0,failure = 0.7):
    start_pre_point =int(len(data)*split)
    start_pre_cycle  =int(data[start_pre_point,0])
    eol_cap_norm = float(failure * data[0,col_index])
    eol_rows = np.where(data[:,col_index]<=eol_cap_norm)
    eol_row = (np.array(eol_rows).reshape(-1,1))[0]
    eol_cycle = int(data[eol_row,0])
    eol_cap = float(data[eol_row,col_index])
    rul = eol_cycle - start_pre_cycle
    with open('../data2/RUL/'+cell_filename+'_split:'+str(split)+'_failure:'+str(failure)+'.csv','w') as f:
        f.writelines('start_predict_cycle,eol_cycle,rul,eol_capacity,eol_norm\n')
        f.writelines('{0},{1},{2},{3},{4}'.format(start_pre_cycle,eol_cycle,rul,eol_cap,eol_cap_norm))
    return start_pre_cycle,eol_cycle,rul, eol_cap,eol_cap_norm

def get_pre_rul(data,eol_cap_norm):
    data = np.array(data).reshape(-1,)#若为（-1,1），则pre_cycle会返回两个tuple，
    pre_index = np.where(data<=eol_cap_norm)#返回满足条件的坐标，维数同data的维度
    pre_cycle = np.array(pre_index)
    if len(pre_cycle[0]) == 0:
        pre_cycle = len(data)
    pre_cycle = np.array(pre_cycle).reshape(-1,1)
    return pre_cycle[0,0]

def get_origin_train_test(usecols = [0,6]):
    df = pd.read_csv(data_path+cell_filename+suffix,sep=',',usecols=usecols)
    num_train =int(split*len(df))
    num_test = len(df) - num_train +timestep
    print("num_train:{0},num_test:{1}".format(num_train,num_test))

    train = df[:num_train][1:]#[cycle_cap_header[1]]
    test = df[num_train-timestep:][1:]#[cycle_cap_header[1]]
    all = df[:][:]#[cycle_cap_header]
    train = np.array(train)#.reshape(-1,len(usecols)-1)
    test = np.array(test)#.reshape(-1,len(usecols)-1)
    all = np.array(all)
    print('train.shape:{0}'.format(train.shape))
    print('test.shape:{0}'.format(test.shape))
    print('all.shape:{0}'.format(all.shape))
    return train[:,1:],test[:,1:],all

def get_scaled_train(data):
    scaler = MinMaxScaler()
    data = np.array(data)
    train_scaled = scaler.fit_transform(data)
    return train_scaled,scaler

def get_x_y(data,timestep,col_index = 1, pre_step = 1):
    x,y = [],[]
    for i in range(timestep,len(data)):
        x.append(data[i-timestep:i])
        y.append(data[i:i+pre_step,col_index])
    x = np.array(x)
    y =np.array(y).reshape(-1,pre_step)
    print("x_shape:{0},y_shape:{1}".format(x.shape,y.shape))
    return x,y

def get_x_scaled_y(data,timestep,scaler,col_index = 1,pre_step = 1):
    x_scaled, y_unscaled, mu_list, std_list = [], [], [], []
    for i in range(0, len(data) - timestep - pre_step+1, pre_step):
        x_scaled.append(scaler.transform(data[i:i + timestep, :]))
        y_unscaled.append(data[i + timestep:i + timestep + pre_step, col_index])
    x_scaled = np.array(x_scaled)
    print("test_x_scaled.shape:{0}".format(x_scaled.shape))
    y_unscaled = np.array(y_unscaled).reshape(-1, pre_step)
    print("test_y.shape:{0}".format(y_unscaled.shape))
    return x_scaled,y_unscaled

def main(split,timestep,pre_step,batchsize,epochs,dropout_prob,failure):
    train,test,all = get_origin_train_test(usecols=multi_usecols)
    start_pre_cycle,eol_cycle,rul, eol_cap,eol_cap_norm = get_rul(all,split,col_index=col_index)
    train_scaled,scaler = get_scaled_train(train)
    x_train_scaled,y_train_scaled = get_x_y(train_scaled,timestep,col_index-1)
    x_test_scaled,y_test = get_x_scaled_y(test,timestep,scaler,col_index-1)
    model = lstm_network.lstm_network().build_network(timestep,feature_num,dense_units=pre_step,dropout_prob=dropout_prob)
    model.fit(x_train_scaled,y_train_scaled,batchsize,epochs)
    pre_scaled_y = model.predict(x_test_scaled)
    train_scaled[:,col_index-1:col_index] = pre_scaled_y
    pre_y = scaler.inverse_transform(train_scaled)[:,col_index-1]
    mape = loss.get_mape(y_test,pre_y)
    mse = loss.get_rmse(y_test,pre_y)
    print("mape:{0},mse:{1}".format(mape,mse))
    pre_rul = get_pre_rul(pre_y,eol_cap_norm)
    rul_error = rul -pre_rul
    info = 'RUL_actual:{0},RUL_LSTM:{1},RUL_error:{2}'.format(rul,pre_rul,rul_error)
    print(info)
    error_2_csv(eol_cap,start_pre_cycle,eol_cycle,rul,pre_rul,rul_error,failure)
    all_y = np.array(all[:,col_index]).reshape(-1,1)
    utils.plot_and_save(eol_cap_norm,cell_filename,timestep,info,'../result/sks_figure/',all_y,test,train,pre_y)

if __name__ == '__main__':
    # split = 0.5
    # timestep = 30
    # pre_step = 1
    # dense_units = pre_step
    # batchsize = 64
    # epochs = 100
    # dropout_prob = 0.5
    parser = argparse.ArgumentParser(description='LSTM RUL Prediction for sks')
    parser.add_argument('--timestep', type=int, default=30, help='time_step in lstm')
    parser.add_argument('--split', default=0.5, help='split of train and test set')
    parser.add_argument('--pre_step',default=1,type=int)
    parser.add_argument('--batch_size', type=int, default=64, help='8的效果不好')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--dropout', default=0.5)
    parser.add_argument('--failure',default=0.7,type=float)
    args = parser.parse_args()

    split = args.split
    dropout_prob = args.dropout
    timestep = args.timestep
    pre_step = args.pre_step
    batchsize = args.batch_size
    epochs = args.epochs
    failure = args.failure

    for timestep in [10,15,20,25,30,35,40]:
        for epochs in [100,50]:
            for batchsize in [32,64,126]:
                main(split,timestep,pre_step,batchsize,epochs,dropout_prob,failure)