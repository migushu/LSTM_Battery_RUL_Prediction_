import argparse
import os.path as osp
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import load_data
import loss
import lstm_model
import lstm_network
import timestep_prediction
from utils import plot_and_save, get_time


def get_base_train_data(dataloader,timestep,pre_step,is_timestep_y = False):
    cells_num = 10
    scale = MinMaxScaler()
    data = np.concatenate((dataloader.data_all[:,0:1],scale.fit_transform(dataloader.data_all[:,1:])),axis=1)
    cells = []
    for i in range(cells_num):
        cells.append([row for row in data if int(row[0]) == i])
    print(len(cells))

    train_x = np.zeros((1,timestep,1))
    train_y = np.zeros((1,pre_step)) if is_timestep_y else np.zeros((1,1))
    for cell in cells:
        cell = np.array(cell)
        cell = cell[:,1:]
        # print(cell.shape)
        if is_timestep_y:
            cell_train_x,cell_train_y = timestep_prediction.get_supervised_timestep_x_y(cell,timestep,pre_step)
        else:
            cell_train_x,cell_train_y = dataloader.get_train_x_y(cell,scale=0,is_shuffle=True)
        train_x = np.concatenate((train_x,cell_train_x),axis=0)
        train_y = np.concatenate((train_y,cell_train_y),axis=0)
    print(train_x.shape)
    print(train_y.shape)

    #打乱训练集顺序，训练的loss降低了很多
    train_x = np.array(train_x[1:,:]).reshape(-1,timestep)
    train_y = train_y[1:,:]
    train = np.concatenate((train_x,train_y),axis=1)
    # random.shuffle(train)#DONE:有问题
    np.random.shuffle(train)
    train_x = np.array(train[:,:timestep]).reshape(-1,timestep,1)
    train_y = np.array(train[:,timestep:])

    return train_x,train_y,scale

def train_base_model(seq_len,usecols,batch_size,epochs,split = 1):
    filename = "multi_battery/cells_multi_input.csv"
    #DONE:此处有bug，多电池数据按timestep分成样本时会把不同电池的数据分到一个样本
    dataloader = load_data.load_data(filename,seq_len,split,usecols)
    train_x,train_y,scale = get_base_train_data(dataloader,seq_len)#TODO:没有效果
    # train_x,train_y = dataloader.get_train_x_y(dataloader.data_all)
    # train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], len(usecols)-1))
    #
    model = lstm_network.lstm_network().build_network(seq_len,len(usecols)-1)
    lstm = lstm_model.lstm()
    model = lstm.train_model(model,train_x,train_y,batch_size,epochs)
    save_path="saved_model/base_seqlen{0}_batchsize{1}_epoch{2}_features{3}" \
              "".format(seq_len,batch_size,epochs,len(usecols)-1)
    scale_x,scale_y = dataloader.get_scaler_x_y()#TODO：scale有问题，不需要y
    lstm.save_model(model,save_path,scale_x,scale_y)

def main():
    #TODO 调试多特征值输入

    parser = argparse.ArgumentParser(description='LSTM RUL Prediction')
    parser.add_argument('--filename', type=str, default="data/2017_06_30_cell8_data.csv")
    parser.add_argument('--output_path',type=str,default="snapshot/single_variable")
    parser.add_argument('--predict_measure', type=int, default=0, choices=[0,1])
    parser.add_argument('--sequence_length', type=int,default=20,
                        help='time_step in lstm')
    parser.add_argument('--split', default=0.6, help='split of train and test set')
    parser.add_argument('--batch_size', type=int, default= 8,
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default= 50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--dropout', default= 0.5)
    parser.add_argument('--saved_figure_path',default='result/single_variable/')
    parser.add_argument('--feature_num',type=int,default= 1,
                        help='single feature use 1,multi use 7')
    parser.add_argument('--usecols',default= [9, 10],type=int,nargs='+',
                        help='single feature imput use [9,10], multi use [3, 4, 5, 6, 7, 8, 9, 10]')
    #必须设置type=int，否则脚本执行时会导致get_model返回None
    parser.add_argument('--get_model_measure',default= 1,type=int,
                        help='0 for define model from begin, 1 for load exist model')
    #cell 个数越小结果越精确， 数据太少？
    parser.add_argument('--lstm_units',default=50,type=int)

    args = parser.parse_args()
    split = args.split
    dropout_prob = args.dropout
    sequence_length = args.sequence_length
    batch_size = args.batch_size
    epochs = args.epochs
    predict_measure = args.predict_measure  # 0 for predicting one cycle,1 for predicting len(test(y)) cycles continuely, use current predicted value as the next input.
    filename = args.filename
    save_filepath = args.saved_figure_path
    feature_num = args.feature_num #用于训练的每个数据样本的特征数
    usecols=args.usecols #要读取的数据文件的列
    get_model_measure = args.get_model_measure
    lstm_units = args.lstm_units

    loss_file_path = args.output_path
    loss_file_name = 'seqlen:{0}_mse_mape_{1}.txt'.format(str(sequence_length),str(get_time()))

    fo = open(osp.join(loss_file_path,loss_file_name),'w')

    fo.write(str('N,batch_size,epochs,mse,mape\n'))
    fo.flush()

    batch_size_list = [8,16,32,64,128]
    epochs_list = [50,75,100,150,200]

    dataloader = load_data.load_data(filename, sequence_length, split, usecols=usecols)
    sca_x, sca_y=dataloader.load_scaler()
    train_x, train_y, test_x, test_y = dataloader.get_x_y()
    all_y = dataloader.get_all_y()

    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], feature_num))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], feature_num))
    print(train_y.shape)

    lstm = lstm_model.lstm()
    model = get_model(lstm, get_model_measure,sequence_length,feature_num,dropout_prob,lstm_units=lstm_units)
    lstm.train_model(model,train_x,train_y,batch_size,epochs)
    predict_y = lstm.predict(model,test_x,pre_way=predict_measure)

    # sca_x, sca_y = dataloader.get_scaler_x_y()
    train_y = sca_y.inverse_transform(train_y)
    test_y = sca_y.inverse_transform(test_y)
    predict_y = sca_y.inverse_transform(predict_y)

    mse = loss.get_rmse(test_y, predict_y)
    mape = loss.get_mape(test_y, predict_y)

    err_str = '{0},{1},{2},{3},{4}\n'.format(sequence_length, batch_size, epochs, mse, mape)
    fo.write(str(err_str))
    fo.flush()

    plotfilename = 'seqLen:{0}_batchsize:{1}_epochs:{2}_preMeasure:{3}_dropout:{4}'.format(sequence_length, batch_size,
                                                                                           epochs, predict_measure,
                                                                                           dropout_prob)
    title = plotfilename + '\nmse:{0}_mape:{1}'.format(mse, mape)
    plot_and_save(title,sequence_length,plotfilename,save_filepath,all_y,test_y,train_y,predict_y)
    fo.close()

#当自定义模型时，需输入后边的三个变量
def get_model(lstm,get_model_measure,sequence_length=0,feature_num=0,dropout_prob=0,lstm_units = 50):
    if get_model_measure == 0: #define model by self.
        return lstm.build_model(sequence_length,feature_num,dropout_prob,lstm_units = lstm_units)
    elif get_model_measure ==1:#load model from file
        json_filepath = 'saved_model/base_seqlen20_batchsize128_epoch100_features1.json'
        model_weight_filepath= 'saved_model/base_seqlen20_batchsize128_epoch100_features1.h5'
        return lstm.load_model(json_filepath,model_weight_filepath)
    else:
        return None

if __name__ == '__main__':
    # scalerx = joblib.load("saved_model/base_seqlen20_batchsize128_epoch1_features1x.scale")
    # train_base_model(20,[0,7],128,epochs=50)
    main()