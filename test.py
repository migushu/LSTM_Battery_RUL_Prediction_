import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import MinMaxScaler
import random

# a = np.arange(9).reshape((3,3))
# print(a)
# a = np.reshape(a,(-1,1))
# print(a)
# print(a.shape)
# np.random.shuffle(a)#np.random 和random有区别。random结果不是想要的
# print(a)



#
# a = np.zeros((8,))
# print(a)


# a=[[[1, 2, 3, 0],[4, 5, 6, 0]],[[1,1,1,1],[2,2,2,2]]]
# print(np.mean(a))
# print(np.std(a))

with open("sks/single_input.sh",'w') as f :
    for i in range(5,100,5):
        for epochs in [50,100,150,200]:
            for batchsize in [8,16,32,64,128]:
                sh_str = "python ./sks/sks_run.py --timestep " + str(i) + " --epochs " + str(
                    epochs) +' --batch_size ' + str(batchsize)

                f.write(sh_str)
                f.write("\n")

                f.flush()

#


# for i in range(1,5):
#     print(i)

# x=np.array([[1,2,0],[0,2,0]]).astype(float)
# scaler = MinMaxScaler()
# x_scaler = scaler.fit_transform(x)
# print(x_scaler)
# x_inv=scaler.inverse_transform(x_scaler)
# print(x_inv)
# print(x)

# print(keras.__version__)
# df = pd.read_csv('B0005_cycle_capacity.csv', sep=',')
# print(df.head(5))