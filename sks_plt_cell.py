import matplotlib.pyplot as plt

import load_data


def plot_from_csv(file_path, filename,suffix = '.csv',usecols=[0,6]):
    data = load_data.load_data(file_path+filename+suffix,usecols=usecols).data_all
    plt.plot(data[:,0],data[:,1])
    plt.title(filename)
    # plt.show()
    plt.savefig(file_path+filename)
    plt.close()

if __name__ == '__main__':
    file_path = 'data2/'
    nums = ['005','013','017','019','022','023','043','046']
    filename = "Statistics_1-"
    suffix = '.csv'
    for num in nums:
        plot_from_csv(file_path,filename+num,suffix)
