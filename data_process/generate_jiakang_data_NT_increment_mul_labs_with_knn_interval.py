#modified successfuly:2020.12.20 17:36
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import random

from datetime import datetime

from data_process.fit_data import fit_data

from Utils.judge_index_inf import *

from params.param_KNN import FINAL_T

np.set_printoptions(threshold=np.inf)

time_step = 7


#all_data_path = "./data/result_date_5000.txt"
all_data_path = "./data/result_date_10000.txt"
#all_data_path = "./data/result_date.txt"


#all_data_path = "../data/result_date_1000.txt"

def get_data(path):
    print("get_data")
    f = open(path, "r")
    d = pd.read_csv(f, header=None).values
    f.close()
    temp1, temp2, temp3, temp4 = [], [], [], []
    x_data, y1_data, y2_data, y3_data, y4_data = [], [], [], [], []
    time_data = []
    id_data = []
    label_date= []
    r = 0
    while r < d.shape[0]:
        nr = r + 1
        while nr < d.shape[0] and d[nr, 0] == d[r, 0]:
            nr += 1
        # temp = d[r:nr, [11, 12, 3, 5, 7, 9]]
        # lrh:no age,no sex,no trab,add time
        temp = d[r:nr, [3, 5, 7]]
        temp_time = d[r:nr, [1]]
        temp_id = d[r, [0]]

        if temp.shape[0] >= 8:
            x_data.append(temp[:time_step].tolist())
            time_data.append(temp_time[:time_step].tolist())
        else:
            tempt = temp[:-1].tolist()
            tempt_time = temp_time[:-1].tolist()
            # 0 padding
            # for i in range(8 - temp.shape[0]):
            #     tempt.append([0, 0, 0, 0, 0, 0])
            x_data.append(tempt)
            time_data.append(tempt_time)
        # temp1.append(temp[-1, 2])
        # temp2.append(temp[-1, 3])
        # temp3.append(temp[-1, 4])
        # temp4.append(temp[-1, 5])
        # lrh
        temp1.append(temp[-1, 0])
        temp2.append(temp[-1, 1])
        temp3.append(temp[-1, 2])
        id_data.append(temp_id)
        label_date.append(temp_time[-1])
        # temp4.append(temp[-1, 5])
        r = nr
    # convert to 2 dimension
    for i in temp1:
        y1_data.append([i])
    for i in temp2:
        y2_data.append([i])
    for i in temp3:
        y3_data.append([i])
    # for i in temp4:
    #     y4_data.append([i])

    # x_data = np.array(x_data)
    # y1_data = np.array(y1_data)
    # y2_data = np.array(y2_data)
    # y3_data = np.array(y3_data)
    # y4_data = np.array(y4_data)
    # print("x_data",x_data.shape,x_data)
    # print("y1_data",y1_data.shape,y1_data)
    # print("y2_data", y2_data.shape, y2_data)
    # print("y3_data", y3_data.shape, y3_data)
    # print("y4_data", y4_data.shape, y4_data)

    # lrh_generate_generate_jiakang x,y,t

    batch = [[], [], [], [], [], [], []]
    batch_time = [[], [], [], [], [], [], []]
    label = [[], [], [], [], [], [], []]
    label_time =[[], [], [], [], [], [], []]
    id = [[], [], [], [], [], [], []]
    #增1：增加y_data
    y_data=[[],[],[],[],[],[],[]]

    statistic_num = np.zeros([7])

    #print("id_data.__len__()", id_data.__len__())/

    #print("x_data.__len__()", x_data.__len__())
    # x_data.shape=1960*n*3

    #print("1. add batch,add label,add id,add_label_time")
    missing_num = 0
    for i in range(x_data.__len__()):
        time = len(x_data[i]) - 1
        statistic_num[time] = statistic_num[time] + 1

        # (1) add batch
        # print("x_data[i]",x_data[i])
        batch[time].append(x_data[i])

        # (2) add label:label convert to one_hot vector,adress missing values"-1" to "1" means "normal"
        # 多指标：添加label的状态，ft3\ft4\tsh,三个指标的三种状态
        temp_label = np.zeros([3, 3])

        # ft3
        if ((ft3(y1_data[i][0])) == -1):
            temp_label[0, 1] = 1
            missing_num = missing_num + 1
        else:
            temp_label[0, ft3(y1_data[i][0])] = 1

        # ft4
        if ((ft4(y2_data[i][0])) == -1):
            temp_label[1, 1] = 1
            # missing_num = missing_num + 1
        else:
            temp_label[1, ft4(y2_data[i][0])] = 1

        # tsh
        if ((tsh(y3_data[i][0])) == -1):
            temp_label[2, 1] = 1
            # missing_num = missing_num + 1
        else:
            temp_label[2, tsh(y3_data[i][0])] = 1

        # perclass改0：tsh异常偏低变为2，普通为1，异常偏高为0，使其变化方向与ft3、ft4一致
        temp_label[2] = temp_label[2][::-1]

        # print("label_test",y1_data[i][0],ft3(y1_data[i][0]),temp_label)
        label[time].append(temp_label)

        #增2：
        y_data[time].append(np.array([y1_data[i][0],y2_data[i][0],y3_data[i][0]]))

        # (3) add id
        id[time].append(id_data[i])

        # (4) add label_time
        #print("label_date[i]",label_date[i][0])
        #all_data改2:
        date_to_time=datetime.strptime(label_date[i][0][:-4], '%Y-%m-%d %H:%M:%S')
        label_time[time].append(date_to_time)

    # addition: test label_time
    # for i in range(id.__len__()):
    #     for j in range(id[i].__len__()):
    #         print("[i,j]")
    #         print("id[i][j]",id[i][j])
    #         print("label_time[i][j]",label_time[i][j])


    #print("missing_num", missing_num)

    # convert each batch\label to np.array
    for i in range(batch.__len__()):
        batch[i] = np.array(batch[i])
        label[i] = np.array(label[i])
        #增3：
        y_data[i]=np.array(y_data[i])


    # validate batch\label
    #print("statistic_num", statistic_num)

    # for i in range(batch.__len__()):
    #     print("id[i].__len__()", id[i].__len__())
    #     print("id[i][0]", id[i][0])
    #     print("batch[i].shape", batch[i].shape)
    #     print("batch[i][0]", batch[i][0])
    #     print("label[i].shape", label[i].shape)
    #     print("label[i][0]", label[i][0])
    #     print("label_time[i].len", label_time[i].__len__())
    #     print("label_time[i][0]", label_time[i][0])

    #print("2. count and add batch_time,calculate batch_delta_time")

    #print("time_data.__len__()", time_data.__len__())
    # time_data.shape=1960*n*1

    # (1) add batch_time
    for i in range(time_data.__len__()):
        time = len(time_data[i]) - 1
        batch_time[time].append(time_data[i])

    # valid batch_time
    # print("batch_time", batch_time.__len__())
    # for i in range(batch_time.__len__()):
    #     print("batch_time_len", batch_time[i].__len__())

    # (2) convert the type of batch_time from str to datetime
    for i in range(batch_time.__len__()):
        for j in range(batch_time[i].__len__()):
            for k in range(batch_time[i][j].__len__()):
                # print("batch_time[i][j][k]",batch_time[i][j][k])
                value = batch_time[i][j][k][0]
                #convert_date = datetime.strptime(value, '%Y/%m/%d %H:%M:%S')
                #all_data改1
                convert_date = datetime.strptime(value[:-4], '%Y-%m-%d %H:%M:%S')
                batch_time[i][j][k] = convert_date
                # print("batch_time[i][j][k] after convert",batch_time[i][j][k])

    # (3) calculate batch_delta_time
    batch_delta_time = []
    for i in range(batch_time.__len__()):
        temp_batch = []
        for j in range(batch_time[i].__len__()):
            temp_patient = np.zeros([i + 1])
            #print("show_batch_time[i][j]",batch_time[i][j])
            for k in range(batch_time[i][j].__len__()):
                #last time_step
                if (k ==batch_time[i][j].__len__()-1):
                    delta_time= label_time[i][j]-batch_time[i][j][k]
                else:
                    # print(batch_time[i][j][k],batch_time[i][j][k+1])
                    delta_time = batch_time[i][j][k+1] - batch_time[i][j][k]
                # print (delta_time.days)
                # 1.days'unit
                temp_patient[k]=delta_time.days
                # 2.hours'unit
                # temp_patient[k] = delta_time.days * 24 + delta_time.seconds / 3600
            temp_batch.append(temp_patient)
        temp_batch = np.array(temp_batch)
        batch_delta_time.append(temp_batch)

    # valid batch_delta_time
    # for i in range(batch_delta_time.__len__()):
    #     print("batch_delta_time.shape", batch_delta_time[i].shape)
    # print(batch_delta_time)

    #print("3. remove patient's data_process whose time_num<3 ")
    id = id[2:]
    batch = batch[2:]
    batch_delta_time = batch_delta_time[2:]
    label = label[2:]
    label_time=label_time[2:]
    #增4：
    y_data=y_data[2:]
    # valid
    # for i in range(batch.__len__()):
    #     print("id[i].__len__()", id[i].__len__())
    #     print("batch[i].__len__()", batch[i].__len__())
    #     print("batch_delta_time[i].__len__()", batch_delta_time[i].__len__())
    #     print("label[i].__len__()", label[i].__len__())
    #     print("label_time[i].__len__()", label_time[i].__len__())

    # reshape batch_delta_time[i] to batch_size*1*time_num
    for i in range(batch_delta_time.__len__()):
        batch_delta_time[i] = np.reshape(batch_delta_time[i],
                                         [batch_delta_time[i].shape[0], 1, batch_delta_time[i].shape[1]])
        #print("batch_delta_time[i].shape", batch_delta_time[i].shape)

    #print("4.content validate")
    # validated completely

    # (1) train_set_validation
    # each batch,validate continus three samples
    # random validate 5 samples
    # random validate 3 abnormal samples' label
    # random validate abnormal\normal 10 examples selected from train_data.txt firstly

    # (2) test_set_validation
    # random validate 5 samples
    # random validate abnormal\normal 5 examples selected from test_data.txt firstly

    # for i in range(batch.__len__()):
    #    for j in range(batch[i].__len__()):
    #         print("index",i,j)
    #         print("id[i][j]", id[i][j])
    #         print("batch[i][j]",batch[i][j])
    #         print("batch_delta_time[i][j]",batch_delta_time[i][j])
    #         print("label[i][j]",label[i][j])
    #         print("label_time[i][j]",label_time[i][j])

    # print("x_data", x_data)
    # print("y1_data", y1_data)
    # print("y2_data", y2_data)
    # print("y3_data", y3_data)
    # print("y4_data", y4_data)

    # 测试输出
    # print("test_output")
    # for i in range(len(id)):
    #     print("id", i, id[i][0])
    #     print("batch", i, batch[i][0])
    #     print("batch_delta_time", i, batch_delta_time[i][0])
    #     print("label", i, label[i][0])
    #     #增5
    #     print("y_data", i, y_data[i][0])
        

    #y_data检查正确：7.15 22:55
    #all_data改:增加 id
    return id,batch, batch_delta_time, label, y_data

#生成 x_coo,y_coo,final_t
#修改了x_coo和_y_coo，增加了被预测的时刻点
def generate_curve_inf_with_y(id, batch, batch_delta_time, label, y_data):

    #1.保存每个病人所有batch的x坐标
    x_coo=[]
    for i in range(len(batch_delta_time)):
        batch_patinet_x_coo=[]
        for j in range(len(batch_delta_time[i])):
            patient_x_coo=[]
            patient_x_coo.append(0)
            #-1保证不计算final_delta_t
            #print("batch_delta_time[i][j]",batch_delta_time[i][j])

            #横坐标不计算final_t
            #for k in range(len(batch_delta_time[i][j][0])-1):
            #改1：横坐标计算final_t
            for k in range(len(batch_delta_time[i][j][0])):
                #print("patient_x_coo[-1]",patient_x_coo[-1])
                #print("batch_delta_time[i][j][0][k]",batch_delta_time[i][j][0][k])
                patient_x_coo.append(patient_x_coo[-1]+batch_delta_time[i][j][0][k])
            patient_x_coo=np.array(patient_x_coo)
            #print("patient_x_coo",patient_x_coo)

            #把天数映射为月份,2位小数
            patient_x_coo=np.around(patient_x_coo/30.0,2)

            batch_patinet_x_coo.append(patient_x_coo)
        batch_patinet_x_coo=np.array(batch_patinet_x_coo)
        x_coo.append(batch_patinet_x_coo)
    x_coo=np.array(x_coo)

    #2.保存每个病人所有batch的y坐标
    y_coo = []
    for i in range(len(batch)):
        batch_patinet_y_coo = []
        for j in range(len(batch[i])):
            patient_y_coo = []
            for k in range(len(batch[i][j])):
                #0代表ft3
                patient_y_coo.append(batch[i][j][k][0])
            #每个病人最后一个时刻点，添加y_data
            #print("id",id[i][j])
            #改2
            patient_y_coo.append(y_data[i][j][0])
            #print("patient_y_coo",patient_y_coo)

            patient_y_coo = np.array(patient_y_coo)
            batch_patinet_y_coo.append(patient_y_coo)
        batch_patinet_y_coo = np.array(batch_patinet_y_coo)
        y_coo.append(batch_patinet_y_coo)
    y_coo = np.array(y_coo)

    #3.保存每个病人的final_t
    final_t = []
    for i in range(len(batch_delta_time)):
        patient_final_t = []
        for j in range(len(batch_delta_time[i])):
            #改3：现在，x_coo[i][j][-1]即为final_t
            #patient_final_t.append(np.around(batch_delta_time[i][j][0][-1]/30.0,2)+x_coo[i][j][-1])
            patient_final_t.append(x_coo[i][j][-1])
        patient_final_t = np.array(patient_final_t)
        final_t.append(patient_final_t)
    final_t = np.array(final_t)

    # 4、生成前六个月对应曲线的参数,以及在曲线上采样的点
    curve_pars = []
    curve_points = []
    for i in range(len(x_coo)):
        batch_curve_pars = []
        batch_curve_points = []
        for j in range(len(x_coo[i])):
            patient_curve_pars,patient_curve_points = fit_data(x_coo[i][j], y_coo[i][j])
            batch_curve_pars.append(patient_curve_pars)
            batch_curve_points.append(patient_curve_points)
        batch_curve_pars = np.array(batch_curve_pars)
        batch_curve_points=np.array(batch_curve_points)
        curve_pars.append(batch_curve_pars)
        curve_points.append(batch_curve_points)
    curve_pars = np.array(curve_pars)
    curve_points=np.array(curve_points)


    #5.test
    print("test_generate_curve_inf_output")
    for i in range(len(x_coo)):
        #前15个
        print("id", i, id[i][:15])
        print("batch_delta_time", i, batch_delta_time[i][:15])
        print("final_t", i, final_t[i][:15])
        print("x_coo",i,x_coo[i][:15])

        print("batch", i, batch[i][:15])
        print("y_coo", i, y_coo[i][:15])

        #后15个
        print("id", i, id[i][-15:])
        print("batch_delta_time", i, batch_delta_time[i][-15:])
        print("final_t", i, final_t[i][-15:])
        print("x_coo",i,x_coo[i][-15:])

        print("batch", i, batch[i][-15:])
        print("y_coo", i, y_coo[i][-15:])

    return x_coo,y_coo,final_t,curve_pars,curve_points

#生成 x_coo,y_coo,final_t
def generate_curve_inf(id, batch, batch_delta_time, label, y_data):

    #1.保存每个病人所有batch的x坐标
    x_coo=[]
    for i in range(len(batch_delta_time)):
        batch_patinet_x_coo=[]
        for j in range(len(batch_delta_time[i])):
            patient_x_coo=[]
            patient_x_coo.append(0)
            #-1保证不计算final_delta_t
            #print("batch_delta_time[i][j]",batch_delta_time[i][j])

            #横坐标不计算final_t
            for k in range(len(batch_delta_time[i][j][0])-1):
                #print("patient_x_coo[-1]",patient_x_coo[-1])
                #print("batch_delta_time[i][j][0][k]",batch_delta_time[i][j][0][k])
                patient_x_coo.append(patient_x_coo[-1]+batch_delta_time[i][j][0][k])
            patient_x_coo=np.array(patient_x_coo)
            #print("patient_x_coo",patient_x_coo)

            #把天数映射为月份,2位小数
            patient_x_coo=np.around(patient_x_coo/30.0,2)

            batch_patinet_x_coo.append(patient_x_coo)
        batch_patinet_x_coo=np.array(batch_patinet_x_coo)
        x_coo.append(batch_patinet_x_coo)
    x_coo=np.array(x_coo)

    #2.保存每个病人所有batch的y坐标
    y_coo = []
    for i in range(len(batch)):
        batch_patinet_y_coo = []
        for j in range(len(batch[i])):
            patient_y_coo = []
            for k in range(len(batch[i][j])):
                #0代表ft3
                patient_y_coo.append(batch[i][j][k][0])

            patient_y_coo = np.array(patient_y_coo)
            batch_patinet_y_coo.append(patient_y_coo)
        batch_patinet_y_coo = np.array(batch_patinet_y_coo)
        y_coo.append(batch_patinet_y_coo)
    y_coo = np.array(y_coo)

    #3.保存每个病人的final_t
    final_t = []
    for i in range(len(batch_delta_time)):
        patient_final_t = []
        for j in range(len(batch_delta_time[i])):
            patient_final_t.append(np.around(batch_delta_time[i][j][0][-1]/30.0,2)+x_coo[i][j][-1])
        patient_final_t = np.array(patient_final_t)
        final_t.append(patient_final_t)
    final_t = np.array(final_t)

    # 4、生成前六个月对应曲线的参数,以及在曲线上采样的点
    curve_pars = []
    curve_points = []
    for i in range(len(x_coo)):
        batch_curve_pars = []
        batch_curve_points = []
        for j in range(len(x_coo[i])):
            patient_curve_pars, patient_curve_points = fit_data(x_coo[i][j], y_coo[i][j],id[i][j])
            batch_curve_pars.append(patient_curve_pars)
            batch_curve_points.append(patient_curve_points)
        batch_curve_pars = np.array(batch_curve_pars)
        batch_curve_points = np.array(batch_curve_points)
        curve_pars.append(batch_curve_pars)
        curve_points.append(batch_curve_points)
    curve_pars = np.array(curve_pars)
    curve_points = np.array(curve_points)


    #5.test
    print("test_generate_curve_inf_output")
    for i in range(len(x_coo)):
        #前15个
        print("id", i, id[i][:15])
        print("batch_delta_time", i, batch_delta_time[i][:15])
        print("final_t", i, final_t[i][:15])
        print("x_coo",i,x_coo[i][:15])

        print("batch", i, batch[i][:15])
        print("y_coo", i, y_coo[i][:15])

        #后15个
        print("id", i, id[i][-15:])
        print("batch_delta_time", i, batch_delta_time[i][-15:])
        print("final_t", i, final_t[i][-15:])
        print("x_coo",i,x_coo[i][-15:])

        print("batch", i, batch[i][-15:])
        print("y_coo", i, y_coo[i][-15:])

    return x_coo,y_coo,final_t,curve_pars,curve_points


#将所有病人的所有数据，合并为 patient_num个字典的形式
#0、原本病人样本的不同数据使用多个batch形式的list保存
#1、将病人样本的多个不同数据，保存在同一个字典中
#2、将不同batch合并，方便在拥有不同时间步数目的病人中搜寻KNN
def generate_sample_all(id_all, batch_all, batch_delta_time_all, label_all, y_data, x_coo, y_coo, final_t, curve_pars, sta_curve_pars,curve_points):
    sample_all=[]
    for i in range(len(id_all)):
        for j in range(len(id_all[i])):
            #只取final_t前后各1个月的样本
            #if(final_t[i][j]>11 and final_t[i][j]<13):
            if (final_t[i][j] > FINAL_T-1 and final_t[i][j] < FINAL_T+1):
                temp_sample = {'id': id_all[i][j], 'x_data': batch_all[i][j], 'delta_time': batch_delta_time_all[i][j],
                                'y_label':label_all[i][j],'y_value': y_data[i][j],'x_coo':x_coo[i][j],'y_coo':y_coo[i][j],
                               'final_t':final_t[i][j],'pars':curve_pars[i][j],'sta_pars':sta_curve_pars[i][j],'curve_points':curve_points[i][j]
                               }
                #print("temp_sample",temp_sample)
                sample_all.append(temp_sample)
    print("len(sample_all)",len(sample_all))
    return sample_all

#绘图验证，为病人寻找到的Knn病人，由它们数据拟合的曲线，是否正确且直观合理
def validate_knn_patient(patient,knn_patient,k):
    #一、绘制源病人
    #1、以散点图形式绘制原始数据
    plt.scatter(patient['x_coo'], patient['y_coo'], c='black',s=100,marker='s')
    plt.scatter(patient['final_t'], patient['y_value'][0], c='black',s=100,marker='s')

    print("validate_knn_patient")
    print("",patient['id'])
    print("",patient['x_coo'])
    print("",patient['y_coo'])
    print("",patient['final_t'])
    print("",patient['y_value'][0])

    #plt.show()
    #2、以曲线图绘制拟合曲线
    # 根据参数，生成函数
    p1 = np.poly1d(patient['pars'])
    #只绘制到该病人前六个月最后一个时刻点
    #o_x_all=np.arange(patient['x_coo'][0], patient['x_coo'][-1] + 0.1, 0.1)
    #根据曲线，绘制前六个月的全部数据（此处间隔0.1,没有什么特殊含义，就是为了让点足够多，不至于把曲线图绘制为折线图）
    o_x_all = np.arange(0, 6, 0.1)

    # 只绘制到patient的最后一个真值点
    # index_patient = len(patient['x_data']) - 1
    # bound_patient = patient['x_coo'][index_patient]
    # o_x_all = np.arange(0,bound_patient , 0.1)

    o_y_all=p1(o_x_all)
    plt.plot(o_x_all, o_y_all, c='black')



    #二、绘制knn病人的数据
    color=['cyan','blue','red','yellow','green']
    for i in range(k):
        temp_patient=knn_patient[i][0]
        temp_distance=knn_patient[i][1]
        #print("temp_patient",temp_patient)

        #1、绘制knn病人的拟合曲线图
        p = np.poly1d(temp_patient['pars'])
        #x_all = np.arange(temp_patient['x_coo'][0],temp_patient['x_coo'][-1] + 0.1, 0.1)
        x_all = np.arange(0, 6, 0.1)
        #只绘制到patient的最后一个真值点
        #x_all = np.arange(0, bound_patient, 0.1)
        y_all = p(x_all)
        plt.plot(x_all, y_all, c=color[i])

        #2、绘制knn病人的原始散点图，观察knn病人都有几个就诊点
        plt.scatter(temp_patient['final_t'],temp_patient['y_value'][0],c=color[i])
        temp_id=temp_patient['id']
        temp_x_coo=temp_patient['x_coo']
        temp_y_coo=temp_patient['y_coo']

        print("temp_id",temp_id)
        print("temp_x_coo",temp_x_coo)
        print("temp_y_coo", temp_y_coo)
        print("temp_distance",temp_distance)
        plt.scatter(temp_x_coo, temp_y_coo, c=color[i],s=50)


    plt.legend(loc=4)  # 指定legend的位置右下角
    plt.title(patient['id'])
    plt.show()


#利用knn病人及源病人的信息，为源病人生成区间
def generate_knn_interval(patient, knn_patient, index):
    all_y_values=[]
    all_y_values.append(patient['y_value'][index])
    for i in range(len(knn_patient)):
        all_y_values.append(knn_patient[i][0]['y_value'][index])
    all_y_values=np.array(all_y_values)
    # print("patient",patient)
    # print("all_y_values",all_y_values)
    y_u=np.max(all_y_values)
    y_l=np.min(all_y_values)
    patient['y_u']=y_u
    patient['y_l']=y_l

    return patient

#为每个病人样本字典，添加为其寻找的knn区间
#1、在每个病人间计算distance，选择与每个病人distance最短的k个病人，作为其knn病人，添加到其字典中
#2、绘图验证
#3、生成knn区间并添加到病人的字典中

def add_knn_interval(sample_all, k):
    #记录每个batch进行validate_knn_patient的数目
    validate_plot_num=[0,0,0,0,0]
    patient_nums = len(sample_all)
    i = 0
    while i < patient_nums:
        #print(i)
        # 保存第i个病人与其它病人的距离
        single_patient_distance = []
        j = 0
        while j < patient_nums:
            #根据曲线参数，计算距离
            #temp_distance = cal_curve_distance(sample_all[i], sample_all[j])
            #根据曲线采样点，计算距离
            #temp_distance = cal_points_distance(sample_all[i], sample_all[j])
            #只比较sample_all[i]最后一个真值之前的曲线段上的采样点
            temp_distance = cal_points_distance_only_src(sample_all[i], sample_all[j])
            # 将第j个病人的信息，以及和第j个病人的distance，都保存下来
            single_patient_distance.append([sample_all[j], temp_distance])
            j = j + 1
        # 把距离最短的前k个病人，保存到当前病人的信息中
        shortest_k_patient = sorted(single_patient_distance, key=lambda x: x[1])[:k]
        #print("shortest_k_patient",shortest_k_patient)
        sample_all[i]['shortest_k_patient'] = shortest_k_patient

        #validate_plot
        # patiment_time_step=len(sample_all[i]['x_data'])
        # validate_plot_index=patiment_time_step-3
        # if(validate_plot_num[validate_plot_index]<10):
        #     #绘图检验
        #     validate_knn_patient(sample_all[i], shortest_k_patient, k)
        #     #该batch的绘制数目+1
        #     validate_plot_num[validate_plot_index]=validate_plot_num[validate_plot_index]+1
        #/validate_plot

        #knn区间生成。21.7.18，19：13
        #index=0,代表ft3
        sample_all[i]=generate_knn_interval(sample_all[i], shortest_k_patient, 0)
        #print("current:",sample_all[i]['id'],sample_all[i]['pars'])

        i = i + 1

    return sample_all


#标准化所有病人，一个指标的数据
def standard(data,way):
    if(way=='Z'):
        avg=np.average(data)
        std=np.std(data)
        print("avg",avg)
        print("std",std)
        data=(data-avg)/std
        return data
    if(way=='M'):
        max=np.max(data)
        min=np.min(data)
        data=(data-min)/(max-min)
        return data


#对全部曲线参数进行标准化,way='Z'或'M'
def standard_curve_pars(curve_pars,way):
    #1.将batch合并，便于标准化
    pars_all = []
    #记录每个batch的数目，便于还原
    batch_num=[]
    batch_num.append(0)
    for i in range(len(curve_pars)):
        #print("len(curve_pars[i])",i,len(curve_pars[i]))
        batch_num.append(batch_num[-1]+len(curve_pars[i]))
        for j in range(len(curve_pars[i])):
            pars_all.append(curve_pars[i][j])
    pars_all = np.array(pars_all)
    #print("pars_all", pars_all.shape, pars_all[:5])


    par_num=len(pars_all[0,:])

    #2.标准化
    for i in range(par_num):
        pars_all[:,i]=standard(pars_all[:,i],way)

    #3.还原为batch形式
    sta_pars_all=[]
    for i in range(len(batch_num)-1):
        temp_sta_pars=pars_all[batch_num[i]:batch_num[i+1]]
        #print("len(temp_sta_pars)",len(temp_sta_pars))
        sta_pars_all.append(temp_sta_pars)

    return sta_pars_all


#计算两个病人之间的距离
def cal_curve_distance(a,b):
    final_t_a=a['final_t']
    final_t_b=b['final_t']

    #final_t不在一个月内
    if(abs(final_t_a-final_t_b)>1):
        # print("final_t")
        # print(a['id'])
        # print(b['id'])
        return np.inf

    #用标准化后的参数计算距离
    pars_a=a['sta_pars']
    pars_b=b['sta_pars']

    distance=cal_distance(pars_a,pars_b)

    #代表是同一个病人的数据
    if(distance==0):
        # print("distance")
        # print(a['id'])
        # print(b['id'])
        return np.inf

    return distance

# 根据曲线上的采样点
def cal_points_distance(a,b):

    final_t_a = a['final_t']
    final_t_b = b['final_t']

    # final_t不在一个月内
    if (abs(final_t_a - final_t_b) > 1):
        # print("final_t")
        # print(a['id'])
        # print(b['id'])
        return np.inf

    # 用曲线上的采样点计算欧式距离

    points_a=a['curve_points']
    points_b=b['curve_points']


    distance = cal_distance(points_a,points_b)


    # 代表是同一个病人的数据
    # （为病人生成字典时，已经限制了final_t仅在11~13之间。为病人生成的多个样本，基本只有一个样本会保存到字典中）
    # 可以打印检验一下。
    if (distance == 0):
        # print("distance")
        # print(a['id'])
        # print(b['id'])
        return np.inf

    # print("points_a", points_a)
    # print("points_b", points_b)
    # print("distance",distance)

    return distance

# 只比较src最后一个真值之前的曲线段上的采样点
def cal_points_distance_only_src(src,tar):

    final_t_src = src['final_t']
    final_t_tar = tar['final_t']

    # final_t不在一个月内
    if (abs(final_t_src - final_t_tar) > 1):
        # print("final_t")
        # print(a['id'])
        # print(b['id'])
        return np.inf

    # 用曲线上的采样点计算欧式距离

    points_src=src['curve_points']
    points_tar=tar['curve_points']


    #index_common由src决定。
    index_common=len(src['x_data'])-1
    #在bound_index之后，src没有真实数据了
    bound_index= src['x_coo'][index_common]
    bound_index_points=int(bound_index*10)

    points_src_new=points_src[:bound_index_points]
    points_tar_new=points_tar[:bound_index_points]

    # print("points_src_new",points_src_new)
    # print("points_tar_new",points_tar_new)

    # # 只保留bound_a之前的points_a
    # points_src_new=list(filter(lambda x: x <= bound_src, points_src))
    # points_tar_new = list(filter(lambda x: x <= bound_tar, points_tar))

    distance = cal_distance(points_src_new,points_tar_new)


    # 代表是同一个病人的数据
    # （为病人生成字典时，已经限制了final_t仅在11~13之间。为病人生成的多个样本，基本只有一个样本会保存到字典中）
    # 可以打印检验一下。
    if (distance == 0):
        # print("distance")
        # print(a['id'])
        # print(b['id'])
        return np.inf

    # print("points_a", points_a)
    # print("points_b", points_b)
    # print("distance",distance)

    return distance


#计算欧氏距离
def cal_distance(a,b):
    distance=0
    for i in range(len(a)):
        distance=distance+np.square(a[i]-b[i])
        #print(distance)

    distance=np.sqrt(distance)
    return distance



#展示一个指标数据的分布情况
def show_distribute(data,i):

    plt.hist(data, bins=400,normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    # # 显示横轴标签
    # plt.xlabel("interval")
    # # 显示纵轴标签
    # plt.ylabel("frequency")
    # 显示图标题
    plt.title("hist" +str(i)+ str(len(data)))
    plt.show()



def get_all_data():
    return get_data(all_data_path)


#为全部数据，生成knn区间
def get_data_with_KNN_Interval():
    #1.读取基本数据
    id_all,batch_all, batch_delta_time_all, label_all,y_data = get_all_data()

    #2.根据基本数据，生成拟合曲线相关的数据
    #拟合曲线用y_data
    #x_coo,y_coo,final_t,curve_pars,curve_points=generate_curve_inf_with_y(id_all, batch_all, batch_delta_time_all, label_all, y_data)
    #不用y_data拟合曲线
    x_coo,y_coo,final_t,curve_pars,curve_points=generate_curve_inf(id_all, batch_all, batch_delta_time_all, label_all, y_data)


    #3.对全部曲线参数进行标准化
    sta_curve_pars=standard_curve_pars(curve_pars, 'Z')


    #4.将所有病人的所有数据，合并为 patient_num个字典的形式
    sample_all=generate_sample_all(id_all, batch_all, batch_delta_time_all, label_all, y_data, x_coo, y_coo, final_t, curve_pars, sta_curve_pars,curve_points)


    #5.为每个病人样本字典，添加为其寻找的knn区间
    #程序逻辑验证正确，是否符合疾病规律待验证：7.18，19：17
    sample_all_with_knn_interval=add_knn_interval(sample_all, 3)

    #***无法打印sample_all_with_knn_interval[i]，有问题
    # for i in range(len(sample_all_with_knn_interval)):
    #     print(i)
    #     print(sample_all_with_knn_interval[i])

    #6、生成训练集、测试集的batch_dict,validated,8.3
    train_batch_dict,test_batch_dict=get_train_test_batch_dict(sample_all_with_knn_interval)

    return train_batch_dict,test_batch_dict


#validated:8.3
#将sample_all_with_knn_interval中的每一类数据解析出来，以batch_dict的形式返回
def recovery_batch(sample_all_with_knn_interval):
    first_sample=sample_all_with_knn_interval[0]


    #字典的key是数据类型名，字典的值是对应的多个batch数据
    batch_dict = {}

    #使每个key对应一个空list
    for key in first_sample.keys():
        batch_dict[key]=[[],[],[],[],[]]


    #sample_all convert to batch_dict
    for temp_sample in sample_all_with_knn_interval:
        #该样本对应的batch_index
        temp_batch_index = len(temp_sample['x_data'])-1-2
        for temp_key in temp_sample.keys():
            #关键
            batch_dict[temp_key][temp_batch_index].append(temp_sample[temp_key])


    return batch_dict

#validated,8.3
def get_train_test_batch_dict(sample_all_with_knn_interval):
    #1、分割训练集、测试集
    random.shuffle(sample_all_with_knn_interval)

    sample_num=len(sample_all_with_knn_interval)
    train_num=int(0.7*sample_num)

    train_sample=sample_all_with_knn_interval[:train_num]
    test_sample=sample_all_with_knn_interval[train_num:]

    #2、将数据转化为batch_dict的形式
    train_sample_batch_dict=recovery_batch(train_sample)
    test_sample_batch_dict=recovery_batch(test_sample)

    return train_sample_batch_dict,test_sample_batch_dict






if __name__ == "__main__":
    # isTrain=input("Train?[y/n]:")
    # if isTrain=='y':
    # train()
    # isTest=input("Test?[y/n]:")
    # if isTest=='y':
    #test()
    get_data_with_KNN_Interval()