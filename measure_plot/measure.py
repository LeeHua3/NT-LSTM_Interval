import numpy as np
from sklearn.metrics import mean_squared_error,classification_report
import matplotlib.pyplot as plt
from Utils.judge_index_inf import ft3
from measure_plot.plot import my_plot

#如果要输出错误信息，必须先输出函数名，便于定位
#必须全部都有返回值

#validated
def cal_PICP_MPIW(y_true, y_pred):
    print("cal_PICP_MPIW")
    # for i in range(y_true.shape[0]):
    #     print("u l t",y_pred[i][0],y_pred[i][1],y_true[i])
    y_true_point=y_true[:,2]

    y_u_pred = y_pred[:,0]
    y_l_pred = y_pred[:,1]

    K_u = y_u_pred > y_true_point
    K_l = y_l_pred < y_true_point


    PICP = np.mean(K_u * K_l)
    MPIW=np.round(np.mean(y_u_pred - y_l_pred),3)
    # print('PICP:', np.mean(K_u * K_l))
    # print('MPIW:', np.round(np.mean(y_u_pred - y_l_pred),3))

    return PICP,MPIW


#validated
#计算对异常偏高病人的PICP和MPIW
def cal_ah_PICP_MPIW(y_true, y_pred):
    print("cal_PICP_MPIW")
    # for i in range(y_true.shape[0]):
    #     print("u l t",y_pred[i][0],y_pred[i][1],y_true[i])
    y_true_point=y_true[:,2]

    y_u_pred = y_pred[:,0]
    y_l_pred = y_pred[:,1]

    K_u = y_u_pred > y_true_point
    K_l = y_l_pred < y_true_point

    # lrh改：只判断异常偏高的PICP
    #真值异常偏高为1，否则为0
    K_high=np.zeros_like(K_l)
    #有多少病人，真值为异常偏高
    abnormal_high_true_num=0
    for i in range(len(K_high)):
        if(ft3(y_true_point[i])==2):
            print("y_true_point[i]",y_true_point[i])
            K_high[i]=1
            print("K_high[i]",K_high[i])
            abnormal_high_true_num=abnormal_high_true_num+1
    print("abnormal_high_sum",abnormal_high_true_num)

    #真值为异常偏高，且预测区间能覆盖真值的病人数目
    abnormal_high_catched_num=np.sum(K_u * K_l * K_high)

    PICP_ah=np.round(abnormal_high_catched_num/abnormal_high_true_num,3)

    #真值为异常偏高的病人的预测区间(非零值代表异常偏高病人的区间宽度)
    abnormal_high_pred_interval=(y_u_pred-y_l_pred)*K_high
    MPIW_ah=np.round(np.sum(abnormal_high_pred_interval)/abnormal_high_true_num,3)

    #lrh改/

    return PICP_ah,MPIW_ah

#区间转换为中点
def interval_to_mid(y):
    mid=(y[:,0]+y[:,1])/2.0
    #print("mid",mid)
    return mid

def cal_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

#计算类别的评估指标,index代表类别名字
def cal_class_measurement(y_true, y_pred, index_function_class):
    y_true_class=convert_to_class(y_true, index_function_class)

    y_pred_class=convert_to_class(y_pred, index_function_class)
    c_r=classification_report(y_true_class,y_pred_class)

    return c_r
def cal_custom_class_measurement(y_true, y_pred, index_function_class, index_state_bound):
    y_true_class = convert_to_class(y_true, index_function_class)
    y_pred_class_custom = convert_to_class_custom(y_pred, index_state_bound)
    c_r_custom = classification_report(y_true_class, y_pred_class_custom)
    #print("c_r_custom", c_r_custom)
    return c_r_custom

#将y的数值转化为类别
def convert_to_class(y, index_function_class):
    y_class=[]
    for i in range(y.__len__()):
        y_class.append(index_function_class(y[i]))

    #print("y", index_class.__name__, y_class)
    return y_class

#计算与点估计相关的评估指标
def measure_point_by_rmse_cr(y_true, y_pred, index_function_class):
    #区间转换为中点
    if(y_pred.shape[1]==2):
        y_pred=interval_to_mid(y_pred)

    rmse=cal_rmse(y_true,y_pred)
    c_r=cal_class_measurement(y_true,y_pred,index_function_class)

    #print('rmse',rmse)
    #print('c_r',c_r)

    return rmse,c_r

#统计预测出的区间，跨异常状态的比例
def statistic_cross_state_num_percent(y_pred, index_state_bound):
    num=0
    for i in range(len(y_pred)):
        u_y_pred=y_pred[i][0]
        l_y_pred=y_pred[i][1]
        if(index_state_bound[0]>l_y_pred and index_state_bound[0]<u_y_pred):
            num=num+1
            #print("l","bound0","u",l_y_pred,index_class[0],u_y_pred)
        elif(index_state_bound[1] > l_y_pred and index_state_bound[1] < u_y_pred):
            num=num+1
            #print("l", "bound1", "u", l_y_pred, index_class[1], u_y_pred)
    #print("cross_num",num)
    #print("percent",num/len(y_pred))

    cross_num=num
    percent=num/len(y_pred)

    return cross_num,percent

def generate_bound(y_pred,train_bins):
    print("generate_bound")
    #print("y_pred",y_pred)
    y_pred_bound=[]
    num=0
    for i in range(y_pred.__len__()):
        j=0
        #是否为y_pred找到所属区间
        flag=False
        while(j<train_bins.__len__()-1):
            if(y_pred[i]>=train_bins[j] and y_pred[i]<train_bins[j+1]):
                u_bound=train_bins[j+1]
                l_bound=train_bins[j]
                y_pred_bound.append(np.array([u_bound,l_bound]))
                flag=True
                break
            j=j+1

        #y_pred不在预测范围内
        if(flag==False):
            y_pred_bound.append(np.array([-1, -1]))
            num=num+1

    y_pred_bound=np.array(y_pred_bound)
    print("bin not found num",num)
    #print("y_pred_bound",y_pred_bound)
    return y_pred_bound


def convert_to_class_custom(y_pred, index_state_bound):
    print("convert_to_class_custom")
    abnormal_low_bound=index_state_bound[0]
    abnormal_high_bound=index_state_bound[1]
    #统计预测失败的数目
    num1=0
    num2=0

    y_pred_class_custom=[]
    for i in range(len(y_pred)):
        y_p_u = y_pred[i,0]
        y_p_l = y_pred[i,1]

        #上边界小于下边界，预测失败
        if(y_p_u<y_p_l):
            y_pred_class_custom.append(-1)
            num1=num1+1

        #1、不跨边界（全部是开区间）
        #正常
        elif(y_p_l>abnormal_low_bound and y_p_u <abnormal_high_bound):
            y_pred_class_custom.append(1)
        #异常偏高
        elif(y_p_l>abnormal_high_bound):
            y_pred_class_custom.append(2)
        #异常偏低
        elif(y_p_u<abnormal_low_bound):
            y_pred_class_custom.append(0)


        #2、跨边界
        #异常偏低
        elif(y_p_l<=abnormal_low_bound and (y_p_u>=abnormal_low_bound and y_p_u <abnormal_high_bound)):
            y_pred_class_custom.append(0)
        #异常偏高
        elif((y_p_l>abnormal_low_bound and y_p_l<=abnormal_high_bound) and y_p_u>=abnormal_high_bound):
            y_pred_class_custom.append(2)
        #预测失败
        elif(y_p_l<=abnormal_low_bound and y_p_u >=abnormal_high_bound):
            y_pred_class_custom.append(-1)
            num2=num2+1
        else:
            print("remaining")
            print("y_p_u",y_p_u)
            print("y_p_l",y_p_l)


    print("predict error num1,num2",num1,num2)
    y_pred_class_custom=np.array(y_pred_class_custom)

    #print("y_pred_class_custom.shape[0]",y_pred_class_custom.shape[0])
    #print("y_pred.shape[0]",y_pred.shape[0])

    return y_pred_class_custom

#6.26
def statistic_interval(y_true, y_pred):
    y_t=y_true[:,2]
    y_p_u=y_pred[:,0]
    y_p_l=y_pred[:,1]
    y_p_width=y_p_u-y_p_l
    #真值与上边界的距离,及其占整个区间宽度的百分比
    u_dist_percent_captured=[]
    #保存捕获和未捕获的样本，对应的真值
    not_captured_point=[]
    captured_point=[]

    # y_true_label=[]
    # y_pred_dream=[]

    for i in range(y_t.shape[0]):
        if(y_t[i]>=y_p_l[i] and y_t[i]<=y_p_u[i]):
            dist=y_p_u[i]-y_t[i]
            percent=dist/y_p_width[i]
            u_dist_percent_captured.append(np.array([dist,percent]))
            captured_point.append(y_true[i,2])
            #已经捕获，我认为它是正常
            # y_pred_dream.append(1)
            # y_true_label.append(ft3(y_true[i,2]))
            # if(percent>=0.7):
            #     print("sta_")
            #     print("sta_pred",y_p_l[i],y_p_u[i],y_p_width[i],dist,percent)
            #     print("sta_true",y_true[i,0],y_true[i,1],y_true[i,0]-y_true[i,1])
        #未被捕获的
        else:
            #print("sta_pred", y_p_l[i], y_p_u[i], y_p_width[i])
            #print("sta_true", y_true[i, 0], y_true[i, 1], y_true[i, 0] - y_true[i, 1])
            not_captured_point.append(y_true[i,2])
            # y_pred_dream.append(2)
            # y_true_label.append(ft3(y_true[i, 2]))

    #防止出错
    # y_pred_dream.append(0)
    # y_true_label.append(0)

    #c_r=classification_report(y_true_label,y_pred_dream)
    #print("c_r",c_r)


    u_dist_percent_captured=np.array(u_dist_percent_captured)

    mean_percent=np.mean(u_dist_percent_captured[:,1])
    mean_dist=np.mean(u_dist_percent_captured[:,0])

    std_percent=np.std(u_dist_percent_captured[:,1])
    std_dist = np.std(u_dist_percent_captured[:, 0])

    print("sta_dist",mean_dist,std_dist)
    print("sta_percent",mean_percent,std_percent)

    # 未捕获
    not_captured_percent=cal_state_percent(not_captured_point)
    not_captured_point=np.array(not_captured_point)
    ncp_mean = np.mean(not_captured_point)
    ncp_std = np.std(not_captured_point)
    print("not captured",ncp_mean,ncp_std,not_captured_percent)

    # 捕获
    captured_percent=cal_state_percent(captured_point)
    captured_point=np.array(captured_point)
    cp_mean = np.mean(captured_point)
    cp_std = np.std(captured_point)
    print("captured",cp_mean,cp_std,captured_percent)

    #绘制百分比分布图
    #plot_percent_hist(u_dist_percent_captured[:,1],50)
    plot_percent_hist(not_captured_point,50)
    plot_percent_hist(captured_point,10)



#6.26
def plot_percent_hist(data,bins):
    # 4、按照生成的bin_list绘制数据的分布直方图，检验bin_list的效果
    plt.hist(data, bins=bins,normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    # 显示横轴标签
    plt.xlabel("patient")
    # 显示纵轴标签
    plt.ylabel("percent")
    # 显示图标题
    plt.title("hist" + str(len(data)))
    plt.show()

def cal_state_percent(data):
    num0=0
    num1=0
    num2=0
    num_all=len(data)
    for i in range(len(data)):
        state=ft3(data[i])
        if(state==0):
            num0=num0+1
        elif(state==1):
            num1=num1+1
        else:
            num2=num2+1

    return [num0/num_all,num1/num_all,num2/num_all]

#评估实验结果
def measure_result(Y_true, Y_pred, inf):
    my_plot(Y_true, Y_pred, inf)
    PICP, MPIW = cal_PICP_MPIW(Y_true, Y_pred)
    PICP_ah, MPIW_ah = cal_ah_PICP_MPIW(Y_true, Y_pred)

    print("PICP,MPIW,PICP_ah, MPIW_ah",PICP,MPIW,PICP_ah, MPIW_ah)


