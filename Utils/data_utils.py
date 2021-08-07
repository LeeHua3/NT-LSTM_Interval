import numpy as np
#将y_u,y_l,y_value按顺序合并返回
def get_y_data(data_dict):
    y_u=data_dict['y_u']
    y_l=data_dict['y_l']

    y_value=data_dict['y_value']

    batch_num=len(y_u)

    batches_y=[]
    for i in range(batch_num):
        batch_size=len(y_u[i])
        batch_y=[]
        for j in range(batch_size):
            #[0]代表ft3
            y=np.array([y_u[i][j],y_l[i][j],y_value[i][j][0]])
            batch_y.append(y)
        batch_y=np.array(batch_y)
        batches_y.append(batch_y)

    return batches_y


#将每个batch转换为np.array
def get_np_array_batches(data_batches):
    batch_num=len(data_batches)
    for i in range(batch_num):
        data_batches[i]=np.array(data_batches[i])
    return data_batches
