import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)


from data_process.generate_jiakang_data_NT_increment_mul_labs_with_knn_interval import get_data_with_KNN_Interval
from measure_plot.measure import measure_result
from model.NTLSTM_Interval import NTLSTM_Interval
from Utils.data_utils import get_np_array_batches
from Utils.data_utils import get_y_data

#
def training(path, learning_rate, training_epochs, train_dropout_prob, hidden_dim, fc_dim, key, model_path, time, train_data_dict):

    tf.reset_default_graph()

    #data_train_batches =train_data_dict[0]
    data_train_batches = train_data_dict['x_data']
    data_train_batches=get_np_array_batches(data_train_batches)
    #print("data_train_batches",data_train_batches)

    #elapsed_train_batches=train_data_dict[1]
    elapsed_train_batches = train_data_dict['delta_time']
    elapsed_train_batches = get_np_array_batches(elapsed_train_batches)
    #print("elapsed_train_batches",elapsed_train_batches)

    #labels_train_batches=train_data_dict[2]

    y_ulp_train_batches = get_y_data(train_data_dict)
    y_ulp_train_batches = get_np_array_batches(y_ulp_train_batches)
    #print("y_ulp_train_batches",y_ulp_train_batches)

    number_train_batches = len(data_train_batches)
    #print("Train data_process is loaded!")

    #input_dim=3：ft3,ft4,tsh
    #print("data_train_batches[0]",data_train_batches[0])
    input_dim = data_train_batches[0].shape[2]

    #output_dim = labels_train_batches[0].shape[1]
    #output_dim=3:ulp-p
    output_dim = y_ulp_train_batches[0].shape[1]-1

    #目前只预测单个指标
    lstm = NTLSTM_Interval(input_dim, output_dim, hidden_dim, fc_dim, key)

    #IP书签：8.3
    #cross_entropy, y_pred, y, logits, labels = lstm.get_cost_acc()
    Loss, y_ulp_true, y_ulp_pred = lstm.get_cost_acc()

    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        #debug
        #sess = tf_dbg.LocalCLIDebugWrapperSession(sess)
        sess.run(init)
        # print_weight
        # print("before train")
        # print("lstm.W_decomp", sess.run(lstm.W_decomp[0]))
        # print("lstm.b_decomp", sess.run(lstm.b_decomp[0]))
        for epoch in range(training_epochs):  #
            # Loop over all batches
            total_cost = 0
            for i in range(number_train_batches):  #
                # batch_xs is [number of patients x sequence length x input dimensionality]
                # batch_xs, batch_ys, batch_ts = data_train_batches[i], labels_train_batches[i], \
                #                                          elapsed_train_batches[i]
                batch_xs, batch_ys, batch_ts = data_train_batches[i], y_ulp_train_batches[i], \
                                               elapsed_train_batches[i]


                batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[2]])

                sess.run(optimizer,feed_dict={lstm.input: batch_xs, lstm.y_ulp_true: batch_ys,\
                                              lstm.keep_prob:train_dropout_prob, lstm.time:batch_ts})

                #显示训练中的详细信息
                # print("batch:",i)
                # train_loss,train_y_ulp_true,train_y_ulp_pred=sess.run(lstm.get_cost_acc(), feed_dict={lstm.input: batch_xs, lstm.y_ulp_true: batch_ys, \
                #                                lstm.keep_prob: train_dropout_prob, lstm.time: batch_ts})
                # print("train_loss",train_loss)
                # print("train_y_ulp_true", train_y_ulp_true)
                # print("train_y_ulp_pred", train_y_ulp_pred)
            #print("epoch:",epoch)


        print("Training is over!")
        saver.save(sess,model_path)
        # print_weight
        # print("after train")
        # print("lstm.W_decomp", sess.run(lstm.W_decomp[0]))
        # print("lstm.b_decomp", sess.run(lstm.b_decomp[0]))

        Y_pred = []
        Y_true = []
        # Labels = []
        # Logits = []
        for i in range(number_train_batches):  #
            batch_xs, batch_ys, batch_ts = data_train_batches[i], y_ulp_train_batches[i], \
                                                     elapsed_train_batches[i]
            batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[2]])
            # c_train, y_pred_train, y_train, logits_train, labels_train = sess.run(lstm.get_cost_acc(), feed_dict={
            #     lstm.input:
            #                                                                                            batch_xs, lstm.labels: batch_ys, \
            #                                lstm.keep_prob: train_dropout_prob, lstm.time: batch_ts})
            train_loss,train_y_ulp_true,train_y_ulp_pred = sess.run(lstm.get_cost_acc(), feed_dict={
                lstm.input:
                    batch_xs, lstm.y_ulp_true: batch_ys, \
                lstm.keep_prob: train_dropout_prob, lstm.time: batch_ts})

            if i > 0:
                # Y_true = np.concatenate([Y_true, y_train], 0)
                # Y_pred = np.concatenate([Y_pred, y_pred_train], 0)
                # Labels = np.concatenate([Labels, labels_train], 0)
                # Logits = np.concatenate([Logits, logits_train], 0)
                Y_true = np.concatenate([Y_true, train_y_ulp_true], 0)
                Y_pred = np.concatenate([Y_pred, train_y_ulp_pred], 0)

            else:
                # Y_true = y_train
                # Y_pred = y_pred_train
                # Labels = labels_train
                # Logits = logits_train
                Y_true = train_y_ulp_true
                Y_pred = train_y_ulp_pred


        print("train")
        print("Y_true", Y_true)
        print("Y_pred",Y_pred)
        return Y_true,Y_pred


def testing(path,hidden_dim,fc_dim,key,model_path,time,test_data_dict):
    tf.reset_default_graph()

    #data_test_batches=test_data_resplit[0]
    data_test_batches = test_data_dict['x_data']
    data_test_batches = get_np_array_batches(data_test_batches)

    #elapsed_test_batches=test_data_resplit[1]
    elapsed_test_batches = test_data_dict['delta_time']
    elapsed_test_batches = get_np_array_batches(elapsed_test_batches)

    #labels_test_batches=test_data_resplit[2]
    y_ulp_test_batches = get_y_data(test_data_dict)
    y_ulp_test_batches = get_np_array_batches(y_ulp_test_batches)

    number_test_batches = len(data_test_batches)

    print("Test data_process is loaded!")

    input_dim = data_test_batches[0].shape[2]
    # output_dim=2:ulp-p
    output_dim = y_ulp_test_batches[0].shape[1]-1

    test_dropout_prob = 1.0
    lstm_load = NTLSTM_Interval(input_dim, output_dim, hidden_dim, fc_dim, key)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)

        Y_true = []
        Y_pred = []
        # Labels = []
        # Logits = []
        for i in range(number_test_batches):
            batch_xs, batch_ys, batch_ts = data_test_batches[i], y_ulp_test_batches[i], \
                                                     elapsed_test_batches[i]
            batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[2]])
            # c_test, y_pred_test, y_test, logits_test, labels_test = sess.run(lstm_load.get_cost_acc(),
            #                                                                  feed_dict={lstm_load.input: batch_xs,
            #                                                                             lstm_load.labels: batch_ys,\
            #                                                                             lstm_load.time: batch_ts,\
            #                                                                             lstm_load.keep_prob: test_dropout_prob})
            test_loss,test_y_ulp_true,test_y_ulp_pred = sess.run(lstm_load.get_cost_acc(),
                                                                             feed_dict={lstm_load.input: batch_xs,
                                                                                        lstm_load.y_ulp_true: batch_ys, \
                                                                                        lstm_load.time: batch_ts, \
                                                                                        lstm_load.keep_prob: test_dropout_prob})

            if i > 0:
                # Y_true = np.concatenate([Y_true, y_test], 0)
                # Y_pred = np.concatenate([Y_pred, y_pred_test], 0)
                # Labels = np.concatenate([Labels, labels_test], 0)
                # Logits = np.concatenate([Logits, logits_test], 0)
                Y_true = np.concatenate([Y_true, test_y_ulp_true], 0)
                Y_pred = np.concatenate([Y_pred, test_y_ulp_pred], 0)
            else:
                # Y_true = y_test
                # Y_pred = y_pred_test
                # Labels = labels_test
                # Logits = logits_test
                Y_true = test_y_ulp_true
                Y_pred = test_y_ulp_pred

        print("test")
        print("Y_true",Y_true)
        print("Y_pred", Y_pred)

        return Y_true,Y_pred


def main(argv):
    #training_mode = int(sys.argv[1])
    training_mode = 1
    #path = str(sys.argv[2])
    path = "./Split0"
    #learning_rate = float(sys.argv[3])
    learning_rate = 0.0002
    #training_epochs = int(sys.argv[4])
    training_epochs = 400
    #dropout_prob = float(sys.argv[5])
    dropout_prob = 1.0
    #hidden_dim = int(sys.argv[6])
    hidden_dim=128
    #fc_dim = int(sys.argv[7])
    fc_dim = 64
    #model_path = str(sys.argv[8])
    model_path = "model_save/final_model"

    experiment_time=10

    for i in range(experiment_time):

        #每一次循环，进行不同的随机分割，比例相同（train0.7,test0.3）
        #train_data_resplit, test_data_resplit = read_data_split_train_test(path)
        train_batch_dict,test_batch_dict =get_data_with_KNN_Interval()
        #train
        training_mode=1
        train_Y_true,train_Y_pred=training(path, learning_rate, training_epochs, dropout_prob, hidden_dim, fc_dim, training_mode, model_path,i,train_batch_dict)
        measure_result(train_Y_true,train_Y_pred,"train")
        #test
        training_mode=0
        test_Y_true,test_Y_pred=testing(path, hidden_dim, fc_dim, training_mode, model_path,i,test_batch_dict)
        measure_result(test_Y_true,test_Y_pred,"test")



if __name__ == "__main__":
   #bash(sys.argv[1:])
    #不传参数
    no_args=0
    main(no_args)

