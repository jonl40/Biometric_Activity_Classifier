import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import tensorflow as tf
import sklearn
from sklearn import svm
from sklearn import metrics 
import pickle
import os 
import sys 
import time 

'''
Phone in pocket 
Watch on wrist of dominant hand 

20Hz sampling rate 
~64800 samples -> ~54 minutes

Subject-id: unique to subject, Range: 1600-1650
ActivityLabel: unique activity, Range: A-S (no “N” value)
Timestamp: Integer, Linux time
x: x axis of sensor 
y: y axis of sensor 
z: z axis of sensor 
'''

# accelerometer units: m/s^2
# gyroscope units: radians/s

WIDTH_IMG = 1980
HEIGHT_IMG = 1080
THEME = "plotly"

DIR = "raw/"
GRAPH_DIR = "graphs/"
ACCEL_P_DATA_1600 = DIR + "data_1600_accel_phone.txt"
GYRO_P_DATA_1600 = DIR + "data_1600_gyro_phone.txt"
ACCEL_W_DATA_1600 = DIR + "data_1600_accel_watch.txt"
GYRO_W_DATA_1600 = DIR + "data_1600_gyro_watch.txt"

ACCEL_COLS = ["Subject-id", "ActivityLabel", "Timestamp", "x_acc", "y_acc", "z_acc"]
GYRO_COLS = ["Subject-id", "ActivityLabel", "Timestamp", "x_gyro", "y_gryo", "z_gyro"]
MODEL_P_NAME = "svm_phone.sav"
MODEL_W_NAME = "svm_watch.sav"

ACT_DICT = {"A": "walking", "B": "jogging", "C": "stairs", "D": "sitting", "E": "standing", 
            "F": "typing", "G": "teeth", "H": "soup", "I": "chips", "J": "pasta", "K": "drinking",
            "L": "sandwich", "M": "kicking", "O": "catch", "P": "dribbling", "Q": "writing", 
            "R": "clapping", "S": "folding"}

SAMPLES_PER_SEC = 20 


class plotter:
    def __init__(self, accel_txt, gyro_txt, acc_col, gyro_col):
        self.accel_txt = accel_txt 
        self.gyro_txt = gyro_txt 
        self.acc_col = acc_col
        self.gyro_col = gyro_col
        self.model = None 
        self.input_tensor = None 
        self.output_tensor = None 

        # load df 
        if accel_txt: 
            self.accel_df = pd.read_csv(accel_txt)

        if gyro_txt: 
            self.gyro_df = pd.read_csv(gyro_txt)

        self.preprocess_df()


    def preprocess_df(self):
        if self.accel_txt:
            # set columns
            self.accel_df.columns = self.acc_col
            # remove ';' from last column
            self.accel_df["z_acc"] = self.accel_df["z_acc"].str.rstrip(";")
            # convert acceleration data to numeric 
            self.accel_df[self.accel_df.columns[3:]] = self.accel_df[self.accel_df.columns[3:]].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
            self.accel_df["Timestamp"] = pd.to_datetime(self.accel_df["Timestamp"], unit='s', errors='coerce') 
        
        if self.gyro_txt:
            # set columns
            self.gyro_df.columns = self.gyro_col
            # remove ';' from last column
            self.gyro_df["z_gyro"] = self.gyro_df["z_gyro"].str.rstrip(";")
            # convert acceleration data to numeric 
            self.gyro_df[self.gyro_df.columns[3:]] = self.gyro_df[self.gyro_df.columns[3:]].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
            self.gyro_df["Timestamp"] = pd.to_datetime(self.gyro_df["Timestamp"], unit='s', errors='coerce') 
        
        # combine accel and gyro data into one df  
        if self.accel_txt and self.gyro_txt:
            df_tmp = self.gyro_df
            df_tmp = df_tmp.drop(["Subject-id", "ActivityLabel", "Timestamp"], axis=1)
            self.merged_df = pd.concat([self.accel_df, df_tmp], axis=1, join="inner")


    def line_graph(self, dframe, ylabel, sensor, act):
        df_act = dframe.loc[dframe["ActivityLabel"] == act]
        dof = dframe.columns[3:]
        print(dof)
        tmp = sensor.split("_")
        title_card = ''.join([ACT_DICT[act], " ", tmp[0], " ", tmp[1], " biometrics"])
        fig = px.line(df_act[:101], x=df_act[:101].index, y=dof, 
                      labels={'x': "Samples", "value": ylabel, "variable": "Axis"}, 
                      title=title_card, template=THEME, markers=True)
        
        fig.show()
        name = ''.join([ACT_DICT[act], "_", sensor])
        # fig.write_html(''.join([GRAPH_DIR, name, ".html"]))
        fig.write_image(''.join([GRAPH_DIR, name, ".png"]), width=WIDTH_IMG, height=HEIGHT_IMG)


    def scatter_graph(self, dframe, sensor, act):
        dof = dframe.columns[3:]
        name = ''.join(["PCA_", ACT_DICT[act], "_", sensor])
        print(dof)

        # scatter matrix with histogram 
        tmp = sensor.split("_")
        s = ' '.join(tmp) 
        t = ''.join(["scatterplot matrix ", ACT_DICT[act], " ", s])
        df_act = dframe.loc[dframe["ActivityLabel"] == act]
        df_act["ActivityLabel"] = df_act["ActivityLabel"].replace(to_replace=act, value=ACT_DICT[act])

        fig = ff.create_scatterplotmatrix(df_act.drop(["Subject-id", "Timestamp"], axis=1),
                                          diag="histogram", index="ActivityLabel",
                                          title=t, height=1000, width=1000)

        fig.show()
        # fig.write_html(''.join([GRAPH_DIR, name, "_histogram", ".html"]))
        fig.write_image(''.join([GRAPH_DIR, name, "_histogram", ".png"]), width=1000, height=1000)

    
    def heatmap(self, cname, y_actual, y_pred, acc, device, clf_svm):
        activites=[ACT_DICT[key] for key in ACT_DICT]
        cmatrix = metrics.confusion_matrix(y_actual, y_pred)
        svm_data = ''.join([" (", "kernel=", str(clf_svm.kernel), ", C=", str(clf_svm.C), ", gamma=", str(clf_svm.gamma), "):"])
        xlabel = ''.join([device, " SVM", svm_data, " Accuracy = ", str(round(acc,3)), "<br><br>Predicted Class"])
        fig = px.imshow(cmatrix, text_auto=True, template=THEME, x=activites, y=activites,   
                        labels={"x": xlabel, "y": "Actual Class", "color": "Guesses"}) 

        fig.update_xaxes(side="top")
        fig.show()
        # fig.write_html(''.join([GRAPH_DIR, cname, ".html"]))
        fig.write_image(''.join([GRAPH_DIR, cname, ".png"]), width=1000, height=1000)


    def plot_activity(self, act, graph, sensor, ylabel=""):
        if "accel" in sensor:
            df = self.accel_df
        elif "gyro" in sensor:
            df = self.gyro_df
        elif "imu" in sensor:
            df = self.merged_df
        else:
            print("sensor must contain 'accel', 'gryo', or 'imu' !")
            sys.exit(0)

        if graph == "line":
            self.line_graph(df, ylabel, sensor, act)
        elif graph == "matrix":
            self.scatter_graph(df, sensor, act)


    def svm_classifier(self, classifier, cname, device, model_name):
        # features, accel and gyro xyz readings
        dof = self.merged_df.columns[3:]
        x = self.merged_df[list(dof)]
        x = np.array(x)

        # class labels
        y = self.merged_df["ActivityLabel"]
        y = np.array(y)

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
        
        # try loading model
        if self.model:
            print("Loading Model")
            clf = self.model
        # train model if unsuccessfully trained 
        else:
            print("Training Model")
            clf = classifier
            clf.fit(x_train, y_train)

        print("Predicting")
        y_predict_svm = clf.predict(x_test)
        acc = metrics.accuracy_score(y_test, y_predict_svm)
        print("SVM accuracy: ", acc)

        pickle.dump(clf, open(model_name, 'wb'))
        self.heatmap(cname, y_test, y_predict_svm, acc, device, clf)


    def run_svm_model(self, cname, device, model_file):
        try:
            self.model = pickle.load(open(model_file, 'rb'))
            self.svm_classifier(None, cname, device, model_file)
            # clear model, model must be loaded each time 
            self.model = None 
        except:
            print(f"Failed to load model file: {model_file} !")
            self.model = None 
            sys.exit(0)


    def heatmap_model(self, device, confusion, accuracy):
        activites=[ACT_DICT[key] for key in ACT_DICT]
        xlabel = ''.join([device, " Neural Network:", " Accuracy = ", str(round(accuracy,3)), "<br><br>Predicted Class"])
        fig = px.imshow(confusion, text_auto=True, template=THEME, x=activites, y=activites,   
                        labels={"x": xlabel, "y": "Actual Class", "color": "Guesses"}) 

        fig.update_xaxes(side="top")
        fig.show()
        fig.write_html(''.join([GRAPH_DIR, "confusion_matrix_neural_net_", device, ".html"]))
        fig.write_image(''.join([GRAPH_DIR, "confusion_matrix_neural_net_", device, ".png"]), width=1000, height=1000)


    def graph_model(self, df_history, metrics, var, t, newnames, device):
        fig = px.line(df_history, x=df_history.index, y=metrics, 
                labels={"index": "Epochs", "variable": var, "value": var}, 
                title=t , template=THEME, markers=True)

        fig.for_each_trace(lambda x: x.update(name = newnames[x.name]))
        fig.show()
        fig.write_html(''.join([GRAPH_DIR, device, "_" , var, ".html"]))
        fig.write_image(''.join([GRAPH_DIR, device, "_" , var, ".png"]), width=1000, height=1000)


    def normalize_tensor(self, hz, interval_s):
        samples_per_activity = int(hz * interval_s)
        inputs = []
        outputs = []
        num = 0 
        # one hot encoded vectors
        encode = np.eye(len(ACT_DICT))
        

        for key in ACT_DICT:
            # reset index
            df_act = self.merged_df.loc[self.merged_df["ActivityLabel"] == key].reset_index()

            num_actions = int(len(df_act.index) / samples_per_activity)
            act_label = encode[num]
            num += 1 
            print(f"key: {key}")
            print(f"len(df_act.index): {len(df_act.index)}")
            print(f"num_actions: {num_actions}\n")

            for i in range(num_actions):
                tensor = [] 
                for j in range(samples_per_activity):
                    indx = i * samples_per_activity + j
                    tensor += [ df_act["x_acc"][indx],
                                df_act["y_acc"][indx],
                                df_act["z_acc"][indx],
                                df_act["x_gyro"][indx],
                                df_act["y_gryo"][indx],
                                df_act["z_gyro"][indx]
                              ]
                
                inputs.append(tensor)
                outputs.append(act_label)
        
        # min max normalization, x' = (x - min(x)) / (max(x) - min(x))
        inputs = tf.math.divide(tf.math.subtract(inputs, tf.math.reduce_min(inputs)),
                                tf.math.subtract(tf.math.reduce_max(inputs), tf.math.reduce_min(inputs)))

        # convert the list to numpy array
        self.input_tensor = np.array(inputs)
        self.output_tensor = np.array(outputs)


    def train_neural_net(self, hz, interval_s, device, e_num, b_num, lr=0.001):

        self.normalize_tensor(hz, interval_s)

        SEED = 1337
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        num_inputs = len(self.input_tensor)
        randomize = np.arange(num_inputs)
        np.random.shuffle(randomize)

        # Swap the consecutive indexes (0, 1, 2, etc) with the randomized indexes
        self.input_tensor = self.input_tensor[randomize]
        self.output_tensor = self.output_tensor[randomize]

        TRAIN_SPLIT = int(0.6 * num_inputs)
        TEST_SPLIT = int(0.2 * num_inputs + TRAIN_SPLIT)

        inputs_train, inputs_test, inputs_validate = np.split(self.input_tensor, [TRAIN_SPLIT, TEST_SPLIT])
        outputs_train, outputs_test, outputs_validate = np.split(self.output_tensor, [TRAIN_SPLIT, TEST_SPLIT])

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(30, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(len(ACT_DICT), activation='softmax')) 
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["accuracy"])
        history = model.fit(inputs_train, outputs_train, epochs=e_num, batch_size=b_num, validation_data=(inputs_validate, outputs_validate))
        pred = model.predict([inputs_test])
        acc = metrics.accuracy_score(np.argmax(outputs_test, axis=1), np.argmax(pred, axis=1))

        # save model
        model.save(''.join([device, "_model.h5"]))

        df_history = pd.DataFrame.from_dict(history.history)

        rename = {"accuracy": "Training Accuracy", "val_accuracy": "Validation Accuracy"}
        self.graph_model(df_history, ["accuracy", "val_accuracy"], "Accuracy", ''.join([device, " Model Accuracy"]), rename, device)

        rename = {"loss": "Training Loss", "val_loss": "Validation Loss"}
        self.graph_model(df_history, ["loss", "val_loss"], "Loss", ''.join([device, " Model Loss"]), rename, device)

        confusion = tf.math.confusion_matrix(labels=np.argmax(outputs_test, axis=1), predictions=np.argmax(pred, axis=1), num_classes=len(ACT_DICT))
        self.heatmap_model(device, confusion, acc)


def main():
    start = time.time()

    # change dir to python script location 
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    exercise = "A"

    phone_1600 = plotter(ACCEL_P_DATA_1600, GYRO_P_DATA_1600, ACCEL_COLS, GYRO_COLS)

    phone_1600.plot_activity(exercise, "matrix", "accel_phone", "Accel (m/s^2)")
    phone_1600.plot_activity(exercise, "matrix", "gyro_phone", "Gyro (radians/s)")
    phone_1600.plot_activity(exercise, "matrix", "imu_phone", "Accel (m/s^2), Gyro (radians/s)")

    phone_1600.plot_activity(exercise, "line", "accel_phone", "Accel (m/s^2)")
    phone_1600.plot_activity(exercise, "line", "gyro_phone", "Gyro (radians/s)")
    phone_1600.plot_activity(exercise, "line", "imu_phone", "Accel (m/s^2), Gyro (radians/s)")

    phone_1600.svm_classifier(svm.SVC(kernel='rbf', gamma=0.1), "confusion_matrix_phone", "Phone", MODEL_P_NAME)
    phone_1600.run_svm_model("confusion_matrix_phone", "Phone", MODEL_P_NAME)
    phone_1600.train_neural_net(SAMPLES_PER_SEC, 0.25, "Phone", 500, 30, 0.00055)


    watch_1600 = plotter(ACCEL_W_DATA_1600, GYRO_W_DATA_1600, ACCEL_COLS, GYRO_COLS)

    watch_1600.plot_activity(exercise, "matrix", "accel_watch", "Accel (m/s^2)")
    watch_1600.plot_activity(exercise, "matrix", "gyro_watch", "Gyro (radians/s)")
    watch_1600.plot_activity(exercise, "matrix", "imu_watch", "Accel (m/s^2), Gyro (radians/s)")

    watch_1600.plot_activity(exercise, "line", "accel_watch", "Accel (m/s^2)")
    watch_1600.plot_activity(exercise, "line", "gyro_watch", "Gyro (radians/s)")
    watch_1600.plot_activity(exercise, "line", "imu_watch", "Accel m/s^2, Gyro (radians/s)")

    watch_1600.svm_classifier(svm.SVC(kernel='rbf', C=3, gamma=0.1), "confusion_matrix_watch", "Watch", MODEL_W_NAME)
    watch_1600.run_svm_model("confusion_matrix_watch", "Watch", MODEL_W_NAME)
    watch_1600.train_neural_net(SAMPLES_PER_SEC, 0.25, "Watch", 400, 40, 0.0009)


    end = time.time()
    print(f"Total Time: {round(end-start, 3)}s")

main()

