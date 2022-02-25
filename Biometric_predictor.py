import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import sklearn
from sklearn import svm
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
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


class plotter:
    def __init__(self, accel_txt, gyro_txt, acc_col, gyro_col):
        self.accel_txt = accel_txt 
        self.gyro_txt = gyro_txt 
        self.acc_col = acc_col
        self.gyro_col = gyro_col
        self.model = None 

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
        fig.write_html(''.join([GRAPH_DIR, name, ".html"]))
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
        fig.write_html(''.join([GRAPH_DIR, name, "_histogram", ".html"]))
        fig.write_image(''.join([GRAPH_DIR, name, "_histogram", ".png"]), width=1000, height=1000)

        '''
        df_act = dframe
        tmp = sensor.split("_")
        title_card = ''.join([ACT_DICT[act], " ", tmp[0], " ", tmp[1], " biometrics"])
        fig = px.scatter_matrix(df_act, dimensions=dof, color="ActivityLabel", 
                                title=title_card, template=THEME)
        
        fig.update_traces(diagonal_visible=False)
        # rename legend items
        fig.for_each_trace(lambda x: x.update(name = ACT_DICT[x.name]))
        fig.show()

        fig.write_html(''.join([GRAPH_DIR, name, ".html"]))
        fig.write_image(''.join([GRAPH_DIR, name, ".png"]), width=WIDTH_IMG, height=HEIGHT_IMG)
        '''

    
    def heatmap(self, cname, y_actual, y_pred, acc, device, clf_svm):
        activites=[ACT_DICT[key] for key in ACT_DICT]
        cmatrix = confusion_matrix(y_actual, y_pred)
        svm_data = ''.join([" (", "kernel=", str(clf_svm.kernel), ", C=", str(clf_svm.C), ", gamma=", str(clf_svm.gamma), "):"])
        xlabel = ''.join([device, " SVM", svm_data, " Accuracy = ", str(round(acc,3)), "<br><br>Actual Class"])
        fig = px.imshow(cmatrix, text_auto=True, template=THEME, x=activites, y=activites,   
                        labels={"x": xlabel, "y": "Predicted Class", "color": "Guesses"}) 

        fig.update_xaxes(side="top")
        fig.show()
        fig.write_html(''.join([GRAPH_DIR, cname, ".html"]))
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


    watch_1600 = plotter(ACCEL_W_DATA_1600, GYRO_W_DATA_1600, ACCEL_COLS, GYRO_COLS)
    watch_1600.plot_activity(exercise, "matrix", "accel_watch", "Accel (m/s^2)")
    watch_1600.plot_activity(exercise, "matrix", "gyro_watch", "Gyro (radians/s)")
    watch_1600.plot_activity(exercise, "matrix", "imu_watch", "Accel (m/s^2), Gyro (radians/s)")

    watch_1600.plot_activity(exercise, "line", "accel_watch", "Accel (m/s^2)")
    watch_1600.plot_activity(exercise, "line", "gyro_watch", "Gyro (radians/s)")
    watch_1600.plot_activity(exercise, "line", "imu_watch", "Accel m/s^2, Gyro (radians/s)")

    watch_1600.svm_classifier(svm.SVC(kernel='rbf', C=3, gamma=0.1), "confusion_matrix_watch", "Watch", MODEL_W_NAME)
    watch_1600.run_svm_model("confusion_matrix_watch", "Watch", MODEL_W_NAME)



    end = time.time()
    print(f"Total Time: {round(end-start, 3)}s")

main()

