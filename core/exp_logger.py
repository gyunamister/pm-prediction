from tensorflow.keras.callbacks import CSVLogger
import os
class ExperimentLogger(CSVLogger):
    def __init__(self, filename,exp_info):
        self.filename = filename
        self.exp_info = exp_info
        print(filename)

    def on_test_begin(self, logs=None):
        # open csv file
        print('test begin')

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        # write the contents of the dictionary logs to csv file
        # sample content of logs {'batch': 0, 'size': 2, 'loss': -0.0, 'accuracy': 1.0}
        print(logs)

    def on_test_end(self, logs=None):
        if os.path.exists(self.filename):
            self.write_exp_info(logs)
        else:
            self.write_header()
            self.write_exp_info(logs)
        print('test end')

    def write_exp_info(self,logs):
        with open(self.filename, 'a') as f:
            for key in self.exp_info:
                f.write(str(self.exp_info[key]) + ",")
            f.write("\n")
            f.close()
    
    def write_header(self):
        with open(self.filename, "w") as f:
            for key in self.exp_info:
                f.write(key + ",")
            f.write("loss\n")
            print('File created: {}'.format(self.filename))
            f.close()

    def write_results(self,logs):
        with open(self.filename, 'a') as f:
            for key in self.exp_info:
                f.write(str(self.exp_info[key]) + ",")
            f.write(str(logs["loss"])+ "\n")
            f.close()
