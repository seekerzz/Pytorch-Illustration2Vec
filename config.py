import os

class Config:
    def __init__(self):
        self.dropout_rate = 0.1
        self.lr = 0.01
        self.lr_decay_per_epoch = 0.5
        self.available_GPUs = "0,1"
        # self.img_path = "../test_imgs"
        self.img_path = "../i2v_data/images"
        self.pkl_path = "./imgid2attr.pkl"
        self.log_dir = "./evts_sgd"
        self.model_saving_path = "./saved_models"
        self.batch_size = 128
        self.img_size = 256
        self.n_workers = 8*len(self.available_GPUs.split(","))
        #self.n_workers = 0
        self.epoch = 400

        self.print_n_iter = 100
        #self.save_n_iter = 30
        #self.val_n_iter = 50
        self.save_n_iter = int(0.5*2500000/self.batch_size)
        self.val_n_iter = int(0.5*2500000/self.batch_size)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_saving_path, exist_ok=True)
        with open(os.path.join(self.log_dir, "config.txt"),"w") as f:
            for k,v in self.__dict__.items():
                print(k,v)
                f.write(k + " " + str(v) +"\n")




