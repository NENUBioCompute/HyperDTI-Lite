import geoopt


class hyperparameter:
    def __init__(self):
        self.K_fold = 5
        self.Epoch = 500
        self.Patience = 7
        self.Batch_size = 128

        self.threshold = 0.36
        self.curvature = 1
        self.poincare_ball = geoopt.PoincareBall(self.curvature)
        self.hyperbolic_dim = 32
        self.hyperbolic_lr = 0.1
        self.lr_patience = 3
        self.lr_factor = 0.8

        self.anchor_factor = 0.80
        self.loss_rate = 200.0
        self.hyper_threshold = 5.3864

        self.FC_Dropout = 0.1
        self.char_embedding_dim = 64
        self.conv_base_size = 40
        self.drug_kernel = [4, 6, 8]
        self.drug_MAX_LENGH = 100
        self.target_kernel = [4, 8, 12]
        self.target_MAX_LENGH=1000