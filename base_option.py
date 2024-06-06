import argparse
class BaseOptions():
    '''
        This class defines the  options used for during training and test time
    '''
    def __init__(self):
        self.parse=argparse.ArgumentParser(description='enhancement underwater image')
    def init(self):
        self.parse.add_argument('--dataset', type=str, default='./dataset/', help='train_dataset')
        self.parse.add_argument('--valset', type=str, default='./dataset/', help='val_dataset')
        self.parse.add_argument('--unpaired_image', type=str, default='./dataset/unpaired_image/', help='unpaired_image')
        self.parse.add_argument('--testset', type=str, default='./dataset/testset/', help='Dataset address for testing')
        self.parse.add_argument('--save_model', type=str, default='./save_model/', help='save_model_dir')
        self.parse.add_argument('--save_image', type=str, default='./save_image/', help='save_image_dir')
        self.parse.add_argument('--batch_size', type=int, default=16, help='batch size')
        self.parse.add_argument('--device', type=str, default='cuda:0', help='set train device')
        self.parse.add_argument('--epoch_end', type=int, default=200, help='train end epoch ')
        self.parse.add_argument('--lr', type=float, default=0.0002, help='learning rate ')
        self.parse.add_argument('--niter', type=int, default=100, help='batch size')
        self.parse.add_argument('--niter_decay', type=int, default=100, help='niter_decay')
        self.parse.add_argument('--beta', type=float, default=0.5, help='beta')
        self.parse.add_argument('--resolution', type=int, default=512, help='use image resolution')
        self.parse.add_argument('--image_save_status', type=bool, default=False, help='save image')
        self.parse.add_argument('--train_log', type=str, default='./DDformer.txt', help='train loss and psnr log')
        self.parse.add_argument('--train_state', type=str, default='DDformer_unpaired： MutiTransformer gan 0.2 L1 10  unpaired 0.5', help='train set state')


        return self.parse