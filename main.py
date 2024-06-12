# -*- coding:utf-8 -*-
import os
import datetime
import torch
import thop
import argparse, sys
import numpy as np
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from data.luad import build_dataset_3d
from algorithm.jocor import JoCoR
# from ptflops import get_model_complexity_info
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.2)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='pairflip')
parser.add_argument('--num_gradual', type=int, default=10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type=float, default=1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--dataset', type=str, help='luad, mnist, cifar10, or cifar100', default='luad')
parser.add_argument('--n_epoch', type=int, default=1000)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)#50
parser.add_argument('--num_workers', type=int, default=1, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=500)#500
parser.add_argument('--epoch_decay_start', type=int, default=80)  # not used in luad dataset
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--co_lambda_constant', type=bool, default=False)
parser.add_argument('--co_lambda', type=float, default=0.01)
parser.add_argument('--co_lambda_spectral', type=float, default=18.)
parser.add_argument('--co_lambda_spatial', type=float, default=9.)
parser.add_argument('--adjust_lr', type=int, default=1)
parser.add_argument('--model_type', type=str, help='[mlp,cnn,arcface,seresnet_3d,se_resnet_2d,se_resnet_2d_20, seresnet_2d_noPre, spec_resnet_2d, spec_resnets_2d, spec_resnets_2d_20]', default='cnn')
parser.add_argument('--load_model', type=str, help='load model?', default="False")
parser.add_argument('--save_model', type=str, help='save model?', default="True")
parser.add_argument('--save_result', type=str, help='save result?', default="True")
parser.add_argument('--output_path', type=str,  default='./savedmodel')
parser.add_argument('--outtxt_path', type=str, default='./outtxt_path')
parser.add_argument('--datadir', '-d', type=str, default='/home/ubuntu/Data1/zq/Dataset/LUAD/patch/')
parser.add_argument('--curvedir', type=str, default='/home/ubuntu/Data1/zq/Project/TAJ-Net/data_curve')
parser.add_argument('--shape', '-sh', default=(40,224,224), type=int)
parser.add_argument('--net', '-n', default='efficientnet-b2', type=str,help="choice effecientb0-b7")
parser.add_argument('--best_val_loss', type=float, default=999)

args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
if args.gpu is not None:
    device = torch.device('cuda:{}'.format(args.gpu))
    torch.cuda.manual_seed(args.seed)

else:
    device = torch.device('cpu')
    torch.manual_seed(args.seed)

# Hyper Parameter
batch_size = args.batch_size
learning_rate = args.lr
backbone = args.net
args.dataset = 'luad'  # luad, mnist, cifar10, cifar100
args.model_type = 'spec_resnets_2d' # se_resnet_2d,se_resnet_2d_20, seresnet_2d_noPre, spec_resnet_2d, spec_resnets_2d, spec_resnets_2d_20
args.co_lambda_spectral = 18.
args.co_lambda_spatial = 10.
args.co_lambda = 0.01
args.co_lambda_constant = True
conv1d_trained_model = 'ConvNet1d_se' # ConvNet1d or ConvNet1d_se
args.output_path = args.output_path + '_' + str(args.co_lambda_spectral)[0:2] + '_' + str(args.co_lambda_spatial)[0:2]
args.outtxt_path = './txt' + '_' + str(args.co_lambda_spectral)[0:2] + '_' + str(args.co_lambda_spatial)[0:2]
args.save_model = True
args.load_model = False
args.best_val_loss = 999
co_lambda_spectral = args.co_lambda_spectral
co_lambda_spatial = args.co_lambda_spatial

# mkdir the lost directory
if not os.path.exists(args.outtxt_path):
    print('outtxt_path does not exist!')
    os.mkdir(args.outtxt_path)
if not os.path.exists(args.output_path):
    print('output_path does not exist!')
    os.mkdir(args.output_path)
    os.mkdir(os.path.join(args.output_path, 'best'))

# load dataset
if args.dataset == 'luad':
    input_channel = args.shape[0]
    num_classes = 3
    datadir = args.datadir
    image_shape = args.shape
    data_dir = args.datadir
    train_dataset = build_dataset_3d(data_root=data_dir,  # '/home/zq/Documents/JoCoR/patch/',
                                     mode='train',
                                     image_shape=image_shape,
                                     is_argument=True
                                     )
    test_dataset = build_dataset_3d(data_root=data_dir, # '/home/zq/Documents/JoCoR/patch/',
                                 mode='val',
                                 image_shape=image_shape,
                                 is_argument=True
                                 )

if args.forget_rate is None:
    forget_rate = args.noise_rate
else:
    forget_rate = args.forget_rate

def main():
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)


    # Define models
    print('building model...')
    model = JoCoR(args, train_dataset, device=device, input_channel=input_channel, num_classes=num_classes, conv1d_trained_model=conv1d_trained_model) # ensure the model for training

    # macs, params = get_model_complexity_info(model, (2, 3, 64, 64), print_per_layer_stat=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    epoch_start = 0
    # epoch_start = 112
    train_acc1 = 0
    train_acc2 = 0
    # evaluate models with random weights
    # test_acc1, test_acc2, AFLoss1, AFLoss2, valScore1, valScore2= model.evaluate(test_loader, epoch, args.output_path)
    test_acc1, test_acc2, test_acc3 = model.evaluate(test_loader, epoch_start, args.output_path, trained_model=args.model_type, outtxt_path=args.outtxt_path)
    print(
        'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 acc=%.4f, Model2 acc=%.4f, Model3 acc=%.4f'
        % (epoch_start + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, test_acc3))

    acc_list = []
    # training
    with open(os.path.join(args.outtxt_path, "trainAcc.txt"), "w") as f_train:
        with open(os.path.join(args.outtxt_path, "evalAcc.txt"), "w") as f_eval:
            with open(os.path.join(args.outtxt_path, "bestAcc.txt"), "w") as f_best:
                for epoch in range(epoch_start + 1, args.n_epoch):
                    # train models
                    train_acc1, train_acc2, train_acc3, pure_ratio_1_list, pure_ratio_2_list = model.train(train_loader, epoch=epoch, trained_model=args.model_type, co_lambda_spectral=co_lambda_spectral, co_lambda_spatial=co_lambda_spatial, outtxt_path=args.outtxt_path)
                    f_train.write(
                        'EPOCH=%03d | train_acc1=%.4f, train_acc2=%.4f, train_acc3=%.4f' % (epoch + 1, train_acc1, train_acc2, train_acc3))
                    f_train.write('\n')
                    f_train.flush()
                    # evaluate models
                    # test_acc1, test_acc2, AFLoss1, AFLoss2, valScore1, valScore2 = model.evaluate(test_loader, epoch, args.output_path)
                    test_acc1, test_acc2, test_acc3 = model.evaluate(test_loader, epoch, args.output_path, trained_model=args.model_type, outtxt_path=args.outtxt_path)
                    # save results
                    if pure_ratio_1_list is None or len(pure_ratio_1_list) == 0:
                        print(
                            'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 acc=%.4f loss=%.4f, Model2 acc=%.4f loss=%.4f, Model3 acc=%.4f loss=%.4f' %
                            (epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, test_acc3)
                        )
                        f_eval.write(
                            'EPOCH=%03d | test_acc1=%.6f, test_acc2=%.6f, test_acc3=%.6f' %
                            (epoch + 1, test_acc1, test_acc2, test_acc3)
                        )
                        f_eval.write('\n')
                        f_eval.flush()
                    else:
                        # save results
                        mean_pure_ratio1 = sum(pure_ratio_1_list) / len(pure_ratio_1_list)
                        mean_pure_ratio2 = sum(pure_ratio_2_list) / len(pure_ratio_2_list)
                        print(
                            'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f, Model2 %.4f, Pure Ratio 1 %.4f, Pure Ratio 2 %.4f' % (
                                epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1,
                                mean_pure_ratio2))
                        f_eval.write(
                            'EPOCH=%03d | test_acc1=%.4f, test_acc2=%.4f | mean_pure_ratio1=%.4f'
                            % (epoch + 1, test_acc1, test_acc2, mean_pure_ratio1))
                        f_eval.write('\n')
                        f_eval.flush()

                    if (test_acc1 + test_acc2) / 2 > 0.85:
                        f_best.write(
                            "EPOCH=%03d | best_acc1=%.4f, best_acc2=%.4ff, best_acc3=%.4f" % (epoch + 1, test_acc1, test_acc2, test_acc3))
                        f_best.write('\n')
                        f_best.flush()
                    if epoch >= 190:
                        acc_list.extend([test_acc1, test_acc2])

    avg_acc = sum(acc_list)/len(acc_list)
    print(len(acc_list))
    print("the average acc in last 10 epochs: {}".format(str(avg_acc)))

if __name__ == '__main__':
    main()

