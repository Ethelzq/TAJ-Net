# -*- coding:utf-8 -*-
import os
import thop
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from model.cnn import MLPNet,CNN
from model.arcface import *
from model.seresnet_3d import *
from common.utils import accuracy
from algorithm.loss import loss_jocor, loss_jocor_tri
from sklearn.metrics import recall_score
import model.resnet_3d as resnet_3d

class JoCoR:
    def __init__(self, args, train_dataset, device, input_channel, num_classes, conv1d_trained_model):
        # Hyper Parameters
        self.batch_size = args.batch_size # 128

        learning_rate = args.lr
        if args.forget_rate is None:
            if args.noise_type == "asymmetric":
                forget_rate = args.noise_rate / 2
            else:
                forget_rate = args.noise_rate
        else:
            forget_rate = args.forget_rate
        # annotate the code when dataset=luad
        # self.noise_or_not = train_dataset.noise_or_not
        backbone = args.net
        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        self.alpha_plan = [learning_rate] * args.n_epoch
        self.beta1_plan = [mom1] * args.n_epoch

        for i in range(args.epoch_decay_start, args.n_epoch):
            self.alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
            self.beta1_plan[i] = mom2

        # define drop rate schedule
        self.rate_schedule = np.ones(args.n_epoch) * forget_rate
        self.rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** args.exponent, args.num_gradual)
        self.device = device
        self.num_iter_per_epoch = args.num_iter_per_epoch
        self.print_freq = args.print_freq
        self.co_lambda = args.co_lambda
        self.co_lambda_constant = args.co_lambda_constant
        self.n_epoch = args.n_epoch
        self.train_dataset = train_dataset
        self.train_model = args.model_type
        self.load_model = args.load_model

        if args.model_type == "cnn":
            self.model1 = CNN(input_channel=input_channel, n_outputs=num_classes)
            self.model2 = CNN(input_channel=input_channel, n_outputs=num_classes)
        elif args.model_type == "mlp":
            self.model1 = MLPNet()
            self.model2 = MLPNet()
        elif args.model_type == "arcface":
            if input_channel == 3:
                self.model1 = enet_arcface(out_dim_2=num_classes,backbone=backbone)
                self.model2 = enet_arcface(out_dim_2=num_classes, backbone=backbone)
            elif input_channel == 40:
                self.model1 = senet_arcface(out_dim_1=num_classes, channels=input_channel)
                self.model2 = senet_arcface(out_dim_1=num_classes, channels=input_channel)
        elif args.model_type == "se_resnet_3d": # 3d resnet model with (40,224,224)
            self.model1 = se_resnet_3d(block=resnet_3d.BasicBlock, n_classes=num_classes, channels=40, reduction=16, n_input_channels=5, block_inplanes = [8,16,32,64])
            self.model2 = se_resnet_3d(block=resnet_3d.BasicBlock, n_classes=num_classes, channels=40, reduction=16, n_input_channels=5, block_inplanes = [8,16,32,64])
        elif args.model_type == "se_resnet_2d": # 2d resnet model with (40,224,224)
            self.model1 = se_resnet_2d(out_dim_1=num_classes, channels=input_channel, reduction=16)
            self.model2 = se_resnet_2d(out_dim_1=num_classes, channels=input_channel, reduction=16)
        elif args.model_type == "se_resnet_2d_20": # 2d resnet model with (40,224,224) and half of the bands (20,224,224) as the input of the models
            self.model1 = se_resnet_2d(out_dim_1=num_classes, channels=20, reduction=8)
            self.model2 = se_resnet_2d(out_dim_1=num_classes, channels=20, reduction=8)
        elif args.model_type == "se_resnet_2d_noPre": # 2d resnet model without pre-trained model and the input size is (40,224,224)
            self.model1 = se_resnet_2d_noPre(block=BasicBlock_2d, channels=input_channel, reduction=16)
            self.model2 = se_resnet_2d_noPre(block=BasicBlock_2d, channels=input_channel, reduction=16)
        elif args.model_type == "spec_resnet_2d": # 2d resnet (40,224,224) + convnet1d (1,40)
            self.model1 = se_resnet_2d(out_dim_1=num_classes, channels=input_channel, reduction=16)
            self.model2 = ConvNet1d(channels=40, reduction=16, conv1d_trained_model=conv1d_trained_model)
        elif args.model_type == "spec_resnets_2d": # two 2d-resnet models (40,224,224) + convnet1d (1,40)
            self.model1 = se_resnet_2d(out_dim_1=num_classes, channels=input_channel, reduction=16)
            self.model2 = se_resnet_2d(out_dim_1=num_classes, channels=input_channel, reduction=16)
            self.model3 = ConvNet1d(channels=40, reduction=16, conv1d_trained_model=conv1d_trained_model)
        elif args.model_type == "spec_resnets_2d_20": # two 2d-resnet models (20,224,224) + convnet1d (1,40)
            self.model1 = se_resnet_2d(out_dim_1=num_classes, channels=20, reduction=8)
            self.model2 = se_resnet_2d(out_dim_1=num_classes, channels=20, reduction=8)
            self.model3 = ConvNet1d(channels=40, reduction=16, conv1d_trained_model=conv1d_trained_model)
        input = torch.randn(16, 40, 244, 244)
        flops, params = thop.profile(self.model1, inputs=(input,))
        print(f'model1: flops = {flops}, params = {params}')
        input = torch.randn(16, 40, 244, 244)
        flops, params = thop.profile(self.model2, inputs=(input,))
        print(f'model2: flops = {flops}, params = {params}')
        input = torch.randn(16, 40, 244, 244)
        flops, params = thop.profile(self.model3, inputs=(input,))
        print(f'model3: flops = {flops}, params = {params}')

        if self.load_model:
            print('loading past model......')
            checkpoint1 = torch.load('/data/home/qlli/zq/spec_res/specResNets/savedmodel/net_113-1-0.8666.pth')
            checkpoint2 = torch.load('/data/home/qlli/zq/spec_res/specResNets/savedmodel/net_113-2-0.8775.pth')
            checkpoint3 = torch.load('/data/home/qlli/zq/spec_res/specResNets/savedmodel/net_113-3-0.5152.pth')
            self.model1.load_state_dict(checkpoint1['model'])
            self.model2.load_state_dict(checkpoint2['model'])
            self.model3.load_state_dict(checkpoint3['model'])
        self.model1.to(device)
        # summary(self.model1, (40, 224, 224))
        self.model2.to(device)
        # if 'spec_resnet_2d' in self.train_model:
            # summary(self.model2, (1, 40))
        if 'spec_resnets' in self.train_model:# spec_resnets_2d, spec_resnets_2d_20
            self.model3.to(device)
            # summary(self.model3, (1,40))
            self.optimizer = torch.optim.Adam(
                list(self.model1.parameters()) + list(self.model2.parameters()) + list(self.model3.parameters()),
                                                      lr=learning_rate)  # , weight_decay=5e-4)
            if self.load_model:
                self.optimizer.load_state_dict(checkpoint1['optimizer'])
        else:
            self.optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()),
                                              lr=learning_rate)  # , weight_decay=5e-4)
            if self.load_model:
                self.optimizer.load_state_dict(checkpoint1['optimizer'])
        # self.optimizer = torch.optim.SGD(list(self.model1.parameters()) + list(self.model2.parameters()), lr=learning_rate, weight_decay=5e-4, momentum=0.9)
        # self.optimizer1 = torch.optim.Adam(list(self.model1.parameters()), lr=learning_rate)
        # self.optimizer2 = torch.optim.Adam(list(self.model2.parameters()), lr=learning_rate)
        self.loss_fn = loss_jocor
        self.loss_fn_tri = loss_jocor_tri
        self.adjust_lr = args.adjust_lr
        self.best_val_loss = args.best_val_loss

    # Evaluate the Model
    def evaluate(self, test_loader, epoch, output_path, trained_model, outtxt_path):
        print('Evaluating ...')
        self.model1.eval()  # Change model to 'eval' mode.
        self.model2.eval()  # Change model to 'eval' mode
        if 'spec_resnets' in trained_model: # spec_resnets_2d, spec_resnets_2d_20
            self.model3.eval()
        criterion1 = ArcFaceLoss()
        criterion2 = nn.CrossEntropyLoss()
        correct1 = 0
        total1 = 0
        for images, labels, _, spec_mean in test_loader:
            if '2d_20' in trained_model: # se_resnet_2d_20, spec_resnets_2d_20
                images1 = images[:,::2,:,:]
            else:
                images1 = images
            images = Variable(images1).to(self.device)
            logits1, spec_mean_batch1 = self.model1(images)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels).sum()
            labels_gpu = labels.cuda()
            AFLoss1 = criterion1(logits1, labels_gpu)

        correct2 = 0
        total2 = 0
        for images, labels, _, spec_mean in test_loader:
            if '2d_20' in trained_model: # se_resnet_2d_20, spec_resnets_2d_20
                images2 = images[:, 1::2, :, :]
            elif trained_model == 'spec_resnet_2d':
                # images2 = images[:,:,111,111]
                # images2 = images2.unsqueeze(1)
                images2 = images
            else:
                images2 = images
            images = Variable(images2).to(self.device)
            logits2, spec_mean_batch2 = self.model2(images)
            outputs2 = F.softmax(logits2, dim=1)
            _, pred2 = torch.max(outputs2.data, 1)
            total2 += labels.size(0)
            correct2 += (pred2.cpu() == labels).sum()
            labels_gpu = labels.cuda()
            if 'spec_resnet_2d' in trained_model:
                AFLoss2 = criterion2(logits2, labels_gpu)
            else:
                AFLoss2 = criterion1(logits2, labels_gpu)

        if 'spec_resnets' in trained_model: # spec_resnets_2d, spec_resnets_2d_20
            correct3 = 0
            total3 = 0
            for images, labels, _, spec_mean in test_loader:
                # images3 = images[:, :, 111, 111]
                # images3 = images3.unsqueeze(1)
                images3 = images
                images = Variable(images3).to(self.device)
                logits3, spec_mean_batch3 = self.model3(images)
                outputs3 = F.softmax(logits3, dim=1)
                _, pred3 = torch.max(outputs3.data, 1)
                total3 += labels.size(0)
                correct3 += (pred3.cpu() == labels).sum()
                labels_gpu = labels.cuda()
                AFLoss3 = criterion2(logits3, labels_gpu)
        else:
            AFLoss3 = AFLoss2

        test_acc1 = float(correct1) / float(total1)
        test_acc2 = float(correct2) / float(total2)
        outputs_cpu1 = pred1.data.cpu().numpy()
        outputs_cpu2 = pred2.data.cpu().numpy()
        valScore1 = recall_score(outputs_cpu1, labels, average='macro')
        valScore2 = recall_score(outputs_cpu2, labels, average='macro')
        if 'spec_resnets' in trained_model: # spec_resnets_2d, spec_resnets_2d_20
            test_acc3 = float(correct3) / float(total3)
            outputs_cpu3 = pred3.data.cpu().numpy()
            valScore3 = recall_score(outputs_cpu3, labels, average='macro')
        else:
            test_acc3 = test_acc2
            valScore3 = valScore2

        print("saving model...")
        print(
            'Epoch [%d/%d], AFLoss1=%.4f, AFLoss2=%.4f, AFLoss3=%.4f, valScore1=%.4f, valScore2=%.4f, valScore3=%.4f' % (
            epoch + 1, self.n_epoch, AFLoss1.item(), AFLoss2.item(), AFLoss3.item(), valScore1, valScore2, valScore3)
        )
        with open(os.path.join(outtxt_path, "file_AF.txt"), "a+") as file_AF:# a+ add
            file_AF.write(
                'EPOCH=%03d | test_acc1=%.6f, test_acc2=%.6f, AFLoss1=%.6f, AFLoss2=%.6f, AFLoss3=%.6f, valScore1=%.6f, valScore2=%.6f, valScore3=%.6f' %
                (epoch + 1, test_acc1, test_acc2, AFLoss1, AFLoss2, AFLoss3, valScore1, valScore2, valScore3)
            )
            file_AF.write('\n')
            file_AF.flush()
        torch.save({'model': self.model1.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch},
                   '%s/net_%03d-1-%.4f.pth' % (output_path, epoch + 1, test_acc1))
        torch.save({'model': self.model2.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch},
                   '%s/net_%03d-2-%.4f.pth' % (output_path, epoch + 1, test_acc2))
        if 'spec_resnets' in trained_model:  # spec_resnets_2d, spec_resnets_2d_20
            torch.save({'model': self.model3.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch},
                       '%s/net_%03d-3-%.4f.pth' % (output_path, epoch + 1, test_acc3))
        # save the best model
        if epoch == 0:
            if AFLoss1 < AFLoss2:
                torch.save(self.model1.state_dict(),
                           '%s/%s/net_%03d-%.4f.pth' % (output_path, "best", epoch + 1, AFLoss1))
                self.best_val_loss = AFLoss1
            else:
                torch.save(self.model1.state_dict(),
                           '%s/%s/net_%03d-%.4f.pth' % (output_path, "best", epoch + 1, AFLoss2))
                self.best_val_loss = AFLoss2
        else:
            if (AFLoss1 + AFLoss2) / 2 < 2.5:
                if AFLoss1 < AFLoss2:
                    if AFLoss1 < self.best_val_loss:
                        print("saving the best model...")
                        torch.save(self.model1.state_dict(),
                                   '%s/%s/net_%03d-%.4f.pth' % (output_path, "best", epoch + 1, AFLoss1))
                        self.best_val_loss = AFLoss1
                else:
                    if AFLoss2 < self.best_val_loss:
                        print("saving the best model...")
                        torch.save(self.model2.state_dict(),
                                   '%s/%s/net_%03d-%.4f.pth' % (output_path, "best", epoch + 1, AFLoss2))
                        self.best_val_loss = AFLoss2
        return test_acc1, test_acc2, test_acc3

    # Train the Model
    def train(self, train_loader, epoch, trained_model, co_lambda_spectral, co_lambda_spatial, outtxt_path):
        print('Training ...')
        self.model1.train()  # Change model to 'train' mode.
        self.model2.train()  # Change model to 'train' mode
        if 'spec_resnets' in trained_model: # spec_resnets_2d, spec_resnets_2d_20
            self.model3.train()

        criterion1 = ArcFaceLoss()
        criterion2 = nn.CrossEntropyLoss()

        if self.adjust_lr == 1:
            self.adjust_learning_rate(self.optimizer, epoch)
            # self.adjust_learning_rate(self.optimizer1, epoch)
            # self.adjust_learning_rate(self.optimizer2, epoch)

        train_total = 0
        train_correct = 0
        train_total2 = 0
        train_correct2 = 0
        train_total3 = 0
        train_correct3 = 0
        pred1 = []
        pred2 = []
        pred3 = []
        pure_ratio_1_list = []
        pure_ratio_2_list = []

        with open(os.path.join(outtxt_path, "trainAccLoss.txt"), "a") as f_train_loss:
            for i, (images, labels, indexes, spec_mean) in enumerate(train_loader):
                ind = indexes.cpu().numpy().transpose()
                if i > self.num_iter_per_epoch:
                    break
                if trained_model == 'se_resnet_2d_20':
                    images1 = images[:, ::2, :, :]
                    images1 = Variable(images1).to(self.device)
                    images2 = images[:, 1::2, :, :]
                    images2 = Variable(images2).to(self.device)
                    labels = Variable(labels).to(self.device)
                elif trained_model == 'spec_resnet_2d':
                    images1 = Variable(images).to(self.device)
                    # images2 = images[:, :, 111, 111]
                    # images2 = images2.unsqueeze(1)
                    images2 = images
                    images2 = Variable(images2, requires_grad=True).to(self.device)
                    # if True in torch.isnan(images2):
                    #     print('there is nan in images2!')
                    labels = Variable(labels).to(self.device)
                elif trained_model == 'spec_resnets_2d':
                    images1 = Variable(images).to(self.device)
                    images2 = Variable(images).to(self.device)
                    # images3 = images[:, :, 111, 111]
                    # images3 = images3.unsqueeze(1)
                    images3 = images
                    images3 = Variable(images3).to(self.device)
                    labels = Variable(labels).to(self.device)
                elif trained_model == 'spec_resnets_2d_20':
                    images1 = images[:, ::2, :, :]
                    images1 = Variable(images1).to(self.device)
                    images2 = images[:, 1::2, :, :]
                    images2 = Variable(images2).to(self.device)
                    # images3 = images[:, :, 111, 111]
                    # images3 = images3.unsqueeze(1)
                    images3 = images
                    images3 = Variable(images3).to(self.device)
                    labels = Variable(labels).to(self.device)
                else:
                    images1 = Variable(images).to(self.device)
                    images2 = Variable(images).to(self.device)
                    labels = Variable(labels).to(self.device)

                # Forward + Backward + Optimize

                logits1, spec_mean_batch1 = self.model1(images1)
                # print('logits1: = ', logits1)
                # print('spec_mean_batch1 = ', spec_mean_batch1)
                outputs1 = F.softmax(logits1, dim=1)
                indexes, pred1 = torch.max(outputs1.data, 1)
                prec1 = accuracy(logits1, labels, topk=(1,)) / 100.
                train_total += 1
                train_correct += prec1

                logits2, spec_mean_batch2 = self.model2(images2)
                # print('logits2: = ', logits2)
                # print('spec_mean_batch2 = ', spec_mean_batch2)
                outputs2 = F.softmax(logits2, dim=1)
                indexes, pred2 = torch.max(outputs2.data, 1)
                prec2 = accuracy(logits2, labels, topk=(1,)) / 100.
                train_total2 += 1
                train_correct2 += prec2

                arcface_loss1 = criterion1(logits1, labels) # arcface loss
                co_lambda1 = spec_mean_batch1 / co_lambda_spatial * 1.5
                # if 'spec_resnet_2d' in trained_model:
                if 'spec_resnet_2d' in trained_model == 'spec_resnet_2d' or trained_model == 'spec_resnet_2d_20':
                    arcface_loss2 = criterion2(logits2, labels)
                    co_lambda2 = spec_mean_batch2 / co_lambda_spectral
                else:
                    arcface_loss2 = criterion1(logits2, labels)
                    co_lambda2 = spec_mean_batch2 / co_lambda_spatial

                # print('co_lambda1=', co_lambda1)
                # print('co_lambda2=', co_lambda2)

                # loss_1, loss_2, pure_ratio_1, pure_ratio_2 = self.loss_fn(logits1, logits2, labels, arcface_loss1, arcface_loss2, self.rate_schedule[epoch], ind, self.co_lambda)
                if 'spec_resnets' in trained_model: # spec_resnets_2d, spec_resnets_2d_20
                    # print('there are three models!')
                    logits3, spec_mean_batch3 = self.model3(images3)
                    # print('logits3: = ', logits3)
                    # print('spec_mean_batch3 = ', spec_mean_batch3)
                    outputs3 = F.softmax(logits3, dim=1)
                    indexes, pred3 = torch.max(outputs3.data, 1)
                    prec3 = accuracy(logits3, labels, topk=(1,)) / 100.
                    train_total3 += 1
                    train_correct3 += prec3
                    arcface_loss3 = criterion2(logits3, labels) # CE loss
                    co_lambda3 = spec_mean_batch3/co_lambda_spectral
                    # print('co_lambda3=', co_lambda3)
                    loss_1, loss_2, loss_3, pure_ratio_1, pure_ratio_2, pure_ratio_3 = self.loss_fn_tri(logits1, logits2, logits3,
                                                                                  arcface_loss1,
                                                                                  arcface_loss2, arcface_loss3,
                                                                                  self.rate_schedule[epoch],
                                                                                  ind, co_lambda1, co_lambda2,
                                                                                  co_lambda3, self.co_lambda, self.co_lambda_constant)
                else:
                    prec3 = prec2
                    arcface_loss3 = arcface_loss2
                    loss_1, loss_2, pure_ratio_1, pure_ratio_2 = self.loss_fn(logits1, logits2, arcface_loss1,
                                                                              arcface_loss2, self.rate_schedule[epoch],
                                                                              ind, co_lambda1, co_lambda2, self.co_lambda, self.co_lambda_constant)

                self.optimizer.zero_grad()
                loss_1.backward()
                self.optimizer.step()
                # self.optimizer1.zero_grad()
                # loss_1.backward()
                # self.optimizer1.step()
                # self.optimizer2.step()
                # self.optimizer2.zero_grad()
                # loss_2.backward()
                # self.optimizer2.step()
                # self.optimizer1.step()
                pure_ratio_1_list.append(pure_ratio_1)
                pure_ratio_2_list.append(pure_ratio_2)

                if (i + 1) % self.print_freq == 0:
                    print(
                        'Epoch [%d/%d], Iter [%d/%d], Training Accuracy1: %.4f, Training Accuracy2: %.4f, Training Accuracy3: %.4f, Loss: %.4f, AFLoss1: %.4f, AFLoss2: %.4f, AFLoss3: %.4f, Pure Ratio1 %.4f'
                        % (
                        epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, prec2, prec3,
                        loss_1.data.item(), arcface_loss1, arcface_loss2, arcface_loss3,
                        sum(pure_ratio_1_list) / len(pure_ratio_1_list)))
                    f_train_loss.write(
                        "Epoch [%04d/%04d] | Iter [%03d/%03d] | Training Accuracy1=%.4f, Training Accuracy2=%.4f, Training Accuracy3=%.4f, Loss=%.4f, AFLoss1=%.4f, AFLoss2=%.4f, AFLoss3=%.4f, Pure Ratio1=%.4f" % (
                        epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, prec2, prec3,
                        loss_1.data.item(), arcface_loss1, arcface_loss2, arcface_loss3,
                        sum(pure_ratio_1_list) / len(pure_ratio_1_list)))
                    f_train_loss.write('\n')
                    f_train_loss.flush()

        train_acc1 = float(train_correct) / float(train_total)
        train_acc2 = float(train_correct2) / float(train_total2)
        outputs_cpu1 = pred1.data.data.cpu().numpy()
        outputs_cpu2 = pred2.data.data.cpu().numpy()
        labels_cpu = labels.data.data.cpu().numpy()
        trainScore1 = recall_score(outputs_cpu1, labels_cpu, average='macro')
        trainScore2 = recall_score(outputs_cpu2, labels_cpu, average='macro')
        if 'spec_resnets' in trained_model: # spec_resnets_2d, spec_resnets_2d_20
            train_acc3 = float(train_correct3) / float(train_total3)
            outputs_cpu3 = pred3.data.data.cpu().numpy()
            trainScore3 = recall_score(outputs_cpu3, labels_cpu, average='macro')
        else:
            train_acc3 = train_acc2
            trainScore3 = trainScore2

        print('tranScore1=%.4f, trainScore2=%.4f, trainScore3=%.4f'% (trainScore1, trainScore2, trainScore3))
        return train_acc1, train_acc2, train_acc3, pure_ratio_1_list, pure_ratio_2_list

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1
