import os, argparse, time
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import time
import torch
import torch.nn as nn
from torchvision import transforms
import random
import torch.backends.cudnn as cudnn
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from models.imgmodal_pcqa import ViSam_PCQA #need to change the name to ViSam_PCQA
from utils.MultimodalDataset import MMDataset
from utils.loss import L2RankLoss
import wandb
import datetime


def set_rand_seed(seed=1998):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # fix the random seed


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)

    return y_output_logistic


# this function is used to put pyperparameters and the path dir
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument("--gpu", help="GPU device id to use [0]", default=0, type=int)
    parser.add_argument(
        "--num_epochs", help="Maximum number of training epochs.", default=30, type=int
    )
    parser.add_argument("--batch_size", help="Batch size.", default=8, type=int)
    parser.add_argument(
        "--learning_rate", default=0.001, type=float, help="learning rate in training"
    )
    parser.add_argument("--decay_rate", type=float, default=1e-4, help="decay rate")
    parser.add_argument("--model", default="", type=str)
    parser.add_argument(
        "--data_dir_texture_img", default="", type=str, help="path to the images"
    )
    parser.add_argument(
        "--data_dir_depth_img", default="", type=str, help="path to the depth images"
    )
    parser.add_argument(
        "--data_dir_normal_img", default="", type=str, help="path to the normal images"
    )
    parser.add_argument(
        "--data_dir_vs_img",
        default="",
        type=str,
        help="path to the visual saliency images",
    )

    parser.add_argument(
        "--img_length_read", default=6, type=int, help="number of the using images"
    )
    parser.add_argument("--loss", default="l2rank", type=str)
    parser.add_argument("--database", default="SJTU", type=str)
    parser.add_argument(
        "--k_fold_num",
        default=9,
        type=int,
        help="9 for the SJTU-PCQA, 5 for the WPC, 4 for the WPC2.0",
    )
    parser.add_argument(
        "--use_classificaiton",
        default=1,
        help="if use classification, 1 or 0",
        type=int,
    )
    parser.add_argument("--use_local", help="", default=1, type=int)
    parser.add_argument("--img_inplanes", help="", default=2048, type=int)
    parser.add_argument("--pc_inplanes", help="", default=1024, type=int)
    parser.add_argument("--cma_planes", help="", default=1024, type=int)
    parser.add_argument(
        "--modality", help=" only single modality", default="img", type=str
    )
    parser.add_argument(
        "--method_label",
        help="Description of the method of the model",
        default="",
        type=str,
    )
    args = parser.parse_args()
    return args


def load_vs_model(
    vs_flag,
):  # vs_flag = 0 # 0 for TranSalNet_Dense, 1 for TranSalNet_Res
    if vs_flag:
        from models.TranSalNet_Res import TranSalNet

        vs_model = TranSalNet()
        vs_model.load_state_dict(torch.load("pretrained_models/TranSalNet_Res.pth"))
    else:
        from models.TranSalNet_Dense import TranSalNet

        vs_model = TranSalNet()
        vs_model.load_state_dict(torch.load("pretrained_models/TranSalNet_Dense.pth"))

    vs_model = vs_model.to(device)
    vs_model.eval()
    return vs_model


if __name__ == "__main__":
    print(
        "*************************************************************************************************************************"
    )

    args = parse_args()
    set_rand_seed()
    gpu = args.gpu
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    database = args.database
    modality = args.modality
    img_length_read = args.img_length_read
    data_dir_texture_img = args.data_dir_texture_img
    data_dir_depth = args.data_dir_depth_img
    data_dir_normal = args.data_dir_normal_img
    data_dir_vs = args.data_dir_vs_img
    best_all = np.zeros([args.k_fold_num, 5])
    # Get the current date and time
    current_datetime = datetime.datetime.now()
    # Convert the datetime object to a string format
    date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M")
    wandb_name = f"dataset_{args.database}_{args.k_fold_num}_Fold_{date_time_string}_Epoch_{args.num_epochs}_BS_{args.batch_size}_Loss_{args.loss}_Classify_{args.use_classificaiton}_Local_{args.use_local}_Method_{args.method_label}_Modality_{args.modality}"
    wandb.login(relogin=True, key="your own key")
    wandb.init(
        project="", config=args, name=wandb_name, sync_tensorboard=True
    )
    print("wandb name: ", wandb_name)
    mean_total_loss = np.zeros(num_epochs)  # {k: 0 for k in range(num_epochs)}
    mean_regression_loss = np.zeros(num_epochs)
    mean_classification_loss = np.zeros(num_epochs)

    for k_fold_id in range(
        1, args.k_fold_num + 1
    ):  # not include the upper value in range and I want to start from 1  in range(1,args.k_fold_num + 1)
        print("The current k_fold_id is " + str(k_fold_id))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        if database == "SJTU":
            train_filename_list = (
                "csvfiles/sjtu_data_info/train_" + str(k_fold_id) + ".csv"
            )
            test_filename_list = (
                "csvfiles/sjtu_data_info/test_" + str(k_fold_id) + ".csv"
            )
        elif database == "WPC":
            train_filename_list = (
                "csvfiles/wpc_data_info/train_" + str(k_fold_id) + ".csv"
            )
            test_filename_list = (
                "csvfiles/wpc_data_info/test_" + str(k_fold_id) + ".csv"
            )
        elif database == "BASICS":
            train_filename_list = (
                "csvfiles/basics_data_info/train_" + str(k_fold_id) + ".csv"
            )
            test_filename_list = (
                "csvfiles/basics_data_info/test_" + str(k_fold_id) + ".csv"
            )

        # do some pre-precessing for the image
        if True:
            transformations_train = transforms.Compose(
                [
                    # transforms.RandomCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            transformations_test = transforms.Compose(
                [
                    # transforms.CenterCrop(224),  # xm: wht test is centercrop?
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            transformations_train = transforms.Compose(
                [
                    transforms.RandomCrop(480),
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            transformations_test = transforms.Compose(
                [
                    transforms.CenterCrop(480),
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        print("Trainging set: " + train_filename_list)
        if args.database == "SJTU":
            _num_class = 7
        elif args.database == "WPC":
            _num_class = 5
        elif args.database == "BASICS":
            _num_class = 4
        else:
            raise
        print("num class: ", _num_class)

        if args.model == "ViSam_PCQA": # Change the name to ViSam-PCQA
            model = ViSam_PCQA(
                num_classes=_num_class,
                args=args,
            )
            model = model.to(device)  # moves the model to the device.
            print("Using model: ViSam-PCQA")

        if args.loss == "l2rank":
            criterion = L2RankLoss(args).to(device)  # is there necessary?
            print("Using l2rank loss")

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate
        )
        print("Using Adam optimizer, initial learning rate: " + str(args.learning_rate))
        print("Dataset: " + str(args.database))
        print("Number of epoches: " + str(args.num_epochs))
        print("Number of folds: " + str(args.k_fold_num))

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.9)

        print("Ready to train network")
        print(
            "*************************************************************************************************************************"
        )
        best_test_criterion = -1  # SROCC min
        best = np.zeros(5)

        train_dataset = MMDataset(
            data_dir_texture=data_dir_texture_img,
            data_dir_depth=data_dir_depth,
            data_dir_normal=data_dir_normal,
            data_dir_vs=data_dir_vs,
            datainfo_path=train_filename_list,
            transform=transformations_train,
        )
        test_dataset = MMDataset(
            data_dir_texture=data_dir_texture_img,
            data_dir_depth=data_dir_depth,
            data_dir_normal=data_dir_normal,
            data_dir_vs=data_dir_vs,
            datainfo_path=test_filename_list,
            transform=transformations_test,
            is_train=False,
        )

        for epoch in range(num_epochs):
            # begin training, during each epoch, the crops and patches are randomly selected for the training set and fixed for the testing set
            # if you want to change the number of images or projections, load the parameters here 'img_length_read = img_length_read, patch_length_read = patch_length_read'
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=12,
                pin_memory=True,
            )
            test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset, batch_size=1, shuffle=False, num_workers=12
            )
            # xm: why test shuffle=false(pair)?
            n_train = len(train_dataset)
            n_test = len(test_dataset)
            # for each epoch, train the model, and get the predicted mos and distortion types
            model.train()
            start = time.time()
            batch_losses = []
            # batch_losses_each_disp = [] #xm: what does this is used for?
            batch_losses_0 = []
            batch_losses_1 = []

            x_output = np.zeros(n_train)
            x_test = np.zeros(n_train)
            for i, data in enumerate(train_loader):
                # traning session normaliza the mos into [0,10] for different datsets
                (
                    imgs,
                    depths,
                    normals,
                    vs,
                    mos,
                    dts,
                ) = data
                if args.database == "SJTU":
                    mos = mos
                elif args.database == "WPC":
                    mos = mos / 10
                elif args.database == "BASICS":
                    mos = mos
                else:
                    raise
                if True:
                    imgs = imgs.to(device)
                    depths = depths.to(device)
                    normals = normals.to(device)
                    vs = vs.to(device)
                    mos = mos[:, np.newaxis]
                    mos = mos.to(device)
                    dts = dts[:, np.newaxis]
                    dts = dts.to(device)
                    mos_output, dts_output = model(imgs, depths, normals, vs)
                    loss_total, loss_regression, loss_classification = criterion(
                        mos_output, dts_output, mos, dts
                    )
                    batch_losses.append(loss_total.item())
                    batch_losses_0.append(loss_regression.item())
                    if args.use_classificaiton:
                        batch_losses_1.append(loss_classification.item())
                    else:
                        batch_losses_1.append(0)

                    optimizer.zero_grad()  # clear gradients for next train
                    torch.autograd.backward(loss_total)
                    optimizer.step()
                else:
                    imgs = imgs.to(device)
                    depths = depths.to(device)
                    normals = normals.to(device)
                    pc_texture = torch.Tensor(pc_texture.float(), requires_grad=True)
                    pc_texture = pc_texture.to(device)
                    pc_normal = torch.Tensor(pc_normal.float())
                    pc_normal = pc_normal.to(device)
                    pc_position = torch.Tensor(pc_position.float())
                    pc_position = pc_position.to(device)
                    mos = mos[:, np.newaxis]
                    mos = mos.to(device)
                    dts = dts[:, np.newaxis]
                    dts = dts.to(device)
                    mos_output, dts_output = model(
                        imgs, depths, normals, pc_texture, pc_normal, pc_position
                    )

                    # compute loss
                    loss_total, loss_regression, loss_classification = criterion(
                        mos_output, dts_output, mos, dts
                    )
                    batch_losses.append(loss_total.item())
                    batch_losses_0.append(loss_regression.item())
                    if args.use_classificaiton:
                        batch_losses_1.append(loss_classification.item())
                    else:
                        batch_losses_1.append(0)

                    optimizer.zero_grad()  # clear gradients for next train
                    torch.autograd.backward(loss_total)
                    print(pc_texture.grad.shape)  # (B, 1024, 6)->(B,1024)
                    # visualize the gradient,based on pc_texture.grad

            avg_loss = sum(batch_losses) / (len(train_dataset) // batch_size)
            avg_loss_0 = sum(batch_losses_0) / (len(train_dataset) // batch_size)
            avg_loss_1 = sum(batch_losses_1) / (len(train_dataset) // batch_size)
            print(
                "Epoch %d averaged training loss: %.4f\t%.4f\t%.4f"
                % (epoch + 1, avg_loss, avg_loss_0, avg_loss_1)
            )
            wandb_dict = dict()
            wandb_dict[f"Loss/Train_Total_Fold_{k_fold_id}"] = avg_loss
            wandb_dict[f"Loss/Train_Regression_Fold_{k_fold_id}"] = avg_loss_0
            wandb_dict[f"Loss/Train_Classification_Fold_{k_fold_id}"] = avg_loss_1
            wandb_dict["epoch"] = epoch
            mean_total_loss[epoch] += avg_loss
            mean_regression_loss[epoch] += avg_loss_0
            mean_classification_loss[epoch] += avg_loss_1
            wandb.log(wandb_dict)

            scheduler.step()  
            lr_current = scheduler.get_last_lr()
            print("The current learning rate is {:.06f}".format(lr_current[0]))
            end = time.time()
            print(
                "Epoch %d training time cost: %.4f seconds" % (epoch + 1, end - start)
            )

            # Test
            model.eval()
            y_output = np.zeros(n_test)
            y_test = np.zeros(n_test)
            # distortion classification accuracy
            distortion_output = np.zeros(n_test)
            distortion_test = np.zeros(n_test)

            with torch.no_grad():
                for i, (
                    tex_imgs,
                    dep_imgs,
                    nor_imgs,
                    vs_imgs,
                    mos,
                    dis,
                ) in enumerate(test_loader):
                    if args.database == "SJTU":
                        mos = mos
                    elif args.database == "WPC":
                        mos = mos / 10
                    elif args.database == "BASICS":
                        mos = mos
                    else:
                        raise
                    tex_imgs = tex_imgs.to(device)
                    dep_imgs = dep_imgs.to(device)
                    nor_imgs = nor_imgs.to(device)
                    vs_imgs = vs_imgs.to(device)

                    y_test[i] = mos.item()
                    distortion_test[i] = dis.item()
                    mos_output, dts_output = model(
                        tex_imgs, dep_imgs, nor_imgs, vs_imgs
                    )
                    _, distortion_predicted = torch.max(dts_output.data, dim=1)
                    y_output[i] = mos_output.item()
                    distortion_output[i] = distortion_predicted.item()

                y_output_logistic = fit_function(y_test, y_output)
                test_PLCC = stats.pearsonr(y_output_logistic, y_test)[0]
                test_SROCC = stats.spearmanr(y_output, y_test)[0]
                test_RMSE = np.sqrt(((y_output_logistic - y_test) ** 2).mean())
                test_KROCC = scipy.stats.kendalltau(y_output, y_test)[0]
                test_DCacc = sum(np.equal(distortion_test, distortion_output)) / len(
                    distortion_test
                )
                print(
                    "Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}, ACC={:.4f}".format(
                        test_SROCC, test_KROCC, test_PLCC, test_RMSE, test_DCacc
                    )
                )

                if test_SROCC > best_test_criterion:
                    print("Update best model using best_val_criterion ")
                    torch.save(
                        model.state_dict(),
                        "ckpts/"
                        + database
                        + "_"
                        + str(k_fold_id)
                        + "Dis_"
                        + str(args.use_classificaiton)
                        + "Local_"
                        + str(args.use_local)
                        + "Epoch_"
                        + str(args.num_epochs)
                        + "BS_"
                        + str(args.batch_size)
                        + "Loss_"
                        + args.loss
                        + "Method_"
                        + str(args.method_label)
                        + "Modality_"
                        + str(args.modality)
                        + "_best_model.pth",  # xm: save the model
                    )
                    best[0:5] = [
                        test_SROCC,
                        test_KROCC,
                        test_PLCC,
                        test_RMSE,
                        test_DCacc,
                    ]
                    best_test_criterion = test_SROCC  # update best val SROCC
                    print(
                        "Update the best Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}, ACC={:.4f}".format(
                            test_SROCC, test_KROCC, test_PLCC, test_RMSE, test_DCacc
                        )
                    )

                    wandb.log(
                        {
                            "Performance/Best_SROCC": test_SROCC,
                            "Performance/Best_PLCC": test_PLCC,
                            "Performance/Best_Distortion_ACC": test_DCacc,
                        }
                    )

        print(database)
        best_all[k_fold_id - 1, :] = best
        print(
            "The best Val results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}, ACC={:.4f}".format(
                best[0], best[1], best[2], best[3], best[4]
            )
        )
        print(
            "*************************************************************************************************************************"
        )

    # average score
    best_mean = np.mean(best_all, 0)
    # print the mean loss for k fold
    if (
        len(mean_total_loss)
        == len(mean_regression_loss)
        == len(mean_classification_loss)
    ):
        print("There are in total {} epochs".format(len(mean_total_loss)))
    else:
        raise
    for epoch in range(0, len(mean_total_loss)):
        print(
            f"Epoch {epoch} averaged total training loss: {mean_total_loss[epoch]/(args.k_fold_num)}"
        )
        mean_total_loss_ = mean_total_loss[epoch] / (args.k_fold_num)
        mean_regression_loss_ = mean_regression_loss[epoch] / (args.k_fold_num)
        mean_classification_loss_ = mean_classification_loss[epoch] / (args.k_fold_num)
        wandb.log(
            {
                "Loss/Train_Total_Mean": mean_total_loss_,
                "Loss/Train_Regression_Mean": mean_regression_loss_,
                "Loss/Train_Classification_Mean": mean_classification_loss_,
            }
        )

    print(
        "*************************************************************************************************************************"
    )
    print(
        "The mean val results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}, ACC={:.4f}".format(
            best_mean[0], best_mean[1], best_mean[2], best_mean[3], best_mean[4]
        )
    )
    print(
        "*************************************************************************************************************************"
    )


