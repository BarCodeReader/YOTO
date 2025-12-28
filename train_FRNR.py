import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random
import importlib
from dataloader import prepare_dataloader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from options import get_option
from scipy.stats import spearmanr, pearsonr
from utils.process import five_point_crop, random_crop


def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def set_logging(config):
    filename = os.path.join(config.output_path, config.model_name, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode="w",
        format="[%(asctime)s %(levelname)-8s] %(message)s",
        datefmt="%Y%m%d %H:%M:%S",
    )


def train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader, mode="mix"):
    losses = []
    net.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []

    # adjust encoder's learing rate
    cur_lr = scheduler.get_last_lr()  # list of [lr_enc, lr_body]

    for data in tqdm(train_loader):

        optimizer.param_groups[0]["lr"] = cur_lr[1] * 0.1  # enc_lr = 0.1 * body_lr

        x_r = data["r_img_org"].cuda()
        x_d = data["d_img_org"].cuda()
        labels = data["score"]
        labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()

        if mode == "mix":
            if random.random() < 0.5:
                pred_d = net(x_d, x_r, mode="FR")
            else:
                pred_d = net(x_d, x_d, mode="NR")
        elif mode == "FR":
            pred_d = net(x_d, x_r)
        elif mode == "NR":
            pred_d = net(x_d, x_d)

        optimizer.zero_grad()
        loss = criterion(torch.squeeze(pred_d), labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)

    # step scheduler every epoch
    scheduler.step()
    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    ret_loss = np.mean(losses)
    msg = "train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}".format(
        epoch + 1, ret_loss, rho_s, rho_p
    )
    logging.info(msg)
    print(msg)

    return ret_loss, rho_s, rho_p


def eval_epoch_FR(config, epoch, net, criterion, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for data in tqdm(test_loader):
            pred = 0
            if config.num_avg_val != 5:
                for i in range(config.num_avg_val):
                    x_d = data["d_img_org"].cuda()
                    x_r = data["r_img_org"].cuda()
                    labels = data["score"]
                    labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                    x_d = random_crop(d_img=x_d, config=config)
                    x_r = random_crop(d_img=x_r, config=config)
                    pred += net(x_d, x_r, mode="FR")
            else:
                for i in range(config.num_avg_val):
                    x_d = data["d_img_org"].cuda()
                    x_r = data["r_img_org"].cuda()
                    labels = data["score"]
                    labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                    x_d = five_point_crop(i, d_img=x_d, config=config)
                    x_r = five_point_crop(i, d_img=x_r, config=config)
                    pred += net(x_d, x_r, mode="FR")

            pred /= config.num_avg_val

            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        msg = "Test epoch FR:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}".format(
            epoch + 1, np.mean(losses), rho_s, rho_p
        )
        logging.info(msg)
        print(msg)
        return np.mean(losses), rho_s, rho_p


def eval_epoch_NR(config, epoch, net, criterion, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for data in tqdm(test_loader):
            pred = 0
            if config.num_avg_val != 5:
                for i in range(config.num_avg_val):
                    x_d = data["d_img_org"].cuda()
                    x_r = data["r_img_org"].cuda()
                    labels = data["score"]
                    labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                    x_d = random_crop(d_img=x_d, config=config)
                    pred += net(x_d, x_d, mode="NR")
            else:
                for i in range(config.num_avg_val):
                    x_d = data["d_img_org"].cuda()
                    x_r = data["r_img_org"].cuda()
                    labels = data["score"]
                    labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                    x_d = five_point_crop(i, d_img=x_d, config=config)
                    pred += net(x_d, x_d, mode="NR")

            pred /= config.num_avg_val

            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        msg = "Test epoch NR:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}".format(
            epoch + 1, np.mean(losses), rho_s, rho_p
        )
        logging.info(msg)
        print(msg)
        return np.mean(losses), rho_s, rho_p


if __name__ == "__main__":
    config = get_option()
    print("=======training mode: {}".format(config.training_mode))

    cpu_num = 1
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU)

    # random seed
    seed = random.randint(0, 9999) if config.random_seed else config.seed
    # config.seed = seed
    print("---random seed: {}, seed:{}".format(config.random_seed, seed))

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path, exist_ok=True)

    model_file = os.path.join(config.output_path, config.model_name)
    if not os.path.exists(model_file):
        os.makedirs(model_file, exist_ok=True)
        os.system("cp -r ./models/*.py {}/models".format(model_file))
        os.system("cp {} {}".format(config.save_sh, model_file))

    set_logging(config)
    logging.info(config)

    writer = SummaryWriter(model_file)
    logging.info("seed used for this training {}".format(seed))

    # dataloader
    train_loader, val_loader = prepare_dataloader(config)

    # model
    module = importlib.import_module("models.{}".format(config.network.lower()))
    net = module.Net(config, device="cuda")
    net = net.cuda()

    # calculate net param
    num_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("total params: {}M".format(num_param / 1e6))

    # loss function
    criterion = torch.nn.MSELoss()

    # gather parameters
    enc, body = [], []
    for name, param in net.named_parameters():
        if "encoder" in name:
            enc.append(param)
        else:
            body.append(param)
    assert enc != [], "encoder is empty"
    optimizer = torch.optim.Adam(
        [{"params": enc}, {"params": body}],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.T_max, eta_min=config.eta_min, verbose=True
    )

    # train & validation
    losses, scores = [], []
    best_sroccf = 0
    best_plccf = 0
    best_sroccn = 0
    best_plccn = 0
    for epoch in range(0, config.n_epoch):
        start_time = time.time()
        logging.info("Running training epoch {}".format(epoch + 1))
        loss_val, rho_s, rho_p = train_epoch(
            epoch,
            net,
            criterion,
            optimizer,
            scheduler,
            train_loader,
            mode=config.training_mode,
        )

        loss_dict = {"train_loss": loss_val}
        if (epoch + 1) % config.val_freq == 0:
            logging.info("Starting eval...")
            logging.info("Running testing in epoch {}".format(epoch + 1))

            if config.training_mode in ("mix", "FR"):
                lossf, trho_sf, trho_pf = eval_epoch_FR(
                    config, epoch, net, criterion, val_loader
                )
                loss_dict.update({"test_loss_fr": lossf})

                # add params to tensorboard
                writer.add_scalars(
                    "metric/FR/SRCC", {"train": rho_s, "test": trho_sf}, epoch
                )
                writer.add_scalars(
                    "metric/FR/PLCC", {"train": rho_p, "test": trho_pf}, epoch
                )

                # record best
                if trho_sf > best_sroccf or trho_pf > best_plccf:
                    best_sroccf = max(best_sroccf, trho_sf)
                    best_plccf = max(best_plccf, trho_pf)
                    # save weights
                    ckpt_name = "FR_epoch{}_plcc_{:.4f}_srocc_{:.4f}.pth".format(
                        epoch + 1, trho_pf, trho_sf
                    )
                    model_save_path = os.path.join(
                        config.output_path, config.model_name, ckpt_name
                    )
                    torch.save(net.state_dict(), model_save_path)
                    logging.info(
                        "Saving FR weights and model of epoch{}, SRCC:{}, PLCC:{}".format(
                            epoch + 1, trho_sf, trho_pf
                        )
                    )

            if config.training_mode in ("mix", "NR"):
                lossn, trho_sn, trho_pn = eval_epoch_NR(
                    config, epoch, net, criterion, val_loader
                )
                loss_dict.update({"test_loss_nr": lossn})

                # add params to tensorboard
                writer.add_scalars(
                    "metric/NR/SRCC", {"train": rho_s, "test": trho_sn}, epoch
                )
                writer.add_scalars(
                    "metric/NR/PLCC", {"train": rho_p, "test": trho_pn}, epoch
                )

                # record best
                if trho_sn > best_sroccn or trho_pn > best_plccn:
                    best_sroccn = max(best_sroccn, trho_sn)
                    best_plccn = max(best_plccn, trho_pn)
                    # save weights
                    ckpt_name = "NR_epoch{}_plcc_{:.4f}_srocc_{:.4f}.pth".format(
                        epoch + 1, trho_pn, trho_sn
                    )
                    model_save_path = os.path.join(
                        config.output_path, config.model_name, ckpt_name
                    )
                    torch.save(net.state_dict(), model_save_path)
                    logging.info(
                        "Saving NR weights and model of epoch{}, SRCC:{}, PLCC:{}".format(
                            epoch + 1, trho_sn, trho_pn
                        )
                    )

            logging.info("Eval done...")

            writer.add_scalars(
                "loss",
                loss_dict,
                epoch,
            )
