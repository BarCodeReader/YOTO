import os
import torch
import numpy as np
import random
import importlib
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.inference_process import sort_file
from utils.process import ToTensor, Normalize, five_point_crop, random_crop
from dataloader import prepare_dataloader
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from options import get_option


def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def eval_epoch(config, net, test_loader):
    with torch.no_grad():
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
                    if config.infer_mode == "FR":
                        pred += net(x_d, x_r)
                    elif config.infer_mode == "NR":
                        pred += net(x_d, x_d)
                    else:
                        raise NotImplementedError("infer mode not implemented")
            else:
                for i in range(config.num_avg_val):
                    x_d = data["d_img_org"].cuda()
                    x_r = data["r_img_org"].cuda()
                    labels = data["score"]
                    labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                    x_d = five_point_crop(i, d_img=x_d, config=config)
                    x_r = five_point_crop(i, d_img=x_r, config=config)
                    if config.infer_mode == "FR":
                        pred += net(x_d, x_r)
                    elif config.infer_mode == "NR":
                        pred += net(x_d, x_d)
                    else:
                        raise NotImplementedError("infer mode not implemented")

            pred /= config.num_avg_val

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        msg = "Test result: ===== SRCC:{:.4} ===== PLCC:{:.4}".format(rho_s, rho_p)
        print(msg)


if __name__ == "__main__":
    config = get_option()

    print(f"=======inference mode: {config.infer_mode}")

    cpu_num = 1
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU)

    setup_seed(20)

    # data load
    _, test_loader = prepare_dataloader(config, cross_check=config.cross_check)

    module = importlib.import_module("models.{}".format(config.network.lower()))
    net = module.Net(config, device="cuda")
    net.load_state_dict(torch.load(config.checkpoint))
    net = net.cuda()

    eval_epoch(config, net, test_loader)
