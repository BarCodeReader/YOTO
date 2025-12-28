from torchvision import transforms
from torch.utils.data import DataLoader
from utils.process import (
    RandCrop,
    ToTensor,
    RandHorizontalFlip,
    Normalize,
    five_point_crop,
    random_crop,
)
from utils.process import (
    split_dataset_live,
    split_dataset_tid2013,
    split_dataset_kadid10k,
    split_dataset_csiq,
    split_dataset_koniq10k,
    split_dataset_livec,
    split_dataset_livefb,
)
from data.tid2013 import Tid2013
from data.live import Live
from data.kadid10k import Kadid10k
from data.csiq import CSIQ
from data.koniq10k import Koniq10k
from data.livec import LiveC
from data.livefb import LiveFB
import logging
from utils.color import BOLD, ENDC, GREEN


def prepare_dataset(config, cross_check=False, ratio=0.8):
    # data load
    if cross_check:
        cross_check_dataset = (config.cross_check_dataset).split(" ")
        if config.dataset in cross_check_dataset:
            ratio = 0
            print(
                BOLD
                + GREEN
                + "--> Dataset [{}] in Cross Check mode! Training ratio is {:.2f}, all will be validation set.".format(
                    config.dataset, ratio
                )
                + ENDC
            )
        else:
            ratio = 1
            print(
                BOLD
                + GREEN
                + "--> Dataset [{}] in Cross Check mode! Training ratio is {:.2f}, all will be training set".format(
                    config.dataset, ratio
                )
                + ENDC
            )
    else:
        print(
            BOLD
            + GREEN
            + "--> Dataset [{}] in Normal mode! Training ratio is {}.".format(
                config.dataset, ratio
            )
            + ENDC
        )

    if config.dataset == "tid2013":
        train_split, val_split = split_dataset_tid2013(
            txt_file_name=config.txt_file, split_seed=config.seed, ratio=ratio
        )
        if train_split:
            train_dataset = Tid2013(
                ref_path=config.ref_path,
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=train_split,
                transform=transforms.Compose(
                    [
                        RandCrop(config.crop_size),
                        Normalize(0.5, 0.5),
                        RandHorizontalFlip(),
                        ToTensor(),
                    ]
                ),
            )
        else:
            train_dataset = []
        if val_split:
            val_dataset = Tid2013(
                ref_path=config.ref_path,
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=val_split,
                transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
            )
        else:
            val_dataset = []
    elif config.dataset == "live":
        train_split, val_split = split_dataset_live(
            txt_file_name=config.txt_file, split_seed=config.seed, ratio=ratio
        )
        if train_split:
            train_dataset = Live(
                ref_path=config.ref_path,
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=train_split,
                transform=transforms.Compose(
                    [
                        RandCrop(config.crop_size),
                        Normalize(0.5, 0.5),
                        RandHorizontalFlip(),
                        ToTensor(),
                    ]
                ),
            )
        else:
            train_dataset = []
        if val_split:
            val_dataset = Live(
                ref_path=config.ref_path,
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=val_split,
                transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
            )
        else:
            val_dataset = []
    elif config.dataset == "kadid10k":
        train_split, val_split = split_dataset_kadid10k(
            txt_file_name=config.txt_file, split_seed=config.seed, ratio=ratio
        )
        if train_split:
            train_dataset = Kadid10k(
                ref_path=config.ref_path,
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=train_split,
                transform=transforms.Compose(
                    [
                        RandCrop(config.crop_size),
                        Normalize(0.5, 0.5),
                        RandHorizontalFlip(),
                        ToTensor(),
                    ]
                ),
            )
        else:
            train_dataset = []
        if val_split:
            val_dataset = Kadid10k(
                ref_path=config.ref_path,
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=val_split,
                transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
            )
        else:
            val_dataset = []
    elif config.dataset == "csiq":
        train_split, val_split = split_dataset_csiq(
            txt_file_name=config.txt_file, split_seed=config.seed, ratio=ratio
        )
        if train_split:
            train_dataset = CSIQ(
                ref_path=config.ref_path,
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=train_split,
                transform=transforms.Compose(
                    [
                        RandCrop(config.crop_size),
                        Normalize(0.5, 0.5),
                        RandHorizontalFlip(),
                        ToTensor(),
                    ]
                ),
            )
        else:
            train_dataset = []
        if val_split:
            val_dataset = CSIQ(
                ref_path=config.ref_path,
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=val_split,
                transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
            )
        else:
            val_dataset = []
    elif config.dataset == "koniq10k":
        train_split, val_split = split_dataset_koniq10k(
            txt_file_name=config.txt_file, split_seed=config.seed, ratio=ratio
        )
        if train_split:
            train_dataset = Koniq10k(
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=train_split,
                transform=transforms.Compose(
                    [
                        RandCrop(config.crop_size),
                        Normalize(0.5, 0.5),
                        RandHorizontalFlip(),
                        ToTensor(),
                    ]
                ),
                resize=True,
            )
        else:
            train_dataset = []
        if val_split:
            val_dataset = Koniq10k(
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=val_split,
                transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
                resize=True,
            )
        else:
            val_dataset = []
    elif config.dataset == "livec":
        train_split, val_split = split_dataset_livec(
            txt_file_name=config.txt_file, split_seed=config.seed, ratio=ratio
        )
        if train_split:
            train_dataset = LiveC(
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=train_split,
                transform=transforms.Compose(
                    [
                        RandCrop(config.crop_size),
                        Normalize(0.5, 0.5),
                        RandHorizontalFlip(),
                        ToTensor(),
                    ]
                ),
            )
        else:
            train_dataset = []
        if val_split:
            val_dataset = LiveC(
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=val_split,
                transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
            )
        else:
            val_dataset = []
    elif config.dataset == "livefb":
        train_split, val_split = split_dataset_livefb(
            txt_file_name=config.txt_file, split_seed=config.seed, ratio=ratio
        )
        if train_split:
            train_dataset = LiveFB(
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=train_split,
                transform=transforms.Compose(
                    [
                        RandCrop(config.crop_size),
                        Normalize(0.5, 0.5),
                        RandHorizontalFlip(),
                        ToTensor(),
                    ]
                ),
            )
        else:
            train_dataset = []
        if val_split:
            val_dataset = LiveFB(
                dis_path=config.dis_path,
                txt_file_name=config.txt_file,
                list_name=val_split,
                transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
            )
        else:
            val_dataset = []
    else:
        raise Exception("dataset not valid")

    logging.info("number of train scenes: {}".format(len(train_dataset)))
    logging.info("number of val scenes: {}".format(len(val_dataset)))

    return train_dataset, val_dataset


def prepare_dataloader(config, cross_check=False):
    train_dataset, val_dataset = prepare_dataset(config, cross_check=cross_check)

    if train_dataset:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            drop_last=False,
            shuffle=True,
        )
    else:
        train_loader = None
    eval_batch = 1 if "live" in config.dataset else config.batch_size
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=eval_batch,  # config.batch_size,
        num_workers=config.num_workers,
        drop_last=False,
        shuffle=False,
    )

    print(
        BOLD
        + GREEN
        + "Dataset: {}, train length: {} val length: {}".format(
            config.dataset, len(train_dataset), len(val_dataset)
        )
        + ENDC
    )

    return train_loader, val_loader
