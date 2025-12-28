from behave import *
import features.helper.data_type
from dataloader import prepare_dataset
from options import make_template, cross_check_template
import argparse


@given("a pseudo config with cross-check mode on")
def step_impl(context):
    class opt:
        def __init__(self):
            self.root_dir = "./"
            self.dataset = None
            self.seed = None
            self.crop_size = None
            self.output_path = None
            self.batch_size = 1
            self.num_workers = 1
            self.ref_path = None
            self.dis_path = None
            self.txt_file = None
            self.cross_check = True
            self.cross_check_dataset = None

    context.configs = opt()


@when(
    'we indicate training dataset as "{dataset}", cross-check dataset as "{cross_check_dataset}" and their detailed info "{data:Json}"'
)
def step_impl(context, dataset, cross_check_dataset, data):
    context.data = data
    context.configs.dataset = dataset
    context.configs.cross_check_dataset = cross_check_dataset
    context.configs.seed = data["seed"]
    context.configs.crop_size = data["crop_size"]


@then(
    "the outcome for FR-IQA loader under cross-check mode should match pre-defined format"
)
def step_impl(context):
    # make training dataset
    make_template(context.configs)
    context.train_set, context.val_set = prepare_dataset(
        context.configs, cross_check=context.configs.cross_check
    )

    # training dataset meta-info
    context.configs.txt_file.should.equal(context.data["txt_path"])
    context.configs.dis_path.should.equal(context.data["dis_path"])
    context.configs.ref_path.should.equal(context.data["ref_path"])
    context.configs.seed.should.equal(context.data["seed"])
    context.configs.crop_size.should.equal(context.data["crop_size"])

    # make cross-check dataset
    cross_check_template(context.configs.cross_check_dataset, context.configs)
    context.configs.dataset = context.configs.cross_check_dataset
    context.train_set_cc, context.val_set_cc = prepare_dataset(
        context.configs, cross_check=context.configs.cross_check
    )

    # cross-check dataset meta-info
    context.configs.txt_file.should.equal(context.data["txt_pathc"])
    context.configs.dis_path.should.equal(context.data["dis_pathc"])
    context.configs.ref_path.should.equal(context.data["ref_pathc"])
    context.configs.seed.should.equal(context.data["seed"])
    context.configs.crop_size.should.equal(context.data["crop_size"])


@then(
    "the outcome for NR-IQA loader under cross-check mode should match pre-defined format"
)
def step_impl(context):
    # make training dataset
    make_template(context.configs)
    context.train_set, context.val_set = prepare_dataset(
        context.configs, cross_check=context.configs.cross_check
    )

    # training dataset meta-info
    context.configs.txt_file.should.equal(context.data["txt_path"])
    context.configs.dis_path.should.equal(context.data["dis_path"])
    context.configs.seed.should.equal(context.data["seed"])
    context.configs.crop_size.should.equal(context.data["crop_size"])

    # make cross-check dataset
    cross_check_template(context.configs.cross_check_dataset, context.configs)
    context.configs.dataset = context.configs.cross_check_dataset
    context.train_set_cc, context.val_set_cc = prepare_dataset(
        context.configs, cross_check=context.configs.cross_check
    )

    # cross-check dataset meta-info
    context.configs.txt_file.should.equal(context.data["txt_pathc"])
    context.configs.dis_path.should.equal(context.data["dis_pathc"])
    context.configs.seed.should.equal(context.data["seed"])
    context.configs.crop_size.should.equal(context.data["crop_size"])


@then(
    'the loaded cross-check dataset GT should match pre-defined score for train dataset: "{GT_score_1:f}", "{GT_score_23:f}", "{GT_score_57:f}", and cross_check dataset: "{GT_score_1c:f}", "{GT_score_23c:f}", "{GT_score_57c:f}", which is the 1, 23, 57th row of label.txt'
)
def step_impl(
    context,
    GT_score_1,
    GT_score_23,
    GT_score_57,
    GT_score_1c,
    GT_score_23c,
    GT_score_57c,
):
    # for cross check train dataset
    float(context.train_set.score_data[0]).should.equal(float(GT_score_1))
    float(context.train_set.score_data[22]).should.equal(float(GT_score_23))
    float(context.train_set.score_data[56]).should.equal(float(GT_score_57))
    # for cross check val dataset
    float(context.val_set_cc.score_data[0]).should.equal(float(GT_score_1c))
    float(context.val_set_cc.score_data[22]).should.equal(float(GT_score_23c))
    float(context.val_set_cc.score_data[56]).should.equal(float(GT_score_57c))
