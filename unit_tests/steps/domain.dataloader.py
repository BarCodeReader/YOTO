from behave import *
import features.helper.data_type
from dataloader import prepare_dataset
from options import make_template
import sure
import argparse


@given("a pseudo config")
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

    context.configs = opt()


@when(
    'we indicate different dataset with name "{dataset}" and detailed info "{data:Json}"'
)
def step_impl(context, dataset, data):
    context.data = data
    context.configs.dataset = dataset
    context.configs.seed = data["seed"]
    context.configs.crop_size = data["crop_size"]


@then("the outcome for FR-IQA loader should match pre-defined format")
def step_impl(context):
    # make dataset
    make_template(context.configs)
    context.train_set, context.val_set = prepare_dataset(
        context.configs, ratio=0
    )  # we set ratio=0 because we want all to be val dataset for ease of checking because shuffle=False in val dataset

    # check dataset meta-info
    context.configs.txt_file.should.equal(context.data["txt_path"])
    context.configs.dis_path.should.equal(context.data["dis_path"])
    context.configs.ref_path.should.equal(context.data["ref_path"])
    context.configs.seed.should.equal(context.data["seed"])
    context.configs.crop_size.should.equal(context.data["crop_size"])


@then("the outcome for NR-IQA loader should match pre-defined format")
def step_impl(context):
    # make dataset
    make_template(context.configs)
    context.train_set, context.val_set = prepare_dataset(
        context.configs, ratio=0
    )  # we set ratio=0 because we want all to be val dataset for ease of checking because shuffle=False in val dataset

    # check dataset meta-info
    context.configs.txt_file.should.equal(context.data["txt_path"])
    context.configs.dis_path.should.equal(context.data["dis_path"])
    context.configs.seed.should.equal(context.data["seed"])
    context.configs.crop_size.should.equal(context.data["crop_size"])


@then(
    'the loaded GT should match pre-defined score "{GT_score_1:f}", "{GT_score_23:f}", "{GT_score_57:f}", which is the 1, 23, 57th row of label.txt'
)
def step_impl(context, GT_score_1, GT_score_23, GT_score_57):
    float(context.val_set.score_data[0]).should.equal(float(GT_score_1))
    float(context.val_set.score_data[22]).should.equal(float(GT_score_23))
    float(context.val_set.score_data[56]).should.equal(float(GT_score_57))
