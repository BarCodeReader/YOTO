Feature: domain.dataloader
   Scenario Outline: check dataloader for FR-IQA datasets
     Given a pseudo config
      When we indicate different dataset with name "<dataset>" and detailed info "<data>"
      Then the outcome for FR-IQA loader should match pre-defined format
      And the loaded GT should match pre-defined score "<GT_score_1>", "<GT_score_23>", "<GT_score_57>", which is the 1, 23, 57th row of label.txt
     Examples:
        | dataset  | data  | GT_score_1 | GT_score_23 | GT_score_57 |
        | live     | {"seed":20, "crop_size":224, "txt_path":"./data/live_label.txt", "dis_path":"./data/datasets/LIVE", "ref_path":"./data/datasets/LIVE"}                                                | 0.113551 |0.262300 | 0.417662 |
        | csiq     | {"seed":20, "crop_size":224, "txt_path":"./data/csiq_label.txt", "dis_path":"./data/datasets/CSIQ/dst_imgs", "ref_path":"./data/datasets/CSIQ/src_imgs"}                              | 0.061989 |0.489880 | 0.754413 |
        | tid2013  | {"seed":20, "crop_size":224, "txt_path":"./data/tid2013_label.txt", "dis_path":"./data/datasets/TID2013/distorted_images", "ref_path":"./data/datasets/TID2013/reference_images"}     | 5.51429  |5.08333  | 5.86486  |
        | kadid10k | {"seed":20, "crop_size":224, "txt_path":"./data/kadid10k_label.txt", "dis_path":"./data/datasets/KADID10K/distorted_images", "ref_path":"./data/datasets/KADID10K/reference_images"}  | 4.57     |1.57     | 3.7      |

   Scenario Outline: check dataloader for NR-IQA datasets
     Given a pseudo config
      When we indicate different dataset with name "<dataset>" and detailed info "<data>"
      Then the outcome for NR-IQA loader should match pre-defined format
      And the loaded GT should match pre-defined score "<GT_score_1>", "<GT_score_23>", "<GT_score_57>", which is the 1, 23, 57th row of label.txt
     Examples:
        | dataset  | data  | GT_score_1 | GT_score_23 | GT_score_57 |
        | livec    | {"seed":20, "crop_size":224, "txt_path":"./data/LIVEC_label.txt", "dis_path":"./data/datasets/LIVEC/Images"}                | 66.3595           |19.75             | 79.4379           |
        | livefb   | {"seed":20, "crop_size":224, "txt_path":"./data/livefb_labels.txt", "dis_path":"./data/datasets/LIVEFB/images"}             | 75.44372347614072 |75.55370958207453 | 72.84853156533667 |
        | koniq10k | {"seed":20, "crop_size":224, "txt_path":"./data/koniq10k_label.txt", "dis_path":"./data/datasets/KONIQ/koniq10k_1024x768"}  | 3.82857142857143  |3.07920792079208  | 3.00934579439252  |
