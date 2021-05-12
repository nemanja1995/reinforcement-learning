# from shutil import copyfile
import shutil

import os

research_path = "data/research"
logs_path = "data/logs"

if not os.path.exists(research_path):
    os.makedirs(research_path)

# parameters = ["epsilon-decay"]
parameters = ["gam2ma"]

for parameter in parameters:
    parameter_path = os.path.join(research_path, parameter)
    if os.path.exists(parameter_path):
        # os.rmt(parameter_path)
        shutil.rmtree(parameter_path)
    os.makedirs(parameter_path)
    data_dir_name = "data"
    rewards_dir_name = "rewards"
    loss_dir_name = "loss"



    log_dirs = os.listdir(logs_path)
    for log_dir in log_dirs:
        label = log_dir.split("_")[0]

        if label.find(parameter) != -1:
            log_files_path = os.path.join(logs_path, log_dir)
            log_files = os.listdir(log_files_path)
            for log_file in log_files:

                if log_file.find("data") != -1 or log_file.find("rewards") != -1 or log_file.find("episodes") != -1:

                    label_path = os.path.join(parameter_path, label)
                    if not os.path.exists(label_path):
                        os.makedirs(label_path)
                        for dir_name in [data_dir_name, rewards_dir_name, loss_dir_name]:
                            os.makedirs(os.path.join(label_path, dir_name))

                    label_subdir_path = label_path
                    if log_file.find("data") != -1:
                        label_subdir_path = os.path.join(label_path, "data")

                    if log_file.find("rewards") != -1:
                        label_subdir_path = os.path.join(label_path, "rewards")

                    if log_file.find("episodes") != -1:
                        label_subdir_path = os.path.join(label_path, "loss")

                    log_file_path = os.path.join(log_files_path, log_file)
                    log_file_new_path = os.path.join(label_subdir_path, "{}_{}".format(label, log_file))
                    shutil.copyfile(log_file_path, log_file_new_path)
