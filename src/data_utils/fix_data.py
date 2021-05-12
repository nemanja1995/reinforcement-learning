import os


logs_path = "data/logs"
log_dirs = os.listdir(logs_path)
for log_dir in log_dirs:
    if log_dir.find("gamma0") != -1:
        old_path = os.path.join(logs_path, log_dir)
        new_name = log_dir.replace("gamma0", "gamma-0")
        new_path = os.path.join(logs_path, new_name)
        os.rename(old_path, new_path)
