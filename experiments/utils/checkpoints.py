import os


def get_best_checkpoint(result_folder, include_checkpoint_file=True):
    """
    gets the checkpoint path from the result folder
    :param result_folder: should be a directory with the format 'Algorithm_EnvironmentName_id_trial_date_time'
    :param include_checkpoint_file: whether to include the checkpoint file within the checkpoint folder
    :return: checkpoint_path
    """
    indicator = 'checkpoint_'
    checkpoint_list = []
    for checkpoint_folder in os.listdir(result_folder):
        if indicator in checkpoint_folder:
            checkpoint_num = int(checkpoint_folder[len(indicator):])
            checkpoint_list.append(checkpoint_num)
    if len(checkpoint_list) == 1:
        checkpoint = checkpoint_list[0]
    elif len(checkpoint_list) > 1:
        checkpoint_list = sorted(checkpoint_list)
        # the second to last checkpoint achieved the best reward
        # the last checkpoint is the latest iteration checkpoint
        checkpoint = checkpoint_list[-1]
    else:
        # raise FileNotFoundError(f'no checkpoints exist at {result_folder}')
        return None

    checkpoint_path = os.path.join(result_folder, indicator + str(checkpoint))
    if include_checkpoint_file:
        checkpoint_path = os.path.join(checkpoint_path, f'checkpoint-{checkpoint}')
    return checkpoint_path


def get_list_checkpoint(path):
    checkpoints = []
    for folder in os.listdir(path):
        if os.path.isdir(path):
            checkpoint_path = get_best_checkpoint(os.path.join(path, folder))
            checkpoints.append(checkpoint_path)
    return checkpoints
