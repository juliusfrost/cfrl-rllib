def get_introduction_text(text_config):
    agent_article = text_config.get('agent_article', 'a')
    agent_name = text_config.get('agent_name', 'driver')
    agent_name_plural = text_config.get('agent_name_plural', 'drivers')
    agent_visual_description = text_config.get('agent_visual_description', 'in the yellow car')
    action_name = text_config.get('action_name', 'steers the car')
    explain_env_dist = text_config.get('explain_env_dist', 'highway route 1')
    test_env_dist = text_config.get('test_env_dist', 'highway route 2')
    p1 = f'The following videos show the behavior of {agent_article} {agent_name} in different situations. ' \
         f'Your task is to watch explanation videos of the {agent_name} {agent_visual_description} ' \
         f'and try to predict how the {agent_name} would react in different situations. ' \
         f'We label the {agent_name} {agent_visual_description} in the explanation videos as {agent_name} A. '
    p2 = f'To measure how well you understood {agent_name} A\'s behavior, you will be given an evaluation task. ' \
         f'In the evaluation task, you will be shown multiple videos with different {agent_name_plural}. ' \
         f'In the first half of the video, there is a single {agent_name} A ' \
         f'corresponding to the {agent_name} in the explanation videos. ' \
         f'In the second half of the video, there is either the same {agent_name} A ' \
         f'or a different {agent_name} that can lead to different outcomes. ' \
         f'Your goal is to select which {agent_name} ' \
         f'you think is the same one as the one you saw in the explanation videos ' \
         f'by selecting the video which did not switch {agent_name_plural}. ' \
         f'In other words, select the outcome obtained by {agent_name} A. '
    p3 = f'The {agent_name} in the explanation videos {action_name} in {explain_env_dist} ' \
         f'while the {agent_name} in the evaluation videos {action_name} in {test_env_dist}. ' \
         f'The different conditions can affect behavior of the {agent_name} ' \
         f'and you should take this into account when selecting the correct outcome.'
    return [p1, p2, p3]


def get_explain_study_text(explanation_method, text_config):
    agent_name = text_config.get('agent_name', 'driver')
    action_name = text_config.get('action_name', 'steers the car')
    explain_env_dist = text_config.get('explain_env_dist', 'highway route 1')
    text = ''
    text += f'The following video shows a segment of {agent_name} A\'s experience on {explain_env_dist}. '
    if explanation_method == 'random':
        text += 'The segments shown here are selected randomly. '
        text += f'In all parts of the video, {agent_name} A {action_name}. '
    elif explanation_method == 'critical':
        text += 'The segments shown here are selected where it is critical to take certain actions. '
        text += f'In all parts of the video, {agent_name} A {action_name}. '
    elif explanation_method == 'counterfactual':
        text += 'The segments shown here are selected randomly. '
        text += f'In the beginning part of the video, {agent_name} A {action_name}. '
        text += f'At some point in time, {agent_name} B takes over ' \
                'to move off course than originally planned. '
        text += f'Lastly, {agent_name} A takes over control again. '
    else:
        raise NotImplementedError
    text += f'Your task: observe the behavior of {agent_name} A and try to detect any patterns. '
    return text


def get_eval_study_text(text_config, config):
    agent_name = text_config.get('agent_name', 'driver')
    action_name = text_config.get('action_name', 'steers the car')
    test_env_dist = text_config.get('test_env_dist', 'highway route 2')
    num_eval_policies = len(config['eval_config']['eval_policies'])
    text = f'The following video shows segments of {agent_name} A\'s experience on {test_env_dist}. '
    text += f'In the beginning half of each video, {agent_name} A {action_name} and shows the same thing for all videos. '
    text += 'In the ending half of each video, ' \
            f'a different {agent_name} {action_name} and shows different possible outcomes. '
    drivers = ['A'] + [chr(i) for i in range(ord('C'), ord('C') + num_eval_policies)]
    text += f'The {agent_name} that {action_name} in the ending half could be '
    for i, driver_name in enumerate(drivers):
        if i == len(drivers) - 1:
            text += f'or {agent_name} {driver_name}. '
        else:
            text += f'{agent_name} {driver_name}, '
    text += f'The videos are shuffled to hide which outcome was from {agent_name} A. '
    text += f'The outcomes are labeled outcome 1 through outcome {len(drivers)} from left to right. '
    text += f'Your task: remember the behavior of {agent_name} A from previous tasks,' \
            f' and guess which outcome was a result of {agent_name} A.'
    return text


def get_title_text():
    return 'Explainable Reinforcement Learning User Evaluation'


def get_explain_heading_text():
    return 'Explanation'


def get_eval_heading_text():
    return 'Evaluation'


def get_question_text(text_config):
    agent_name = text_config.get('agent_name', 'driver')
    return f'Which outcome was a result of {agent_name} A? ' \
           f'Write your answer in the accompanying answer sheet. '
