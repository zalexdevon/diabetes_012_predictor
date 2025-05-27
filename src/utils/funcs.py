from Mylib import myfuncs
import os
import re
import pandas as pd
from IPython.display import Audio, display
import numpy as np
import itertools

SCORINGS_PREFER_MININUM = ["log_loss", "mse", "mae"]
SCORINGS_PREFER_MAXIMUM = ["accuracy"]


def get_batch_size_from_model_training_name(name):
    pattern = r"batch_(\d+)"
    return int(re.findall(pattern, name)[0])


def gather_result_from_model_training():
    components = []
    model_training_path = "artifacts/model_training"

    list_models_folder_path = [
        f"{model_training_path}/{item}" for item in os.listdir(model_training_path)
    ]
    for models_folder_path in list_models_folder_path:
        list_model_path = [
            f"{models_folder_path}/{item}" for item in os.listdir(models_folder_path)
        ]
        for model_path in list_model_path:
            result = myfuncs.load_python_object(f"{model_path}/result.pkl")
            components.append(result)

    return components


def gather_result_from_model_training_for_1folder(folder):
    components = []
    folder_path = f"artifacts/model_training/{folder}"

    list_model_path = [f"{folder_path}/{item}" for item in os.listdir(folder_path)]

    for model_path in list_model_path:
        result = myfuncs.load_python_object(f"{model_path}/result.pkl")
        components.append(result)

    return components


def gather_result_from_model_training_for_many_folders(folders):
    list_components_for_1folder = [
        gather_result_from_model_training_for_1folder(item) for item in folders
    ]
    return list(itertools.chain(*list_components_for_1folder))


def get_reverse_param_in_sorted(scoring):
    return scoring in SCORINGS_PREFER_MAXIMUM


def make_beep_sound(frequency=1000, duration=1.0, rate=44100, volume=1.0):
    """Tạo ra âm thanh beep <br>
    Tác dụng: Báo hiệu kết thúc của 1 quá trình, vd: model training

    Args:
        frequency (int, optional): _description_. Defaults to 1000.
        duration (float, optional): _description_. Defaults to 1.0.
        rate (int, optional): _description_. Defaults to 44100.
        volume (float, optional): _description_. Defaults to 1.0.
    """
    t = np.linspace(0, duration, int(rate * duration), False)
    tone = np.sin(frequency * 2 * np.pi * t)
    tone *= volume  # Adjust volume (0.0 to 1.0)
    display(Audio(tone, rate=rate, autoplay=True))


def display_model_training_results(results, scoring):
    reverse_param = get_reverse_param_in_sorted(scoring)
    sorted_results = sorted(results, key=lambda item: item[2], reverse=reverse_param)
    for result in sorted_results:
        print(
            f"Model {result[0]}\n-> Train {scoring}: {result[1]}, Val {scoring}: {result[2]}, Time: {result[3]} (s)"
        )


def get_model_training_result_from_1model_index(model_name, model_index):
    result_path = f"artifacts/model_training/{model_name}/{model_index}/result.pkl"

    if os.path.exists(result_path) == False:
        return None

    result = myfuncs.load_python_object(result_path)
    return result


def get_model_training_result_from_model_indices(model_name, model_indices, scoring):
    if model_indices == []:
        return

    results = [
        get_model_training_result_from_1model_index(model_name, model_index)
        for model_index in model_indices
    ]
    for result, model_index in zip(results, model_indices):
        if result is None:
            print(f"Model {model_index} chưa được trained")

        print(
            f"Model {result[0]}\n-> Train {scoring}: {result[1]}, Val {scoring}: {result[2]}, Time: {result[3]} (s)"
        )
