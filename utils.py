import json
import random
import re
import numpy as np
import pandas as pd


def save_to_jsonl(dataset, file_name: str) -> None:
    with open(file_name, "w") as file:
        for entry in dataset:
            json_line = json.dumps(entry)
            file.write(json_line + "\n")


def load_from_jsonl(file_name: str) -> list:
    with open(file_name, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def extract_predictions(response: str) -> list:
    return [int(label) for label in re.findall(r"OUTPUT:\s*(\d)", response)]


def get_accuracy(labels: list, outputs: list) -> float:
    if len(labels) != len(outputs):
        return np.nan
    return sum(
        [1 if label == output else 0 for label, output in zip(labels, outputs)]
    ) / len(labels)


def sample_dataset(dataset: list, n: int, seed=42):
    """
    Samples 2n items by sampling n labels 0 and n labels 1 and removes them from the original list.

    :param data: List of dictionaries.
    :param n: Number of items to sample.
    :return: Tuple of (sampled items, modified data list).
    """
    random.seed(seed)
    # Filter items with the specified label
    filtered_items_pos = [item for item in dataset if item["label"] == 1]
    filtered_items_neg = [item for item in dataset if item["label"] == 0]

    sampled_items_neg = random.sample(filtered_items_neg, n // 2)
    sampled_items_pos = random.sample(filtered_items_pos, n // 2)
    sampled_items = sampled_items_neg + sampled_items_pos

    random.shuffle(sampled_items)
    modified_dataset = [item for item in dataset if item not in sampled_items]

    return sampled_items, modified_dataset


def get_prompt_and_labels(
    dataset: list, start_prompt: str, n_shots: int = 2, test_size: int = 20
):
    samples, modified_dataset = sample_dataset(dataset, n_shots)
    for s in samples:
        start_prompt += f"INPUT: \"{s['text']}\" OUTPUT: {s['label']}\n\n"

    labels = []
    for i in range(test_size):
        start_prompt += f"INPUT: \"{modified_dataset[i]['text']}\"\n\n"
        labels.append(modified_dataset[i]["label"])

    return start_prompt, labels


def process_results_table(results_dict: dict, n_shots: list) -> pd.DataFrame:
    df_dict = {}
    for dataset, values in results_dict.items():
        for i in range(len(results_dict[dataset]["preds"])):
            df_dict[f"{dataset}_{i}"] = {
                f"accuracy_{n_shots}": get_accuracy(
                    results_dict[dataset]["labels"][i],
                    results_dict[dataset]["preds"][i],
                ),
                f"explanation_{n_shots}": results_dict[dataset]["explanations"][i],
            }
    return pd.DataFrame.from_dict(df_dict, orient="index")
