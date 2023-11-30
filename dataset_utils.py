import random
from utils import save_to_jsonl
from gpt_utils import GPTChat
import re


def zip_dataset(inputs, labels):
    return [{"text": input, "label": label} for input, label in zip(inputs, labels)]


def get_dataset_numbers(
    dataset,
    size: int = 100,
    str_length: int = 30,
    true_proportion: float = 0.5,
    seed: int = 42,
) -> list:
    dataset = dataset.shuffle(seed=seed)

    def containsNumber(string):
        return any([char.isdigit() for char in string])

    dataset_true = []
    dataset_false = []
    for j, text in enumerate(dataset["text"]):
        t = text[:str_length]
        if containsNumber(t) and len(dataset_true) < (size * true_proportion):
            dataset_true.append(t)
        elif not containsNumber(t) and len(dataset_false) < (
            size * (1 - true_proportion)
        ):
            dataset_false.append(t)
        if (len(dataset_true) + len(dataset_false)) == size:
            break

    dataset = dataset_true.copy()
    dataset.extend(dataset_false)
    dataset = zip_dataset(dataset, [1] * len(dataset_true) + [0] * len(dataset_false))
    random.shuffle(dataset)
    return dataset


def get_dataset_capitalized(
    dataset,
    size: int = 100,
    str_length: int = 30,
    true_proportion: float = 0.5,
    seed: int = 42,
) -> list:
    dataset = dataset.shuffle(seed=seed)
    random.seed(seed)

    dataset_list = []
    labels_list = []
    for j, text in enumerate(dataset["text"]):
        t = text[:str_length]
        if r := random.random() > true_proportion:
            t = t.lower()
            labels_list.append(1)
        elif r > (1 - true_proportion) / 2:
            t = t.upper()
            labels_list.append(0)
        else:
            t = list(t.lower())
            for _ in range(random.randint(1, 5)):
                i = random.randint(0, len(t) - 1)
                t[i] = t[i].upper()
            t = "".join(t)
            labels_list.append(0)
        dataset_list.append(t)
        if j == size:
            break
    dataset = zip_dataset(dataset_list, labels_list)
    random.shuffle(dataset)
    return dataset


def get_dataset_primes(min: int = 100, max: int = 1000) -> list:
    nums = []
    labels = []
    for i in range(min, max):
        if all(i % j != 0 for j in range(2, i)):
            labels.append(1)
        else:
            labels.append(0)
        nums.append(i)
    dataset = zip_dataset(nums, labels)
    random.shuffle(dataset)
    return dataset


def _generate_dataset_gpt(
    positive_prompt: str, negative_prompt: str, n_sentences: int = 200
) -> list:
    chat = GPTChat(
        system_message=f"""You are a sentence generator. 
                   I will give you a sentence generating task and you will generate {n_sentences} sentences that fulfill the condition.
                   Make sure that sentences are very short and understandable by a 7 year old child. 
                    Each sentence is separated by '\n'""",
        model="gpt-3.5-turbo",
    )
    response_pos = chat.send_message(
        "Task: " + positive_prompt + f" Generate exactly {n_sentences // 2} sentences."
    )
    response_pos = response_pos.split("\n")

    chat.messages = chat.messages[:-2]  # remove last two messages, to save context
    response_neg = chat.send_message(
        "Task: " + negative_prompt + f" Generate exactly {n_sentences // 2} sentences."
    )
    response_neg = response_neg.split("\n")

    response = response_pos + response_neg
    dataset = [re.sub(r"^\d+\.\s*", "", s) for s in response]
    dataset = zip_dataset(dataset, [1] * len(response_pos) + [0] * len(response_neg))
    random.shuffle(dataset)
    return dataset


def save_dataset_fresh(
    n_sentences: int = 200, file_path: str = "data/fruits_and_vegetables.json"
) -> None:
    pos_prompt = "Generate unique sentences that contain any fruit, make sure there are no vegatables in the sentences."
    neg_prompt = "Generate unique sentences that contain any vegetable, make sure there are no fruits in the sentences."
    dataset = _generate_dataset_gpt(pos_prompt, neg_prompt, n_sentences)
    save_to_jsonl(dataset, file_path)


def save_dataset_contradictions(
    n_sentences: int = 200, file_path: str = "data/contradictions.json"
) -> None:
    pos_prompt = "Generate unique sentences that contain a very obvious contradiction."
    neg_prompt = "Generate unique short sentences, make sure there are no contradictions in the sentences."
    dataset = _generate_dataset_gpt(pos_prompt, neg_prompt, n_sentences)
    save_to_jsonl(dataset, file_path)
