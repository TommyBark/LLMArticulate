import asyncio
import numpy as np
from gpt_utils import (
    AsyncGPTChat,
    manage_chat_async,
    FIRST_PROMPT,
    SECOND_PROMPT,
)
from utils import (
    get_prompt_and_labels,
    extract_predictions,
    get_accuracy,
    load_from_jsonl,
)
from datasets import load_dataset
from dataset_utils import (
    get_dataset_numbers,
    get_dataset_capitalized,
    get_dataset_primes,
)
from time import time


async def run_chats(dataset, number_of_chats=5, n_shots=1, test_size=10):
    chats = [
        AsyncGPTChat(system_message="", model="gpt-3.5-turbo")
        for _ in range(number_of_chats)
    ]
    prompt, labels = get_prompt_and_labels(
        dataset, FIRST_PROMPT, n_shots=n_shots, test_size=test_size
    )
    print("Starting Async Chats")
    start_time = time()
    results = await asyncio.gather(*(manage_chat_async(chat, prompt) for chat in chats))

    # Process results
    outputs = [extract_predictions(response) for response in results]
    accuracies = [get_accuracy(labels, output) for output in outputs]

    explanations = await asyncio.gather(
        *(manage_chat_async(chat, SECOND_PROMPT) for chat in chats)
    )

    end_time = time() - start_time
    print(f"Async Chats took {end_time:.2f} seconds")
    return accuracies, explanations


if __name__ == "__main__":
    number_of_chats = 5
    n_shots = 20
    test_size = 50

    # dataset = load_dataset("imdb", split="train")
    # dataset = get_dataset_numbers(dataset, size=100, true_proportion=0.5, str_length=50)
    # dataset = get_dataset_capitalized(dataset, size=100, str_length=50)
    # dataset = get_dataset_primes(3, 200)
    dataset = load_from_jsonl("data/fruits_and_vegetables.jsonl")
    accuracies, explanations = asyncio.run(
        run_chats(dataset, number_of_chats, n_shots, test_size)
    )
    print(accuracies)
    print(f"Average acc: {np.mean(accuracies):.2f}")
    print(*zip(explanations, accuracies), sep="\n")
