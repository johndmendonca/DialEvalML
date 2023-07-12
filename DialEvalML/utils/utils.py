import pickle
import re
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import numpy as np
import json
import pandas as pd

MAX_LENGTH = 512


def save_pkl(obj, file: str) -> None:
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def completions_with_backoff(**kwargs: dict):
    return openai.ChatCompletion.create(**kwargs)


def normalize_df(
    dataset_name: str, df: pd.DataFrame, ds_meta: pd.DataFrame
) -> pd.DataFrame:
    dataset_meta = ds_meta[dataset_name]
    for annotation in dataset_meta["annotations"]:
        df["annotations." + annotation] = df["annotations." + annotation].apply(
            dataset_meta["aggregation"]
        )
    return df


def test_normalize(
    predictions: dict, dev_dir: str, d_ids: pd.Series = None, dial: bool = False
) -> None:
    for submetric in predictions:
        if submetric in ["nsp", "vsp", "engagement"]:
            for model_name in predictions[submetric]:
                with open(
                    dev_dir + f"/{submetric}/{model_name}/minmax.json", "r"
                ) as outfile:
                    metric_minimum, metric_maximum = json.load(outfile)

                normalized_preds = (
                    predictions[submetric][model_name] - np.float64(metric_minimum)
                ) / (metric_maximum - metric_minimum)

                if dial:
                    group_by_means = []
                    for i in np.unique(d_ids):
                        tmp = normalized_preds[np.where(d_ids == i)]
                        group_by_means.append(np.mean(tmp))
                    predictions[submetric][model_name] = group_by_means
                else:
                    predictions[submetric][
                        model_name
                    ] = normalized_preds.squeeze().tolist()

        else:
            with open(dev_dir + f"/{submetric}/minmax.json", "r") as outfile:
                metric_minimum, metric_maximum = json.load(outfile)

            normalized_preds = (predictions[submetric] - np.float64(metric_minimum)) / (
                metric_maximum - metric_minimum
            )

            if dial and submetric not in [
                "gpt-overall",
                "gpt-relevant",
                "gpt-engaging",
                "gpt-content",
            ]:
                group_by_means = []
                for i in np.unique(d_ids):
                    tmp = normalized_preds[np.where(d_ids == i)]
                    group_by_means.append(np.mean(tmp))
                predictions[submetric] = group_by_means
            else:
                predictions[submetric] = normalized_preds.squeeze().tolist()


def dev_normalize(
    predictions: dict, output_dir: str, dial_level_datasets: dict, dev: dict
) -> None:
    for submetric in predictions:
        if submetric in ["nsp", "vsp", "engagement"]:
            for model_name in predictions[submetric]:
                metric_minimum = np.inf
                metric_maximum = -np.inf
                for dataset_name in predictions[submetric][model_name]:
                    for lang in predictions[submetric][model_name][dataset_name]:
                        max_element = np.max(
                            predictions[submetric][model_name][dataset_name][lang]
                        )
                        min_element = np.min(
                            predictions[submetric][model_name][dataset_name][lang]
                        )

                        if max_element > metric_maximum:
                            metric_maximum = max_element
                        if min_element < metric_minimum:
                            metric_minimum = min_element

                with open(
                    output_dir + f"/{submetric}/{model_name}/minmax.json", "w"
                ) as outfile:
                    json.dump([metric_minimum.item(), metric_maximum.item()], outfile)

                for dataset_name in predictions[submetric][model_name]:
                    for lang in predictions[submetric][model_name][dataset_name]:
                        normalized_preds = (
                            predictions[submetric][model_name][dataset_name][lang]
                            - metric_minimum
                        ) / (metric_maximum - metric_minimum)

                        if dataset_name in dial_level_datasets:
                            group_by_means = []
                            d_ids = pd.read_csv(dev[dataset_name]["path"] + "_en.csv")[
                                "did"
                            ]
                            for i in np.unique(d_ids):
                                tmp = normalized_preds[np.where(d_ids == i)]
                                group_by_means.append(np.mean(tmp))
                            predictions[submetric][model_name][dataset_name][
                                lang
                            ] = group_by_means
                        else:
                            predictions[submetric][model_name][dataset_name][
                                lang
                            ] = normalized_preds.squeeze().tolist()

        else:
            metric_minimum = np.inf
            metric_maximum = -np.inf
            for dataset_name in predictions[submetric]:
                for lang in predictions[submetric][dataset_name]:
                    max_element = np.max(predictions[submetric][dataset_name][lang])
                    min_element = np.min(predictions[submetric][dataset_name][lang])

                    if max_element > metric_maximum:
                        metric_maximum = max_element
                    if min_element < metric_minimum:
                        metric_minimum = min_element
            with open(output_dir + f"/{submetric}/minmax.json", "w") as outfile:
                json.dump([metric_minimum.item(), metric_maximum.item()], outfile)

            for dataset_name in predictions[submetric]:
                for lang in predictions[submetric][dataset_name]:
                    normalized_preds = (
                        predictions[submetric][dataset_name][lang] - metric_minimum
                    ) / (metric_maximum - metric_minimum)

                    if dataset_name in dial_level_datasets and submetric not in [
                        "gpt-overall",
                        "gpt-relevant",
                        "gpt-engaging",
                        "gpt-content",
                    ]:
                        group_by_means = []
                        d_ids = pd.read_csv(dev[dataset_name]["path"] + "_en.csv")[
                            "did"
                        ]
                        for i in np.unique(d_ids):
                            tmp = normalized_preds[np.where(d_ids == i)]
                            group_by_means.append(np.mean(tmp))
                        predictions[submetric][dataset_name][lang] = group_by_means
                    else:
                        predictions[submetric][dataset_name][
                            lang
                        ] = normalized_preds.squeeze().tolist()


def completion_to_score(message):
    matches = re.findall(r"\b[1-5]\b", message)

    if not matches:
        return -1

    return np.mean(matches)
