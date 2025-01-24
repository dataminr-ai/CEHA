import pandas as pd
import re


def extract_between_tags(tag: str, string: str, strip: bool = False) -> list[str]:
    string = string.replace("\_type", "_type")
    ext_list = re.findall(f"<{tag}>(.+?)<{tag}>", string, re.DOTALL)
    if strip:
        ext_list = [e.strip() for e in ext_list]
    if ext_list == []:
        ext_list = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
    return ext_list


def load_data(input_path):
    df = pd.read_csv(input_path, header=0)
    df = df.fillna("")
    all_categories_list = []
    for relevance_tag, index, tag1, tag2, tag3, tag4, tag5, train_dev_test_split in zip(
            df["Is the event relevant?"],
            df["Index"],
            df["tribal/communal/ethnic conflict"],
            df["religious conflict"],
            df["socio-political violence against women"],
            df["climate-related security risks"],
            df["Other"],
            df["train_dev_test_split"],
    ):

        all_tags = []
        if tag1.strip() == "X":
            all_tags.append("tribal/communal/ethnic conflict")
        if tag2.strip() == "X":
            all_tags.append("religious conflict")
        if tag3.strip() == "X":
            all_tags.append("socio-political violence against women")
        if tag4.strip() == "X":
            all_tags.append("climate-related security risks")
            all_tags.append("Other")
        all_categories_list.append("; ".join(all_tags))

    df.insert(2, "All Categories", all_categories_list, True)
    annotated_cols = [
        "Is the event relevant?",
        "Why is the event NOT relevant? \n(if applicable)",
        "All Categories",
    ]
    rename_dict = {}
    for col in annotated_cols:
        rename_dict[col] = col + "_DM"
        rename_dict[col + ".2"] = col + "_A1"
        rename_dict[col + ".3"] = col + "_A2"
        rename_dict[col + ".4"] = col + "_A3"
        rename_dict[col + ".5"] = col + "_A4"
        rename_dict[col + ".6"] = col + "_A5"
    df = df.rename(columns=rename_dict)

    df_train = df[df["train_dev_test_split"] == "train"]
    df_dev = df[df["train_dev_test_split"] == "dev"]
    df_test = df[df["train_dev_test_split"] == "test"]
    return df_train, df_dev, df_test
