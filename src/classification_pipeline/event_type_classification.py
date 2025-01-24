import argparse
import random
import time
from tqdm import tqdm
from textwrap import dedent
import openai

from ..utils.prompts import (
    system_prompt_type,
    format_prompt_zero_shot_tribal,
    format_prompt_zero_shot_religious,
    format_prompt_zero_shot_female,
    format_prompt_zero_shot_climate,
    databricks_llm_prompt,
    generate_few_shot_prompt_list_type,
    clean_label,
    databricks_llm_prompt_chat,
)
from ..utils.evaluation import event_type_scorer_type
from ..utils.utils import extract_between_tags, load_data
from ..utils.llm_backbone import OpenAILLMCaller, MistralAiLLMCaller

def predict_event_type(args, df_train, df_test):
    """
    Predicts the event type for given test data using a specified large language model (LLM).
    
    Args:
        args (argparse.Namespace):
            - llm_name (str, default="gpt4"): Name of the LLM to use ("mistral", "gpt4"). 
            - max_tokens (int, default=512): Maximum number of tokens for LLM responses.
            - temperature (float, default=0.0): Sampling temperature for LLM.
            - few_shot_num (int, default=0): Number of few-shot examples to include in the prompts.
            - openai_api_key (str, default=None): API key for OpenAI (if using GPT models).
            - mistralai_api_key (str, default=None): API key for Mistral.ai (if using mistral models).
            - mistralai_rps (float, default=0): Request per second limit for Mistral.ai (if using mistral models).

        df_train (pd.DataFrame): 
            A DataFrame containing training data with event descriptions and their event types. 
            Used for generating few-shot examples if `few_shot_num > 0`.

        df_test (pd.DataFrame): 
            A DataFrame containing test data with event descriptions. 
    
    Returns:
        tuple:
            - all_gold_labels (list): A list of ground truth event type labels for the test dataset if provided. Otherwise, this list will be empty.
            - all_sys_labels (list): A list of sets where each set contains the predicted event type labels for an event in the test dataset.

    Notes:
        - This function uses different LLM backends based on the specified `llm_name`. Please refer to the instruction to set up correct access.
    """
    model_id_dic = {
        "gpt4": "gpt-4o",
        "mistral": "mistral-large-latest",
    }


    # randomly select few-shot examples
    random.seed(42)
    if args.few_shot_num > 0:
        df_train_trial = df_train[df_train["tribal/communal/ethnic conflict"] == "X"]
        df_train_religious = df_train[df_train["religious conflict"] == "X"]
        df_train_female = df_train[
            df_train["socio-political violence against women"] == "X"
        ]
        df_train_climate = df_train[df_train["climate-related security risks"] == "X"]
        df_train_other = df_train[df_train["Other"] == "X"]

        df_train_trial_neg = df_train[
            df_train["tribal/communal/ethnic conflict"] != "X"
        ]
        df_train_religious_neg = df_train[df_train["religious conflict"] != "X"]
        df_train_female_neg = df_train[
            df_train["socio-political violence against women"] != "X"
        ]
        df_train_climate_neg = df_train[
            df_train["climate-related security risks"] != "X"
        ]
        df_train_other_neg = df_train[df_train["Other"] != "X"]

        few_shot_examples = {
            "tribal": {"pos": [], "neg": []},
            "religious": {"pos": [], "neg": []},
            "female": {"pos": [], "neg": []},
            "climate": {"pos": [], "neg": []},
            "other": {"pos": [], "neg": []},
        }

        for _ in range(0, args.few_shot_num):
            random_trial_pos = random.sample(
                range(0, len(df_train_trial)), args.few_shot_num
            )
            random_trial_neg = random.sample(
                range(0, len(df_train_trial_neg)), args.few_shot_num
            )
            random_religious_pos = random.sample(
                range(0, len(df_train_religious)), args.few_shot_num
            )
            random_religious_neg = random.sample(
                range(0, len(df_train_religious_neg)), args.few_shot_num
            )
            random_female_pos = random.sample(
                range(0, len(df_train_female)), args.few_shot_num
            )
            random_female_neg = random.sample(
                range(0, len(df_train_female_neg)), args.few_shot_num
            )
            random_climate_pos = random.sample(
                range(0, len(df_train_climate)), args.few_shot_num
            )
            random_climate_neg = random.sample(
                range(0, len(df_train_climate_neg)), args.few_shot_num
            )
            random_other_pos = random.sample(
                range(0, len(df_train_other)), args.few_shot_num
            )
            random_other_neg = random.sample(
                range(0, len(df_train_other_neg)), args.few_shot_num
            )

            few_shot_examples["tribal"]["pos"] = df_train_trial.iloc[random_trial_pos]
            few_shot_examples["tribal"]["neg"] = df_train_trial_neg.iloc[
                random_trial_neg
            ]
            few_shot_examples["religious"]["pos"] = df_train_religious.iloc[
                random_religious_pos
            ]
            few_shot_examples["religious"]["neg"] = df_train_religious_neg.iloc[
                random_religious_neg
            ]
            few_shot_examples["female"]["pos"] = df_train_female.iloc[random_female_pos]
            few_shot_examples["female"]["neg"] = df_train_female_neg.iloc[
                random_female_neg
            ]
            few_shot_examples["climate"]["pos"] = df_train_climate.iloc[
                random_climate_pos
            ]
            few_shot_examples["climate"]["neg"] = df_train_climate_neg.iloc[
                random_climate_neg
            ]
            few_shot_examples["other"]["pos"] = df_train_other.iloc[random_other_pos]
            few_shot_examples["other"]["neg"] = df_train_other_neg.iloc[
                random_other_neg
            ]

    if args.llm_name == "mistral":
        # Define LLM
        llm_caller = MistralAiLLMCaller(mistralai_api_key=args.mistralai_api_key, model_name=model_id_dic["mistral"], timeout=180)
    elif args.llm_name == "gpt4":
        # Define LLM
        llm_caller = OpenAILLMCaller(openai_api_key=args.openai_api_key, model_name=model_id_dic["gpt4"], timeout=180)


    # Define sampling parameters
    sampling_params = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    # evaluation
    all_gold_labels = []
    all_sys_labels = []
    evaluation_flag = True if "All Categories_DM" in df_test else False
    for i in tqdm(df_test.to_dict("records")):

        if args.few_shot_num > 0:
            i["prompt_tribal"] = []
            i["prompt_religious"] = []
            i["prompt_female"] = []
            i["prompt_climate"] = []

            i["prompt_tribal"], pos_examples1, neg_examples1 = (
                generate_few_shot_prompt_list_type(
                    i["Event Description"],
                    i["Actor 1"],
                    i["Actor 2"],
                    few_shot_examples["tribal"],
                    "tribal",
                )
            )
            i["prompt_religious"], pos_examples2, neg_examples2 = (
                generate_few_shot_prompt_list_type(
                    i["Event Description"],
                    i["Actor 1"],
                    i["Actor 2"],
                    few_shot_examples["religious"],
                    "religious",
                )
            )
            i["prompt_female"], pos_examples3, neg_examples3 = (
                generate_few_shot_prompt_list_type(
                    i["Event Description"],
                    i["Actor 1"],
                    i["Actor 2"],
                    few_shot_examples["female"],
                    "female",
                )
            )
            i["prompt_climate"], pos_examples4, neg_examples4 = (
                generate_few_shot_prompt_list_type(
                    i["Event Description"],
                    i["Actor 1"],
                    i["Actor 2"],
                    few_shot_examples["climate"],
                    "climate",
                )
            )

            input_prompt1 = databricks_llm_prompt_chat(
                dedent(system_prompt_type).strip(),
                dedent(i["prompt_tribal"][0]).strip(),
                pos_examples1,
                neg_examples1,
            )
            input_prompt2 = databricks_llm_prompt_chat(
                dedent(system_prompt_type).strip(),
                dedent(i["prompt_religious"][0]).strip(),
                pos_examples2,
                neg_examples2,
            )
            input_prompt3 = databricks_llm_prompt_chat(
                dedent(system_prompt_type).strip(),
                dedent(i["prompt_female"][0]).strip(),
                pos_examples3,
                neg_examples3,
            )
            input_prompt4 = databricks_llm_prompt_chat(
                dedent(system_prompt_type).strip(),
                dedent(i["prompt_climate"][0]).strip(),
                pos_examples4,
                neg_examples4,
            )
        else:
            pos_examples1, neg_examples1 = [], []
            pos_examples2, neg_examples2 = [], []
            pos_examples3, neg_examples3 = [], []
            pos_examples4, neg_examples4 = [], []
            i["prompt_tribal"] = format_prompt_zero_shot_tribal(
                i["Event Description"], i["Actor 1"], i["Actor 2"]
            )
            i["prompt_religious"] = format_prompt_zero_shot_religious(
                i["Event Description"], i["Actor 1"], i["Actor 2"]
            )
            i["prompt_female"] = format_prompt_zero_shot_female(
                i["Event Description"], i["Actor 1"], i["Actor 2"]
            )
            i["prompt_climate"] = format_prompt_zero_shot_climate(
                i["Event Description"], i["Actor 1"], i["Actor 2"]
            )

            input_prompt1 = databricks_llm_prompt(
                dedent(system_prompt_type).strip(), dedent(i["prompt_tribal"]).strip()
            )
            input_prompt2 = databricks_llm_prompt(
                dedent(system_prompt_type).strip(),
                dedent(i["prompt_religious"]).strip(),
            )
            input_prompt3 = databricks_llm_prompt(
                dedent(system_prompt_type).strip(), dedent(i["prompt_female"]).strip()
            )
            input_prompt4 = databricks_llm_prompt(
                dedent(system_prompt_type).strip(), dedent(i["prompt_climate"]).strip()
            )

            i["prompts"] = {
                "tribal": input_prompt1,
                "religious": input_prompt2,
                "female": input_prompt3,
                "climate": input_prompt4,
            }

        # Wait for some time between requests for Mistral.ai to avoid rate limit
        if args.llm_name == "mistral" and args.mistralai_rps:
            time.sleep(1/args.mistralai_rps)
        i["mystral_answer_tribal"] = llm_caller(input_prompt1, sampling_params)[
            0
        ].strip()
        if args.llm_name == "mistral" and args.mistralai_rps:
            time.sleep(1/args.mistralai_rps)
        i["mystral_answer_religious"] = llm_caller(input_prompt2, sampling_params)[
            0
        ].strip()
        if args.llm_name == "mistral" and args.mistralai_rps:
            time.sleep(1/args.mistralai_rps)
        i["mystral_answer_female"] = llm_caller(input_prompt3, sampling_params)[
            0
        ].strip()
        if args.llm_name == "mistral" and args.mistralai_rps:
            time.sleep(1/args.mistralai_rps)
        i["mystral_answer_climate"] = llm_caller(input_prompt4, sampling_params)[
            0
        ].strip()

        i["mystral_answer_parsed_event_type_tribal"] = (
            extract_between_tags("event_type", i["mystral_answer_tribal"])[0]
            if extract_between_tags("event_type", i["mystral_answer_tribal"])
            else "No"
        )

        i["mystral_answer_parsed_event_type_religious"] = (
            extract_between_tags("event_type", i["mystral_answer_religious"])[0]
            if extract_between_tags("event_type", i["mystral_answer_religious"])
            else "No"
        )

        i["mystral_answer_parsed_event_type_female"] = (
            extract_between_tags("event_type", i["mystral_answer_female"])[0]
            if extract_between_tags("event_type", i["mystral_answer_female"])
            else "No"
        )

        i["mystral_answer_parsed_event_type_climate"] = (
            extract_between_tags("event_type", i["mystral_answer_climate"])[0]
            if extract_between_tags("event_type", i["mystral_answer_climate"])
            else "No"
        )
        if evaluation_flag:
            label_set = set()
            gold_label = clean_label(i["All Categories_DM"])
            for one_tag in gold_label.split(","):
                label_set.add(one_tag.strip())
            all_gold_labels.append(label_set)

        sys_set = set()
        if i["mystral_answer_parsed_event_type_tribal"].strip().lower() == "yes":
            sys_set.add("Tribal/communal/ethnic conflict")
        if i["mystral_answer_parsed_event_type_religious"].strip().lower() == "yes":
            sys_set.add("Religious conflict")
        if i["mystral_answer_parsed_event_type_female"].strip().lower() == "yes":
            sys_set.add("Socio-political violence against women")
        if i["mystral_answer_parsed_event_type_climate"].strip().lower() == "yes":
            sys_set.add("Climate-related security risk")
        if len(sys_set) == 0:
            sys_set.add("Other")
        
        all_sys_labels.append(sys_set)

    return all_gold_labels, all_sys_labels

def main():
    parser = argparse.ArgumentParser(description="Event Type Classification")
    parser.add_argument("--llm_name", type=str, default="mistral", help="llm name")
    parser.add_argument(
        "--max_tokens", type=int, default=512, help="max tokens for llm"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="temperature of llm"
    )
    parser.add_argument(
        "--few_shot_num", type=int, default=0, help="number of few shot examples"
    )
    parser.add_argument(
        "--openai_api_key", type=str, default=None, help="openai api key"
    )
    parser.add_augment(
        "--mistralai_api_key", type=str, default=None, help="mistral.ai api key"
    )
    parser.add_augment(
        "--mistralai_rps", type=float, default=0, help="mistral.ai request per second limit"
    )
    args = parser.parse_args()

    # Load Data
    df_train, df_dev, df_test = load_data("../../data/data.csv")
    df_train = df_train[df_train["Is the event relevant?_DM"] == "Yes"]
    df_dev = df_dev[df_dev["Is the event relevant?_DM"] == "Yes"]
    df_test = df_test[df_test["Is the event relevant?_DM"] == "Yes"]

    # Get LLM Model Predictions 
    all_gold_labels, all_sys_labels = predict_event_type(args, df_train, df_test)
    
    # Calculate Scores
    scores = event_type_scorer_type(all_sys_labels, all_gold_labels)
    print(
        f"precision: {scores['precision'][:-1]}; recall: {scores['recall'][:-1]}; f1: {scores['f1'][:-1]}"
    )


if __name__ == "__main__":
    main()
