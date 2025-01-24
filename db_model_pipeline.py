import argparse
import json
import os
import pandas as pd
import numpy as np
from src.db_utils import load_config, parse_args, parse_shared_config, configure_default_logger
from src.utils.utils import load_data
from src.classification_pipeline.event_relevance_classification import predict_event_relevance
from src.classification_pipeline.event_type_classification import predict_event_type

# Configure logging
logger = configure_default_logger()

def load_secrets(databricks_secret_scope):
    secret_dict = {}
    potential_secrets_keys = ["openai_api_key", "mistralai_api_key"]
    if databricks_secret_scope in [scope.name for scope in dbutils.secrets.listScopes()]:
        secrets = dbutils.secrets.list(databricks_secret_scope)
        existing_secret_keys = [secret.key for secret in secrets]
        for secret_key in potential_secrets_keys:
            if secret_key in existing_secret_keys:
                secret_dict[secret_key] = dbutils.secrets.get(databricks_secret_scope, secret_key)
        logger.info(f"Loaded secrets: {list(secret_dict.keys())}")
    else:
        logger.error("Secret scope does not exist.")
    return secret_dict


def valid_model_configs(llm, few_shot_num, train_example_path, secret_dict, output_folder):
    if few_shot_num > 0 and not train_example_path:
        raise ValueError("train_example_path must be provided if few_shot_num > 0")
    
    if llm == "gpt4" and "openai_api_key" not in secret_dict:
        raise ValueError("openai_api_key must be provided in secret_dict for gpt4 model")
    elif llm == "mistral" and "mistralai_api_key" not in secret_dict:
        raise ValueError(f"mistralai_api_key must be provided in secret_dict for mistral model")
    
    if not output_folder:
        raise ValueError("output_folder must be provided")


def run_event_relevance_classification(df_test, secret_dict, llm, max_tokens=512, temperature=0, few_shot_num=0, train_example_path=None, mistralai_rps=0):
    # load train data
    if train_example_path:
        df_train, _, _ = load_data(train_example_path)
    else:
        df_train = pd.DataFrame()
    # run event relevance model
    event_relevance_args = argparse.Namespace(
        llm_name=llm,
        max_tokens=max_tokens,
        temperature=temperature,
        few_shot_num=few_shot_num,
        openai_api_key=secret_dict.get("openai_api_key"),
        mistralai_api_key=secret_dict.get("mistralai_api_key"),
        mistralai_rps=mistralai_rps,
    )
    _, event_relevance_prediction = predict_event_relevance(event_relevance_args, df_train, df_test)

    return event_relevance_prediction
    
def run_event_type_classification(df_test, secret_dict, llm, max_tokens=512, temperature=0, few_shot_num=0, train_example_path=None, mistralai_rps=0):
    # load train data
    if train_example_path:
        df_train, _, _ = load_data(train_example_path)
    else:
        df_train = pd.DataFrame()

    # run event type model
    event_type_args = argparse.Namespace(
        llm_name=llm,
        max_tokens=max_tokens,
        temperature=temperature,
        few_shot_num=few_shot_num,
        openai_api_key=secret_dict.get("openai_api_key"),
        mistralai_api_key=secret_dict.get("mistralai_api_key"),
        mistralai_rps=mistralai_rps,
    )
    
    _, event_type_prediction = predict_event_type(event_type_args, df_train, df_test)

    return event_type_prediction

    
if __name__ == "__main__":
    # get config file path
    args = parse_args()
    config = load_config(args.config_path)

    # load shared config
    data_folder, start_date, end_date, data_sources, databricks_secret_scope = parse_shared_config(config)

    # load secrets
    secret_dict = load_secrets(databricks_secret_scope)
    
    # load output folder
    output_folder = config.get("model_pipeline", {}).get("output_folder")

    # load event relevance classification config
    event_relevance_classification_config = config.get("model_pipeline", {}).get("event_relevance_classification", {})
    event_relevance_llm = event_relevance_classification_config.get("llm_name")
    event_relevance_few_shot_num = event_relevance_classification_config.get("few_shot_num", 0)
    event_relevance_train_example_path = event_relevance_classification_config.get("train_example_path")
    event_relevance_max_tokens = event_relevance_classification_config.get("max_tokens", 512)
    event_relevance_temperature = event_relevance_classification_config.get("temperature", 0)
    event_relevance_mistralai_rps = event_relevance_classification_config.get("mistralai_rps", 0)
    valid_model_configs(event_relevance_llm, event_relevance_few_shot_num, event_relevance_train_example_path, secret_dict, output_folder)
    logger.info(f"Selected models for Event Relevance Classification: -LLM: {event_relevance_llm}\n -few_shot_num: {event_relevance_few_shot_num}\n -train_example_path: {event_relevance_train_example_path}\n -max_tokens: {event_relevance_max_tokens}\n -temperature: {event_relevance_temperature}")
    
    # load event type classification config
    event_type_classification_config = config.get("model_pipeline", {}).get("event_type_classification", {})
    event_type_llm = event_type_classification_config.get("llm_name")
    event_type_few_shot_num = event_type_classification_config.get("few_shot_num", 0)
    event_type_train_example_path = event_type_classification_config.get("train_example_path")
    event_type_max_tokens = event_type_classification_config.get("max_tokens", 512)
    event_type_temperature = event_type_classification_config.get("temperature", 0)
    event_type_mistralai_rps = event_type_classification_config.get("mistralai_rps", 0)
    valid_model_configs(event_type_llm, event_type_few_shot_num, event_type_train_example_path, secret_dict, output_folder)
    logger.info(f"Selected models for Event Type Classification: -LLM: {event_type_llm}\n -few_shot_num: {event_type_few_shot_num}\n -train_example_path: {event_type_train_example_path}\n -max_tokens: {event_type_max_tokens}\n -temperature: {event_type_temperature}")

    # load test data 
    if "ACLED" in data_sources:
        acled_data = pd.read_csv(os.path.join(data_folder, "final_data_for_classification", f"acled_{start_date}_{end_date}.csv"))
    else:
        acled_data = pd.DataFrame()
    if "GDELT" in data_sources:
        gdelt_data = pd.read_csv(os.path.join(data_folder, "final_data_for_classification", f"gdelt_{start_date}_{end_date}.csv"))
    else:
        gdelt_data = pd.DataFrame()
    final_test_data = pd.concat([acled_data, gdelt_data], ignore_index=True)

    # event relevance classification
    event_relevance_prediction = run_event_relevance_classification(final_test_data, secret_dict, event_relevance_llm, max_tokens=event_relevance_max_tokens, temperature=event_relevance_temperature, few_shot_num=event_relevance_few_shot_num, train_example_path=event_relevance_train_example_path, mistralai_rps=event_relevance_mistralai_rps)
    final_test_data["event_relevance_prediction"] = event_relevance_prediction

    # event type classification on relevant event
    relevant_events = final_test_data[final_test_data["event_relevance_prediction"] == "relevant"]
    event_type_prediction = run_event_type_classification(relevant_events, secret_dict, event_type_llm, max_tokens=event_type_max_tokens, temperature=event_type_temperature, few_shot_num=event_type_few_shot_num, train_example_path=event_type_train_example_path, mistralai_rps=event_type_mistralai_rps)
    final_test_data["event_type_prediction"] = np.nan
    final_test_data.loc[relevant_events.index, "event_type_prediction"] = event_type_prediction

    # save results
    os.makedirs(output_folder, exist_ok=True)
    final_test_data.to_csv(os.path.join(output_folder, f"{'_'.join(data_sources).lower()}_{start_date}_{end_date}_with_predictions.csv"), index=False)


