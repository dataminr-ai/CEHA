import argparse
import logging
import yaml
from datetime import datetime, timedelta

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, help="Path to the config file")
    return parser.parse_args()

def configure_default_logger(level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.getLogger().setLevel(level)
    logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
    logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)
    logger = logging.getLogger()
    return logger

def get_last_friday():
    today = datetime.today()
    last_friday = today - timedelta(days=(today.weekday() - 4) % 7)
    previous_friday = last_friday - timedelta(days=7)
    return previous_friday, last_friday

def parse_shared_config(config):
    shared_config = config.get("shared_config", {})
    
    start_date = shared_config.get("data_info", {}).get("start_date")
    end_date = shared_config.get("data_info", {}).get("end_date")
    data_folder = shared_config.get("data_info", {}).get("data_folder")
    data_sources = shared_config.get("data_info", {}).get("data_sources", ["ACLED", "GDELT"])
    databricks_secret_scope = shared_config.get("secret_info", {}).get("databricks_secret_scope")

    # start and end date
    # Default the start date as previous Saturday and end date as last Friday when it is not provided. ACLED would update data on Monday and covers data for the last friday
    if start_date is None or end_date is None:
        previous_friday, last_friday = get_last_friday()
        start_date = datetime.strftime(previous_friday + timedelta(days=1), "%Y-%m-%d")
        end_date = datetime.strftime(last_friday, "%Y-%m-%d")  

    return data_folder, start_date, end_date, data_sources, databricks_secret_scope
