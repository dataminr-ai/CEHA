from src.data_pipeline import data_pipeline
from src.db_utils import load_config, parse_args, parse_shared_config, configure_default_logger

# Configure logging
logger = configure_default_logger()


if __name__ == "__main__":
    # get config file path
    args = parse_args()
    config = load_config(args.config_path)

    # load shared config
    data_folder, start_date, end_date, data_sources, databricks_secret_scope = parse_shared_config(config)

    store_intermediate_data = config.get("data_pipeline", {}).get("store_intermediate_data", False)

    # access ACLED email and keys from Databricks secret for databricks implementation if provided
    acled_email = None
    acled_key = None
    if databricks_secret_scope:
        try:
            acled_email = dbutils.secrets.get(databricks_secret_scope, "acled_email")
            acled_key = dbutils.secrets.get(databricks_secret_scope, "acled_key")
        except Exception as e:
            logger.error(f"Secret scope {databricks_secret_scope} does not exist or secrets are not accessible.")

    # Verify the input arguments
    data_pipeline.verify_args(data_folder, start_date, end_date, data_sources, acled_email, acled_key)

    logger.info(f"Pulling data from {data_sources} from {start_date} to {end_date}. Output data will be stored in {data_folder}.")
    data_pipeline.run_data_pipeline(data_sources, start_date, end_date, data_folder, store_intermediate_data, acled_email, acled_key)

