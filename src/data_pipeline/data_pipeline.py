import argparse
import pandas as pd
from datetime import datetime, timedelta
from .ACLED_data_loader import ACLED_data_loader
from .GDELT_data_loader import GDELT_data_loader

def get_last_friday():
    today = datetime.today()
    last_friday = today - timedelta(days=(today.weekday() - 4) % 7)
    previous_friday = last_friday - timedelta(days=7)
    return previous_friday, last_friday

def run_data_pipeline(data_source, start_date, end_date, data_folder, store_intermediate_data, acled_email, acled_key):
    acled_data = pd.DataFrame()
    gdelt_data = pd.DataFrame()
    for source in set(data_source):
        if source == "ACLED":
            acled_data_loader = ACLED_data_loader(acled_email, acled_key)
            acled_data = acled_data_loader.get_acled_relevant_events(start_date, end_date, data_folder, store_intermediate_data)
        if source == "GDELT":
            gdelt_data_loader = GDELT_data_loader()
            gdelt_data = gdelt_data_loader.get_gdelt_relevant_events_with_scraped_text(start_date, end_date, data_folder, store_intermediate_data)

    final_data = pd.concat([acled_data, gdelt_data], ignore_index=True)

def verify_args(data_folder, start_date, end_date, data_source, acled_email, acled_key):
    if data_folder is None:
        raise Exception("Please provide a data_folder")
    try: 
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except:
        raise Exception("Invalid date format. Date format as 'yyyy-mm-dd'.")

    for source in data_source:
        if source not in ["ACLED", "GDELT"]:
            raise Exception("Invalid data source. Please select from 'ACLED', 'GDELT'.")
    
    if "ACLED" in data_source:
        if (acled_email is None) or (acled_key is None):        
            raise Exception("ACLED data source requires ACLED login info: Email and Key")

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Data Pipeline")
    parser.add_argument("--data_folder",  type=str, default=None, help="Output folder for data and predictions",required=True)
    parser.add_argument("--start_date", type=str, default=None, help="Start date for data download. Date format as 'yyyy-mm-dd'.")
    parser.add_argument("--end_date", type=str, default=None, help="End date for data download. Date format as 'yyyy-mm-dd'. End Date is inclusive")
    parser.add_argument("--data_source", type=list, nargs='+', default=["ACLED", "GDELT"], help="Data source to download")
    parser.add_argument("--acled_email", type=str, default=None, help="ACLED login info: Email")
    parser.add_argument("--acled_key", type=str, default=None, help="ACLED login info: Key")
    parser.add_argument("--store_intermediate_data", action="store_true", help="Store intermediate data")

    args, unknown_args = parser.parse_known_args()
    start_date = args.start_date
    end_date = args.end_date
    data_folder = args.data_folder
    data_source = args.data_source
    store_intermediate_data = args.store_intermediate_data
    acled_email = args.acled_email
    acled_key = args.acled_key

    # Default the start date as previous Saturday and end date as last Friday when it is not provided
    if start_date is None or end_date is None:
        # ACLED would update data on Monday and covers data for the last friday
        previous_friday, last_friday = get_last_friday()
        start_date = datetime.strftime(previous_friday + timedelta(days=1), "%Y-%m-%d")
        end_date = datetime.strftime(last_friday, "%Y-%m-%d")  

    # check input args
    verify_args(start_date, end_date, data_source, acled_email, acled_key)

    # Data loader
    run_data_pipeline(data_source, start_date, end_date, data_folder, store_intermediate_data, acled_email, acled_key)
    





        
    

    