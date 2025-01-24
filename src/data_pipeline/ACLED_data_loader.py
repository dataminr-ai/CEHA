import requests
import os
import pandas as pd
import pickle

HORN_OF_AFRICA_COUNTRY_LIST = ["Djibouti", "Eritrea", "Ethiopia", "Kenya", "Somalia", "Sudan", "Uganda", "South Sudan"]

class ACLED_data_loader():
    """
    email: String. Email for ACLED account
    key: String. Api key for ACLED account
    """
    def __init__(self, email, key):
        self.email = email
        self.key = key

    def get_acled_data(self, start_date, end_date):
        """
        Get ACLED Data from online database for Horn of Africa within given time range
        """

        parameters = {
            "email": self.email,
            "key": self.key,
            "country": "|".join(HORN_OF_AFRICA_COUNTRY_LIST), 
            "event_date": f"{start_date}|{end_date}", 
            "event_date_where": "BETWEEN"
        }

        response_params_dic = requests.get("https://api.acleddata.com/acled/read", params= parameters)
        if response_params_dic.json()['status'] == 200:
            acled = pd.DataFrame(response_params_dic.json()['data'])
            return acled
        else:
            raise ValueError("Failed to get ACLED data")

    def filter_events(self, df):
        """
        Filter ACLED data for relevant event types.
        Geo filter and time filter are automatically applied when requesting data.
        """
        df = df[~df["sub_event_type"].isin(["Agreement",  "Peaceful protest", "Non-violent transfer of territory"])]
        return df

    
    def get_acled_relevant_events(self, start_date, end_date, data_folder, store_intermediate_data=False):
        """
        Get ACLED Data from online database and apply related filters

        Args:
            start_date: String. Format as "YYYY-MM-dd", e.g. 2024-01-01
            end_date: String. Format as "YYYY-MM-dd", e.g. 2024-01-04 (Inclusive)
            data_folder: String, base data folder to save data
            store_intermediate_data: Boolean. Whether to store intermediate data or not.
        """
        intermediate_data_folder = os.path.join(data_folder, "intermediate_data", "acled")
        final_data_folder = os.path.join(data_folder, "final_data_for_classification")
        os.makedirs(intermediate_data_folder, exist_ok=True)
        os.makedirs(final_data_folder, exist_ok=True)
        final_output_filepath = os.path.join(final_data_folder, f'acled_{start_date}_{end_date}.csv')

        # output columns
        cols = ["ACLED/GDELT", "Index", "Time", "Country", "Actor 1", "Actor 2", "Event Description"]

        # get events
        print("==" * 30)
        print(f"Start getting ACLED events from {start_date} to {end_date}")
        # get event data from ACLED
        events = self.get_acled_data(start_date, end_date)
        if len(events) > 0:
            # filter for possible relevant events
            events = self.filter_events(events)
            print(f"Get ACLED events Completed.\n")
            if store_intermediate_data:
                with open(f"{intermediate_data_folder}/acled_data_{start_date}_{end_date}.pkl", "wb") as f_w:
                    pickle.dump(events, f_w)

            # Save data for annotation
            acled = events.drop_duplicates(subset=['event_id_cnty'])
            acled["actor1_merged"] = acled.apply(lambda df: "; ".join([df["actor1"], df["assoc_actor_1"]]) if pd.notnull(df["assoc_actor_1"]) and (df["assoc_actor_1"] != "") else df["actor1"], axis=1)
            acled["actor2_merged"] = acled.apply(lambda df: "; ".join([df["actor2"], df["assoc_actor_2"]]) if pd.notnull(df["assoc_actor_2"]) and (df["assoc_actor_2"] != "") else df["actor2"], axis=1)
            acled["ACLED/GDELT"] = "ACLED"
            acled = acled.rename(columns={
                "event_id_cnty": "Index",
                "event_date": "Time",
                "country": "Country",
                "actor1_merged": "Actor 1",
                "actor2_merged": "Actor 2",
                # "url": "Article URL",
                "notes": "Event Description"
            })
        else:
            print("No data found for this time range.")
            acled = pd.DataFrame(columns=cols)
            
        
        acled[cols].to_csv(final_output_filepath, index=False)
        print(f"Finally, we got {len(acled)} ACLED data for classifications")
        print(f"Data is saved to {final_output_filepath}")
        print("==" * 30)
        return acled[cols]


    
