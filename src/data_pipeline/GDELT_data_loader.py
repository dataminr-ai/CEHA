import gdelt
import os
import pandas as pd
import pickle
import re
import time
from datetime import datetime, timedelta
from newspaper import Article
from multiprocessing import Pool

import warnings
warnings.filterwarnings("ignore")

HORN_OF_AFRICA_COUNTRY_LIST = ["Djibouti", "Eritrea", "Ethiopia", "Somalia", "Kenya", "Sudan", "South Sudan", "Uganda"]
HORN_OF_AFRICA_COUNTRY_CODES = ['DJ', 'ER', 'ET', 'SO', 'KE', 'SU', 'OD', 'UG'] #https://en.wikipedia.org/wiki/List_of_FIPS_country_codes
CAMEO_3CHAR_COUNTRY_CODES = ['DJI', 'ERI', 'ETH', 'SOM', 'KEN', 'SDN', 'SSD', 'UGA']

def news_web_scraping(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        title = article.title
        text = article.text
        meta_description = None
    except:
        title, text, meta_description = None, None, None
    return url, title, text, meta_description

class GDELT_data_loader():
    def __init__(self):

        # Version 2 queries
        self.gd2 = gdelt.gdelt(version=2)
        self.meaningless_text = [
            "",
            "Please click here to view our site optimised for your device.",
            "Crédits\n\nAgence Olloweb : Etude et conduite de projet, prototypage, graphisme, codage de l'environnement graphique - Publinoves communication : Logotype - Orthographe Plus : Rédactionnels - SiteGround : Hébergement - Google : Statistiques, polices de caractères - Développement : HTML5, CSS3, JQUERY, PHP7 - Applications : Adobe® Photoshop®, Adobe® Illustrator®, Panic® Nova® - Année/projet: 2022-2023",
            "Legal Disclaimer:\n\nMENAFN provides the information “as is” without warranty of any kind. We do not accept any responsibility or liability for the accuracy, content, images, videos, licenses, completeness, legality, or reliability of the information contained in this article. If you have any complaints or copyright issues related to this article, kindly contact the provider above.",
            "Email\n\nPassword\n\nRemember me\n\nBy clicking on the login button, you agree to our Terms & Conditions and the Privacy Policy",
            "Facebook users\n\nUse your Facebook account to login or register with JapanToday. By doing so, you will also receive an email inviting you to receive our news alerts.",
            "Dear Reader,\n\nThis section is about Living in UAE and essential information you cannot live without.\n\nRegister to read and get full access to gulfnews.com",
            "Error:\n\nJavascript is disabled in this browser. This page requires Javascript. Modify your browser's settings to allow Javascript to execute. See your browser's documentation for specific instructions.",
            "Recipes\n\nWhat do you want to cook today?",
            "This website uses cookies to improve your experience. We'll assume you're ok with this, but you can opt-out if you wish. Cookie settings ACCEPT",
            "The Standard Group Plc is a multi-media organization with investments in media platforms spanning newspaper print operations, television, radio broadcasting, digital and online services. The Standard Group is recognized as a leading multi-media house in Kenya with a key influence in matters of national and international interest.",
            """Close Get email notifications on {{subject}} daily!\n\nYour notification has been saved.\n\nThere was a problem saving your notification.\n\n{{description}}\n\nEmail notifications are only sent once a day, and only if there are new matching items""",
            "To enjoy our website, you'll need to enable JavaScript in your web browser. Please click here to learn how."   
        ]

    def get_gdelt_data(self, start_date, end_date):
        """
        Args:
            start_date: String. Format as "YYYY MM dd", e.g. 2024 01 01
            end_date: String. Format as "YYYY MM dd", e.g. 2024 01 04
        """
        # Full day pull, output to pandas dataframe, events table
        # translation=False  for English only posts. If non-english posts are requested, Modify to "translation=True". 
        # The database for English only posts and non-English posts don't overlap
        # table: 'events' or 'mentions' or 'gkg'
        if start_date == end_date:
            events = self.gd2.Search(start_date,table='events',coverage=True, translation=False)
        else:
            events = self.gd2.Search([start_date, end_date],table='events',coverage=True, translation=False)
        return events

    def country_identifier(self, df):
        """
        Identify if the country of the event is in Horn of Africa. If the event is not in Horn of Africa, returns empty string

        The filters follows by the instruction in GDELT codebook page 6, which suggests two methods: filter by Actor/action geo code or the actor code
        """
        matched_countries = []
        for country, fips_code, cameo_code in zip(HORN_OF_AFRICA_COUNTRY_LIST, HORN_OF_AFRICA_COUNTRY_CODES, CAMEO_3CHAR_COUNTRY_CODES):
            if ((df['Actor1CountryCode'] == cameo_code)
            or (df['Actor2CountryCode'] == cameo_code)
            or (df['Actor1Geo_CountryCode'] == fips_code)
            or (df['Actor2Geo_CountryCode'] == fips_code)
                or (df['ActionGeo_CountryCode'] == fips_code)):
                matched_countries.append(country)
        return ";".join(matched_countries)

    def _transform_event_code(self, i):
        i = str(i)
        if len(i) == 2:
            i = "0" + i + "0"
        elif len(i) == 3:
            if int(i[:2]) <=20:
                i = i + "0"
            else:
                i = "0" + i
        return i


    def filter_events(self, events):
        """
        Filter for geo location and event type
        """
        # filter for Horn of Africa. 
        events["inferred_country"] = events.apply(self.country_identifier, axis=1)
        events = events[events["inferred_country"]!= ""]
        # fix wrong geo location assignment for red sea --> Djibouti
        events = events[~(((events["Actor1Geo_FullName"].apply(lambda s: "Red Sea" in s if pd.notnull(s) else False)) |
                            (events["Actor2Geo_FullName"].apply(lambda s: "Red Sea" in s if pd.notnull(s) else False)) |
                            (events["ActionGeo_FullName"].apply(lambda s: "Red Sea" in s if pd.notnull(s) else False)))
                        & (events["inferred_country"] == "Djibouti"))]
        # filter for violent events based on CAMEO event code
        events['EventCode'] = events['EventCode'].apply(lambda i: self._transform_event_code(i))
        events = events[events["EventCode"] > "1000"]
        return events


    def get_gdelt_relevant_events(self, start_date, end_date):
        """
        Args:
            start_date: String. Format as "YYYY-MM-dd", e.g. 2024-01-01
            end_date: String. Format as "YYYY-MM-dd", e.g. 2024-01-04
        """
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        diff_days = (end_date - start_date).days + 1
        batch_size = 5
        
        # Get data from GDELT every five days in consideration of time
        events_horn_of_africa_list = []
        for i in range(diff_days//batch_size + 1):
            s_day = start_date + timedelta(days=batch_size*i)
            e_day = min(start_date + timedelta(days=batch_size*(i+1) -1), end_date)
            batch_days = (e_day - s_day).days + 1
            s_day_str = datetime.strftime(s_day, "%Y %m %d")
            e_day_str = datetime.strftime(e_day, "%Y %m %d")
            if e_day >= s_day:
                print(f"Starting loading data from {s_day_str} for {batch_days} days")
                # get event data from GDELT
                events = self.get_gdelt_data(s_day_str, e_day_str)
                # re-filter for date
                events = events[events["SQLDATE"].apply(lambda s: (datetime.strptime(str(s), "%Y%m%d") >=s_day) and (datetime.strptime(str(s), "%Y%m%d") <=e_day))]
                # filter for possible relevant events
                events = self.filter_events(events)
                events_horn_of_africa_list.append(events)

        merged_df = pd.concat(events_horn_of_africa_list)

        return merged_df
        

    
    def get_gdelt_relevant_events_with_scraped_text(self, start_date, end_date, data_folder, store_intermediate_data=False):
        """
        Get GDELT Data from online database, apply related filters and merge with scraped text

        Args:
            start_date: String. Format as "YYYY-MM-dd", e.g. 2024-01-01
            end_date: String. Format as "YYYY-MM-dd", e.g. 2024-01-04 (Inclusive)
            data_folder: String, base data folder to save data
            store_intermediate_data: Boolean. Whether to store intermediate data or not.
        """
        intermediate_data_folder = os.path.join(data_folder, "intermediate_data", "gdelt")
        final_data_folder = os.path.join(data_folder, "final_data_for_classification")
        os.makedirs(intermediate_data_folder, exist_ok=True)
        os.makedirs(final_data_folder, exist_ok=True)
        final_output_filepath = os.path.join(final_data_folder, f'gdelt_{start_date}_{end_date}.csv')

        s_time = time.time()
        # get events
        print("==" * 30)
        print(f"Start getting GDLET events from {start_date} to {end_date}")
        events = self.get_gdelt_relevant_events(start_date, end_date)
        print(f"Get GDLET events Completed. It takes {int((time.time() - s_time)/60)} minutes.")
        if store_intermediate_data:
            with open(f"{intermediate_data_folder}/gdelt_events_table_{start_date}_{end_date}.pkl", "wb") as f_w:
                pickle.dump(events, f_w)

        # scrape based on urls
        urls = list(events["SOURCEURL"].unique())
        print(f"Start web scraping for {len(urls)} urls")
        s_time = time.time()
        p = Pool(8)
        scraped_results = p.map(news_web_scraping, urls)
        p.terminate()
        p.join()
        print(f"Scraped text Completed. It takes {int((time.time() - s_time)/60)} minutes.")
        if store_intermediate_data:
            with open(f"{intermediate_data_folder}/gdelt_web_scrape_{start_date}_{end_date}.pkl", "wb") as f_w:
                pickle.dump(scraped_results, f_w)
        
        # merge data 
        scraped_df = pd.DataFrame(scraped_results)
        scraped_df.columns = ["url", "title", "text", "metadata_description"]
        scraped_df = scraped_df[~scraped_df["text"].isin(self.meaningless_text)]
        scraped_df = scraped_df.dropna(subset=["title", "text"]).drop_duplicates(subset = "url")
        merged_df = events.merge(scraped_df, left_on="SOURCEURL", right_on="url").drop_duplicates()
        with open(f"{intermediate_data_folder}/gdelt_data_{start_date}_{end_date}.pkl", "wb") as f_w:
            pickle.dump(merged_df, f_w)

        # Only save unique articles and text for first five paragraphs
        gdelt = merged_df.drop_duplicates(subset=['url'])
        gdelt["text"] = gdelt["text"].apply(lambda s: "\n".join(re.sub(r"\n+", "\n", s).split("\n")[:5]))
        gdelt["SQLDATE"] = gdelt["SQLDATE"].apply(lambda s: datetime.strftime(datetime.strptime(str(s), "%Y%m%d"), "%Y-%m-%d"))
        gdelt["ACLED/GDELT"] = "GDELT"
        gdelt = gdelt.rename(columns={
            "GLOBALEVENTID": "Index",
            "SQLDATE": "Time",
            "inferred_country": "Country",
            "Actor1Name": "Actor 1",
            "Actor2Name": "Actor 2",
            "url": "Article URL",
            "text": "Event Description"
        })
        cols = ["ACLED/GDELT", "Index", "Time", "Country", "Actor 1", "Actor 2", "Article URL", "Event Description"]
        gdelt[cols].to_csv(final_output_filepath, index=False)
        print(f"Finally, we got {len(gdelt)} GDELT data for classifications")
        print(f"Data is saved to {final_output_filepath}")
        print("==" * 30)
        return gdelt[cols]
