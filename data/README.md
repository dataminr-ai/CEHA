
CEHA includes 500 events in the Horn of Africa with annotated labels by experts specialized in that area. Here are the descriptions for each field in this dataset:

- Annotator: Indicates the annotator who annotated the records.
- ACLED/GDELT: Indicates whether the event is sourced from ACLED or GDELT.
- Index: Refers to the original index from ACLED or GDELT, which can be used to retrieve additional metadata from these datasets.
- Time: Represents the timestamp of when the event occurred, sourced from ACLED or GDELT.
- Country: Indicates the country where the event occurred, extracted from ACLED or GDELT. Please refer to Section 3.2 ""Data Sampling"" in our main paper for details on how this label is extracted.
- Actor 1/Actor 2: Provides information about the primary and secondary actors involved in the event, sourced from ACLED or GDELT.
- Article URL: Contains the link to the original articles. This field is only available for events sourced from GDELT.
- Event Description: Provides a summary of the event. For ACLED, this summary is directly sourced from the dataset. For GDELT, this description is scraped from the article URL.
- Relevance Annotations:  
    - Is the event relevant?: Indicates whether the event is annotated as relevant.
    - Why is the event NOT relevant? (if applicable): Specifies reasons for why the event is not considered relevant, selected from categories as ""1. Not in Horn of Africa region"", ""2. Not violence / conflict related"", or ""3. A summary of different situations"".
- Event Type Annotations:
    - Tribal/Communal/Ethnic Conflict, Religious Conflict, Socio-political Violence Against Women, Climate-related Security Risks.
    - Other: Includes annotations for events that do not fit into any of the specified event types.
- train_dev_test_split: Indicates whether the event belongs to the training, development, or test set.