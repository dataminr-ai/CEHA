from textwrap import dedent

system_prompt = """\
You are a state-of-the-art event detection system. Given an news article regarding a specific event, your job is to classify if the article is relevant based on the guidelines.
"""

system_prompt_type = """\
You are a state-of-the-art event classification system. Given an news article, your job is to identify if the main event mentioned in the article can be classified as a particular event type based on the guidance.
"""


def format_prompt(document, country):
    return f"""
Guidelines:
The article is relevant if:
1. the event it describes takes place in the Horn of Africa Region, which includes Djibouti, Eritrea, Ethiopia, Kenya, Somalia, Sudan, South Sudan, or Uganda.
2. the event it describes is violent and/or occurs in a conflict setting involving or aimed at a person or people (intended to intimidate/terrorize) instead of unassociated objects or things (general expression of anger, etc.).
3. the article describes a *specific event* and is not a summary of multiple events or different events, i.e., it is not describing multiple events or developments showing trends or general information. If an article mentions more than ONE event, it is not relevant in our setting.

News Article:
{document}

This event was POSSIBLY reported in {country}.

Is this article relevant based on the guidelines? Answer "Yes" or "No" in the following format (it must be valid XML):
<response>
<answer>Answer</answer>
<reason>reason of your selection</reason>
</response>
"""


def format_prompt_few_shot(document, country):
    return f"""
Guidelines:
The article is relevant if:
1. the event it describes takes place in the Horn of Africa Region, which includes Djibouti, Eritrea, Ethiopia, Kenya, Somalia, Sudan, South Sudan, or Uganda.
2. the event it describes is violent and/or occurs in a conflict setting involving or aimed at a person or people (intended to intimidate/terrorize) instead of unassociated objects or things (general expression of anger, etc.).
3. the article describes a *specific event* and is not a summary of multiple events or different events, i.e., it is not describing multiple events or developments showing trends or general information. If an article mentions more than ONE event, it is not relevant in our setting.

News Article:
{document}

This event was POSSIBLY reported in {country}.

Is this article relevant based on the guidelines? Answer "Yes" or "No" in the following format (it must be valid XML):
<response>
<answer>Answer</answer>
</response>
"""


def format_answer(Answer):
    return f"""
<response>
<answer>{Answer}</answer>
</response>
"""


def clean_label(event_type):
    event_type = event_type.strip("; ")
    event_type_ls = event_type.split("; ")
    final_event_type_label = ", ".join(
        [i.capitalize().strip("s") for i in event_type_ls]
    )
    return final_event_type_label


def generate_few_shot_prompt_list(document, country, examples):
    pos_examples = []
    neg_examples = []
    for i in examples["pos"].to_dict("records"):
        Answer = "Yes"
        pos_examples.append(
            [
                format_prompt_few_shot(i["Event Description"], i["Country"]),
                format_answer(Answer),
            ]
        )

    for i in examples["neg"].to_dict("records"):
        Answer = "No"
        neg_examples.append(
            [
                format_prompt_few_shot(i["Event Description"], i["Country"]),
                format_answer(Answer),
            ]
        )

    input_prompt = ""
    input_prompt = format_prompt_few_shot(document, country)
    return input_prompt, pos_examples, neg_examples


def databricks_llm_prompt(system_prompt, user_prompt):
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": user_prompt},
    ]


def databricks_llm_prompt_chat(system_prompt, user_prompt, pos_examples, neg_examples):
    output = []
    output.append(
        {
            "role": "system",
            "content": system_prompt,
        }
    )
    for pos_example, neg_example in zip(pos_examples, neg_examples):
        output.append(
            {
                "role": "user",
                "content": dedent(pos_example[0]).strip(),
            }
        )
        output.append(
            {
                "role": "assistant",
                "content": dedent(pos_example[1]).strip(),
            }
        )
        output.append(
            {
                "role": "user",
                "content": dedent(neg_example[0]).strip(),
            }
        )
        output.append(
            {
                "role": "assistant",
                "content": dedent(neg_example[1]).strip(),
            }
        )
    output.append(
        {"role": "user", "content": user_prompt},
    )
    return output


def format_prompt_zero_shot_female(document, actor1, actor2):
    return f"""
Guidance:
A Socio-political violence against women is civilian targeting event in which women and/or girls are the ‘target’ of the violence.
An event should be categorized as socio-political violence against women when:
- The victim(s) of the event are composed entirely of women/girls, or when the majority of victims are women/girls.
- The primary target was a woman/girl (e.g. a female politician attacked alongside two men working as bodyguards).
NOTE:
- DO NOT identify it as a socio-political violence against women event if the targeting of women or girls has the potential to be random. (Ex. woman killed while in a car that ran over an IED).

News Article:
{document}

Event Actors: 
{actor1};{actor2}

Is the main event mentioned in the news article can be classified as a socio-political violence against women? Answer "Yes" or "No" in the following format (it must be valid XML):
<response>
<event_type>Answer</event_type>
<reason>reason of your selection</reason>
</response>
"""


def format_prompt_zero_shot_climate(document, actor1, actor2):
    return f"""
Guidance:
A climate-related security risk is a conflict event (like migration-induced conflict or pastoral conflict) influenced by environmental and climate-related factors.
These events should have two components: 1) a climate related phenomenon and 2) a conflict event, both of which are explicitly stated.
An event should be categorized as climate-related security risk when:
- A conflict event is influenced by environmental or climate related factors, which include, but are not limited to: drought, desertification, temperature rise, flooding.
- Conflict event centers around resources that have become limited due to environmental and climate-related factors, such as water access, grazing land, farmland, etc.
NOTE:
- DO NOT identify it as a climate-related security risk event if that is not directly influenced by climate-related factors, such as climate change protests.

News Article:
{document}

Event Actors: 
{actor1};{actor2}

Is the main event mentioned in the news article can be classified as a climate-related security risk? Answer "Yes" or "No" in the following format (it must be valid XML):
<response>
<event_type>Answer</event_type>
<reason>reason of your selection</reason>
</response>
"""


def format_prompt_zero_shot_religious(document, actor1, actor2):
    return f"""
Guidance:
An event should be categorized as religious conflict as long as it meets any of the following requirements:
- Religion-related entity invlove in the conflict, which include, but are not limited to religious leaders, reglious military groups and religious staff; OR
- The conflict targets individuals who engage in religious practice or expressing their religious belief (e.g. pastor), no matter if the conflict itself is religiously motivated or not; OR
- It involves the enforcement of specific religious norms to force or prevent actions; OR
- The conflict happend at a religious institution.
NOTE:
- An event should be categoried as religious conflict when it meets any one of the above requirements.
- ALWAYS identity it as a religious conflict when military groups such as Al Shabaab and ISIS are involved.
- An event may also be categoried as a religous conflict  even though the conflict was not religiously motivated or targeted.
- DO NOT identify it as a religious conflict if it is explicitly mentioned in the article that the religious group / institution / person is a random target rather than a specific target. (Ex. mortar fire hits church in addition to many other nearby targets).

News Article:
{document}

Event Actors: 
{actor1};{actor2}

Is the main event mentioned in the news article can be classified as a religious conflict based on the guidance? Answer "Yes" or "No" in the following format (it must be valid XML):
<response>
<event_type>Answer</event_type>
<reason>reason of your selection</reason>
</response>
"""


def format_prompt_zero_shot_tribal(document, actor1, actor2):
    return f"""
Guidance:
A tribal/communal/ethnic conflict is a dispute or violence involving ethnic, tribal, OR communal individuals/groups.
An event should be categorized as tribal/communal/ethnic conflict when:
- It falls into ANY of the following categories: tribal (including clans) OR communal OR ethnic.
NOTE:
- Disputes or violence can be one-sided from ethnic, tribal (including clans), OR communal individuals/groups.
- If the actor names are confirmed rather than presumed, please reference them to categorize a tribal/communal/ethnic conflict.
- DO NOT make conclusions based on presumed information.

News Article:
{document}

Event Actors: 
{actor1};{actor2}

Is the main event mentioned in the news article can be classified as a tribal/communal/ethnic conflict? Answer "Yes" or "No" in the following format (it must be valid XML):
<response>
<event_type>Answer</event_type>
<reason>reason of your selection</reason>
</response>
"""


def generate_few_shot_prompt_list_type(document, actor1, actor2, examples, event):
    pos_examples = []
    neg_examples = []
    for i in examples["pos"].to_dict("records"):
        sample = {
            "index": i["Index"],
            "event_description": i["Event Description"],
            "actor1": i["Actor 1"],
            "actor2": i["Actor 2"],
            "label": clean_label(i["All Categories_DM"]),
            "country": i["Country"],
            "source": i["ACLED/GDELT"],
        }
        # print(sample)
        Answer = "Yes"
        if event == "female":
            pos_examples.append(
                [
                    format_prompt_few_shot_female(
                        i["Event Description"], i["Actor 1"], i["Actor 2"]
                    ),
                    format_answer_type(Answer),
                ]
            )
        elif event == "climate":
            pos_examples.append(
                [
                    format_prompt_few_shot_climate(
                        i["Event Description"], i["Actor 1"], i["Actor 2"]
                    ),
                    format_answer_type(Answer),
                ]
            )
        elif event == "religious":
            pos_examples.append(
                [
                    format_prompt_few_shot_religious(
                        i["Event Description"], i["Actor 1"], i["Actor 2"]
                    ),
                    format_answer_type(Answer),
                ]
            )
        elif event == "tribal":
            pos_examples.append(
                [
                    format_prompt_few_shot_tribal(
                        i["Event Description"], i["Actor 1"], i["Actor 2"]
                    ),
                    format_answer_type(Answer),
                ]
            )

    for i in examples["neg"].to_dict("records"):
        sample = {
            "index": i["Index"],
            "event_description": i["Event Description"],
            "actor1": i["Actor 1"],
            "actor2": i["Actor 2"],
            "label": clean_label(i["All Categories_DM"]),
            "country": i["Country"],
            "source": i["ACLED/GDELT"],
        }
        # print(sample)
        Answer = "No"
        if event == "female":
            neg_examples.append(
                [
                    format_prompt_few_shot_female(
                        i["Event Description"], i["Actor 1"], i["Actor 2"]
                    ),
                    format_answer_type(Answer),
                ]
            )
        elif event == "climate":
            neg_examples.append(
                [
                    format_prompt_few_shot_climate(
                        i["Event Description"], i["Actor 1"], i["Actor 2"]
                    ),
                    format_answer_type(Answer),
                ]
            )
        elif event == "religious":
            neg_examples.append(
                [
                    format_prompt_few_shot_religious(
                        i["Event Description"], i["Actor 1"], i["Actor 2"]
                    ),
                    format_answer_type(Answer),
                ]
            )
        elif event == "tribal":
            neg_examples.append(
                [
                    format_prompt_few_shot_tribal(
                        i["Event Description"], i["Actor 1"], i["Actor 2"]
                    ),
                    format_answer_type(Answer),
                ]
            )

    input_prompt = ""

    if event == "female":
        input_prompt = format_prompt_few_shot_female(document, actor1, actor2)
    elif event == "climate":
        input_prompt = format_prompt_few_shot_climate(document, actor1, actor2)
    elif event == "religious":
        input_prompt = format_prompt_few_shot_religious(document, actor1, actor2)
    elif event == "tribal":
        input_prompt = format_prompt_few_shot_tribal(document, actor1, actor2)
    return input_prompt, pos_examples, neg_examples


def format_answer_type(Answer):
    return f"""
<response>
<event_type>{Answer}</event_type>
</response>
"""


def format_prompt_few_shot_female(document, actor1, actor2):
    return f"""
Guidance:
A Socio-political violence against women is civilian targeting event in which women and/or girls are the ‘target’ of the violence.
An event should be categorized as socio-political violence against women when:
- The victim(s) of the event are composed entirely of women/girls, or when the majority of victims are women/girls.
- The primary target was a woman/girl (e.g. a female politician attacked alongside two men working as bodyguards).
NOTE:
- DO NOT identify it as a socio-political violence against women event if the targeting of women or girls has the potential to be random. (Ex. woman killed while in a car that ran over an IED).

News Article:
{document}

Event Actors: 
{actor1};{actor2}

Is the main event mentioned in the news article can be classified as a socio-political violence against women? Answer "Yes" or "No" in the following format (it must be valid XML):
<response>
<event_type>Answer</event_type>
</response>
"""


def format_prompt_few_shot_climate(document, actor1, actor2):
    return f"""
Guidance:
A climate-related security risk is a conflict event (like migration-induced conflict or pastoral conflict) influenced by environmental and climate-related factors.
These events should have two components: 1) a climate related phenomenon and 2) a conflict event, both of which are explicitly stated.
An event should be categorized as climate-related security risk when:
- A conflict event is influenced by environmental or climate related factors, which include, but are not limited to: drought, desertification, temperature rise, flooding.
- Conflict event centers around resources that have become limited due to environmental and climate-related factors, such as water access, grazing land, farmland, etc.
NOTE:
- DO NOT identify it as a climate-related security risk event if that is not directly influenced by climate-related factors, such as climate change protests.

News Article:
{document}

Event Actors: 
{actor1};{actor2}

Is the main event mentioned in the news article can be classified as a climate-related security risk? Answer "Yes" or "No" in the following format (it must be valid XML):
<response>
<event_type>Answer</event_type>
</response>
"""


def format_prompt_few_shot_religious(document, actor1, actor2):
    return f"""
Guidance:
An event should be categorized as religious conflict as long as it meets any of the following requirements:
- Religion-related entity invlove in the conflict, which include, but are not limited to religious leaders, reglious military groups and religious staff; OR
- The conflict targets individuals who engage in religious practice or expressing their religious belief (e.g. pastor), no matter if the conflict itself is religiously motivated or not; OR
- It involves the enforcement of specific religious norms to force or prevent actions; OR
- The conflict happend at a religious institution.
NOTE:
- An event should be categoried as religious conflict when it meets any one of the above requirements.
- ALWAYS identity it as a religious conflict when military groups such as Al Shabaab and ISIS are involved.
- An event may also be categoried as a religous conflict  even though the conflict was not religiously motivated or targeted.
- DO NOT identify it as a religious conflict if it is explicitly mentioned in the article that the religious group / institution / person is a random target rather than a specific target. (Ex. mortar fire hits church in addition to many other nearby targets).

News Article:
{document}

Event Actors: 
{actor1};{actor2}

Is the main event mentioned in the news article can be classified as a religious conflict based on the guidance? Answer "Yes" or "No" in the following format (it must be valid XML):
<response>
<event_type>Answer</event_type>
</response>
"""


def format_prompt_few_shot_tribal(document, actor1, actor2):
    return f"""
Guidance:
A tribal/communal/ethnic conflict is a dispute or violence involving ethnic, tribal, OR communal individuals/groups.
An event should be categorized as tribal/communal/ethnic conflict when:
- It falls into ANY of the following categories: tribal (including clans) OR communal OR ethnic.
NOTE:
- Disputes or violence can be one-sided from ethnic, tribal (including clans), OR communal individuals/groups.
- If the actor names are confirmed rather than presumed, please reference them to categorize a tribal/communal/ethnic conflict.
- DO NOT make conclusions based on presumed information.

News Article:
{document}

Event Actors: 
{actor1};{actor2}

Is the main event mentioned in the news article can be classified as a tribal/communal/ethnic conflict? Answer "Yes" or "No" in the following format (it must be valid XML):
<response>
<event_type>Answer</event_type>
</response>
"""
