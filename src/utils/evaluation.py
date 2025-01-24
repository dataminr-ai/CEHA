def event_type_scorer(system_labels, gold_labels):
    count_system = 0
    count_gold = 0
    count_overlap = 0
    scores = {}
    for system_label, gold_label in zip(system_labels, gold_labels):
        if system_label == "Yes":
            count_system += 1

        if gold_label == "Yes":
            count_gold += 1

            if system_label == "Yes":
                count_overlap += 1
    p = count_overlap / count_system if count_system != 0 else 0.0
    r = count_overlap / count_gold if count_gold != 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0.0
    scores = {
        "precision": f"{100 * p:.2f}%",
        "recall": f"{100 * r:.2f}%",
        "f1": f"{100 * f1:.2f}%",
    }
    return scores


def event_type_scorer_type(system_labels, gold_labels):
    count_dic_system = {"overall": 0}
    count_dic_gold = {"overall": 0}
    count_dic_overlap = {"overall": 0}
    scores = {}
    for system_label, gold_label in zip(system_labels, gold_labels):
        for one_label in system_label:
            if one_label not in count_dic_system:
                count_dic_system[one_label] = 0
            count_dic_system[one_label] += 1
            count_dic_system["overall"] += 1
        for one_label in gold_label:
            if one_label not in count_dic_gold:
                print(one_label)
                count_dic_gold[one_label] = 0
            count_dic_gold[one_label] += 1
            count_dic_gold["overall"] += 1

            if one_label in system_label:
                if one_label not in count_dic_overlap:
                    count_dic_overlap[one_label] = 0
                count_dic_overlap[one_label] += 1
                count_dic_overlap["overall"] += 1
    for metric in count_dic_gold:
        count_dic_overlap[metric] = (
            0 if metric not in count_dic_overlap else count_dic_overlap[metric]
        )
        count_dic_system[metric] = (
            0 if metric not in count_dic_system else count_dic_system[metric]
        )
        p = (
            count_dic_overlap[metric] / count_dic_system[metric]
            if count_dic_system[metric] != 0
            else 0.0
        )
        r = (
            count_dic_overlap[metric] / count_dic_gold[metric]
            if count_dic_gold[metric] != 0
            else 0.0
        )
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0.0
        scores[metric] = {
            "precision": f"{100 * p:.2f}%",
            "recall": f"{100 * r:.2f}%",
            "f1": f"{100 * f1:.2f}%",
        }
    return scores
