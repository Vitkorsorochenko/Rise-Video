import json
import os


sub_tasks = ["commonsense_knowledge", "domain_knowledge", "perceptual_knowledge", "societal_knowledge",
                "logical_capability", "experiential_knowledge", "spatial_knowledge", "temporal_knowledge"]
metric_list = ["reasoning", "consistency", "image quality", "physical rationality"]

def merge_result(root, metric_folder, write = True):
    folder = os.path.join(root, metric_folder)
    datas = []
    for dir, _, files in os.walk(folder):
        for file in files:
            if file.split('.')[1] == "json":
                json_file = os.path.join(dir, file)
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                datas.append(data)
    if write:
        out_file = os.path.join(root, f"{metric_folder}.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(datas, f, indent=4, ensure_ascii=False)
    return datas

def metric_cal(datas, metric_name, score_dict, full_set):
    score_dict[metric_name] = {}
    total_score = 0
    for data in datas:
        if metric_name == "consistency":
            single_score = (data["Final Score"]-1)/4
        elif metric_name == "reasoning":
            single_score = data["score"]
        elif metric_name == "physical rationality":
            single_score = (data["score"]-1)/4
        else: 
            single_score = (data["score"]-1)/2
        assert single_score is not None, f"{data['task_id']} {metric_name} data type error"
        total_score += single_score

    total_score = total_score * 100
    
    full_set[metric_name] = {
                            "count": len(datas),
                            "normalized score": round(total_score/len(datas), ndigits=2)}

    for sub_task in sub_tasks:
        if sub_task == "logical_capability" and metric_name == "physical rationality":
            continue
        else:
            sub_score = 0
            sub_count = 0
            for data in datas:
                if data["task_id"].split('_')[0] == sub_task[:3]:
                    if metric_name == "consistency":
                        single_score = (data["Final Score"]-1)/4
                    elif metric_name == "reasoning":
                        single_score = data["score"]
                    elif metric_name == "physical rationality":
                        single_score = (data["score"]-1)/4
                    else: 
                        single_score = (data["score"]-1)/2
                    sub_score += single_score
                    sub_count += 1
           
            sub_score = sub_score * 100
            
            score_dict[metric_name][sub_task] = {
                                                "count": sub_count,
                                                "normalized score": round(sub_score/sub_count, ndigits=2)}
   
    return score_dict, full_set


def cal_relax(result_root, save_root, save_json = True):
    
    base = os.path.basename(result_root).split('_result')[0]
    model_name = f"{save_root}/{base}"


    resoning_file = os.path.join(result_root, "temp_reasoning_score.json")
    consis_file = os.path.join(result_root, "temp_consis_result.json")
    image_q_file = os.path.join(result_root, "temp_image_quality_result.json")
    phy_ration_file = os.path.join(result_root, "temp_phy_rationality_result.json")


    with open(resoning_file, 'r', encoding='utf-8') as f:
        reasoning = json.load(f)
    with open(consis_file, 'r', encoding='utf-8') as f:
        consis = json.load(f)
    with open(image_q_file , 'r', encoding='utf-8') as f:
        image_q = json.load(f)
    with open(phy_ration_file, 'r', encoding='utf-8') as f:
        phy_ration = json.load(f)

    score_dict = {}
    full_set = {}
    
    score_dict, full_set = metric_cal(reasoning, "reasoning", score_dict, full_set)
    score_dict, full_set = metric_cal(consis, "consistency", score_dict, full_set)
    score_dict, full_set = metric_cal(image_q, "image quality", score_dict, full_set)
    score_dict, full_set = metric_cal(phy_ration, "physical rationality", score_dict, full_set)
    
    sum_dict = {}
    relax = 0.4 * full_set["reasoning"]["normalized score"] + 0.25 * full_set["consistency"]["normalized score"] + \
            0.25 * full_set["physical rationality"]["normalized score"] + 0.1 * full_set["image quality"]["normalized score"]
    sum_dict["full set"] = round(relax, ndigits = 2)
    for sub in sub_tasks:
        if sub == "logical_capability":
            sub_relax = 0.5 * score_dict["reasoning"][sub]["normalized score"] + 0.3 * score_dict["consistency"][sub]["normalized score"] + \
                 0.2 * score_dict["image quality"][sub]["normalized score"]
            
        else:
            sub_relax = 0.4 * score_dict["reasoning"][sub]["normalized score"] + 0.25 * score_dict["consistency"][sub]["normalized score"] + \
            0.25 * score_dict["physical rationality"][sub]["normalized score"] + 0.1 * score_dict["image quality"][sub]["normalized score"]
        sum_dict[sub] = round(sub_relax, ndigits = 2)


    final_data = [sum_dict, full_set, score_dict]
    
    if save_json:
        with open(f"{model_name}_score_sum.json", 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)




