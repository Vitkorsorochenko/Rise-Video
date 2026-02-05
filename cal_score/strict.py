import json
import os


def load_datas(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    return datas

def cal_strict(root, save_root, save_full_score = True):
   
    base = os.path.basename(root).split('_result')[0]
    model_name = f"{save_root}/{base}"

    reasoning_score_file = os.path.join(root, "temp_reasoning_score.json")
    consis_score_file = os.path.join(root, "temp_consis_result.json")
    phy_score_file = os.path.join(root, "temp_phy_rationality_result.json")
    imq_score_file = os.path.join(root, "temp_image_quality_result.json")


    reasoning_datas = load_datas(reasoning_score_file)
    consis_datas = load_datas(consis_score_file)
    phy_datas = load_datas(phy_score_file)
    imq_datas = load_datas(imq_score_file)

    full_score = []
    for r in reasoning_datas:
        task_id = r["task_id"]
        if task_id[:3] == "log":
            for c in consis_datas:
                if task_id == c["task_id"]:
                    for im in imq_datas:
                        if task_id == im['task_id']:
                            single = {}
                            single["task_id"] = task_id
                            single["reasoninig"] = r["score"]
                            single["consistency"] = c["Final Score"]
                            single["physical"] = 5
                            single["image quality"] = im["score"]
                        
        else:
            for c in consis_datas:
                if task_id == c["task_id"]:
                    for p in phy_datas:
                        if task_id == p["task_id"]:
                            for im in imq_datas:
                                if task_id == im['task_id']:
                                    single = {}
                                    single["task_id"] = task_id
                                    single["reasoninig"] = r["score"]
                                    single["consistency"] = c["Final Score"]
                                    single["physical"] = p["score"]
                                    single["image quality"] = im["score"]
        full_score.append(single)
    if save_full_score:
        with open(os.path.join(save_root, f"{base}_full_score.json"), 'w', encoding='utf-8') as f:
            json.dump(full_score, f, indent=4, ensure_ascii=False)

    static_dict = {}
    sub_tasks = ["commonsense_knowledge", "domain_knowledge", "perceptual_knowledge", "societal_knowledge",
                "logical_capability", "experiential_knowledge", "spatial_knowledge", "temporal_knowledge"]
    for task in sub_tasks:
        static_dict[task] = {}
        static_dict[task]["pass_num"] = 0
        static_dict[task]["total count"] = 0
    pass_count = 0
    total_num = len(full_score)
    for sample in full_score:
        for t in sub_tasks:
            if t[:3] == sample["task_id"][:3]:
                key = t
        static_dict[key]["total count"] += 1
        if sample["reasoninig"] == 1 and sample["consistency"] == 5 and sample["physical"]==5 and sample["image quality"] == 3:
        #if sample["reasoninig"] == 1:
            pass_count += 1
            static_dict[key]["pass_num"] += 1
    for task in sub_tasks:
        static_dict[task]["strict score"] = round(static_dict[task]["pass_num"]/static_dict[task]["total count"] * 100, ndigits=2)
    print("total num:",total_num)
    print("pass num", pass_count)
    print("strict score:", pass_count / total_num)
    static_dict["full set"] = {}
    static_dict["full set"]["pass_num"] = pass_count
    static_dict["full set"]["total count"] = total_num
    static_dict["full set"]["strict score"] =round(pass_count / total_num * 100,ndigits=2)
    with open(f"{model_name}_strict.json", 'w', encoding='utf-8') as f:
        json.dump(static_dict, f, indent=4, ensure_ascii=False)

