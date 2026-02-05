import json
import os

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def filter_by_common_task_ids(json_paths, datas):
    """
    json_paths: List[str]
    """
    #all_data = {}
    task_id_sets = []

    for p in json_paths:
        # data = load_json(p)
        # all_data[p] = data
        
        for _, _, files in os.walk(p):
            
            task_ids = set() # init container
            for d in files:
                # if isinstance(d, dict):
                #     if "task_id" in d:
                if os.path.splitext(d)[1] == ".json":
                    task_ids.add(os.path.splitext(d)[0])

            if p.split('/')[-1].split('_')[1] == "phy":
                print("phy auto add logical")
                for i in range(1,43):
                    task_ids.add(f"log_{i}")
                
               
        task_id_sets.append(task_ids)


    # 2) intersection
    common_task_ids = set.intersection(*task_id_sets)

    print(f"[INFO] Common task_ids: {len(common_task_ids)}")
    return common_task_ids