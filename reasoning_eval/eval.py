import os
import json
from .vqa_merge import call_gpt
from .maze import draw_test_traj
from .symmetry import compute_colored_match, compute_grid_score_step

def eval_reasoning(data, root_dir, result_json, score_json, GPT_URL, GPT_KEY, GPT_MODEL):
   
    if data["extra_frame"] == "sym":
        gt_path = data["ref_path"]
        shrink = 0.8
        grid_size = (10, 16)
        color_name = data["color_name"]  #TODO

        # for last_frame_path in all_frames:
        last_frame_path = data["frame_path"][0]
        save_dir = os.path.join(os.path.join(root_dir, "sym_cache"), data['task_id'])
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "last_frame_marked_fps.jpg")

        num_fn, num_fp = compute_colored_match(
            gt_path,
            last_frame_path,
            color_name=color_name,
            grid_size=(10, 16),
            save_path=save_path,
            shrink=shrink)
        num_err, err_rate, acc, score = compute_grid_score_step(num_fn, num_fp, grid_size)
        score["task_id"] = data['task_id']
     
        with open(f"{score_json}/{data['task_id']}.json", 'w', encoding='utf-8') as js:
            json.dump(score, js, indent=4, ensure_ascii=False)

    elif data["extra_frame"] == "maze":
        
        
        outframe_path = os.path.join(os.path.join(root_dir, "maze_cache"), data['task_id'])
        score = draw_test_traj(data["video_path"], outframe_path, iou_thres = 0)
        score["task_id"] = data['task_id']
     
        with open(f"{score_json}/{data['task_id']}.json", 'w', encoding='utf-8') as js:
            json.dump(score, js, indent=4, ensure_ascii=False)
    else:
        result, score = call_gpt(data, GPT_URL, GPT_KEY, GPT_MODEL)
        
       
        if not score == "bad response":
            with open(f"{result_json}/{data['task_id']}.json", 'w',encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

            with open(f"{score_json}/{data['task_id']}.json",'w', encoding='utf-8') as js:
                json.dump(score, js, indent=4, ensure_ascii=False)
    
   
    
    return score['score']
