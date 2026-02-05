import pandas as pd
import json
import itertools
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os

def cal_avg_instruction(s):
    cnt = len(s["question_scores"])
    correct = sum(s["question_scores"])
    score_1 = correct / cnt
    score_100 = score_1 * 100
    return score_100
def cal_phy_vis_cons(s, TOTAL_PHY_VIS_CONS):
    phy, vis, cons = s
    total_phy, total_vis, total_cons = TOTAL_PHY_VIS_CONS

    phy_100 = 100*((phy-1)/(total_phy-1))
    vis_100 = 100*((vis-1)/(total_vis-1))
    cons_100 = 100*((cons-1)/(total_cons-1))

    # phy_100 = 100*((phy)/(total_phy))
    # vis_100 = 100*((vis)/(total_vis))
    # cons_100 = 100*((cons)/(total_cons))
    return [phy_100, vis_100, cons_100]

def load_scores_from_xlsx(xlsx_path, sheet_num):
    """
    sheet_num: 读取前 N 个 sheet（N=1 只读第0个）
    return:
    {
      task_id: {
        sheet_name: {
          "question_scores": [...],      # col 5
          "task_scores": [phy, visual, con]  # col 6,7,8
        }
      }
    }
    """
    xls = pd.ExcelFile(xlsx_path)
    all_scores = {}

    for sheet_idx, sheet_name in enumerate(xls.sheet_names):
        if sheet_idx >= sheet_num:
            break

        df = pd.read_excel(xls, sheet_name=sheet_name)

        # 解开合并的单元格
        df = df.ffill() # 没有需要merge的就保持原样，也不会错乱。

        # 以 task_id 分组（第1列）
        task_id_col = df.columns[1]
        for task_id, g in df.groupby(task_id_col, dropna=False):
            if pd.isna(task_id):
                continue
            task_id = str(task_id)

            question_scores = g.iloc[:, 5].tolist()     # 第5列：question yes/no,取所有的
            task_scores = g.iloc[0, 6:9].tolist()       # 第6~8列：phy/visual/con，只取第一个

            all_scores.setdefault(task_id, {})[sheet_name] = {
                "question_scores": question_scores,
                "task_scores": task_scores
            }

    return all_scores

def convert_int64(obj):
    if isinstance(obj, dict):
        return {k: convert_int64(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_int64(v) for v in obj]
    elif isinstance(obj, (int, float)):  # Python 原生 int/float
        return obj
    try:
        if isinstance(obj, (np.integer,)):
            return int(obj)
    except:
        pass
    return obj

def write_scores_to_json(json_path, score_dict, out_path, field_name, TOTAL_PHY_VIS_CONS):
    data = json.load(open(json_path, "r", encoding="utf-8"))

    for item in data:
        tid = item.get("task_id")
        if tid not in score_dict:
            continue

        score_list = []
        for sheet in sorted(score_dict[tid].keys()): # 每一个人类专家
            s = score_dict[tid][sheet]
            # instruction 总分: 
            try:
                ins_100 = cal_avg_instruction(s) # 根据yes no算reasoning百分制得分
            except:
                import ipdb;ipdb.set_trace()
            phy_100, vis_100, cons_100 = cal_phy_vis_cons(s["task_scores"], TOTAL_PHY_VIS_CONS)
            score_list.append(
                {"instruction": s["question_scores"],  
                "phy_vis_con": s["task_scores"],       
                "ins_100": ins_100,  
                "phy_100": phy_100,
                "vis_100": vis_100,
                "cons_100": cons_100}
            )
        item[field_name] = score_list
            
    data = convert_int64(data)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def human_human_mae(human_scores):
    """
    human_scores:
        dict[task_id][metric] -> List[float]  # multi experts
    return:
        dict[metric] -> mean MAE over all tasks
    """

    metric_mae_sum = defaultdict(float)
    metric_task_cnt = defaultdict(int)

    for task_id, task_scores in human_scores.items():
        for metric, scores in task_scores.items():
            scores = [s for s in scores if s is not None]
            if len(scores) < 2:
                continue  # at least 2

            # all pairs
            pairwise = [
                abs(a - b)
                for a, b in itertools.combinations(scores, 2)
            ]

            task_mae = np.mean(pairwise)

            metric_mae_sum[metric] += task_mae
            metric_task_cnt[metric] += 1

    # mean among tasks
    metric_mae = {
        metric: metric_mae_sum[metric] / metric_task_cnt[metric]
        for metric in metric_mae_sum
    }

    return metric_mae
def get_human_scores_map(out_json):
    '''
    human_scores:
        dict[task_id][metric] -> List[float]  # 多个专家
    '''
    all_data = json.load(open(out_json, "r"))
    new_entry = {}

    for data in tqdm(all_data):
        task_id = data["task_id"]
        new_entry[task_id] = {'ins_yesno_100': [], "ins_100":[], "phy_100":[], "vis_100":[], "cons_100":[]} # init
        if "score_human" not in data:
            import ipdb;ipdb.set_trace()
        for scores_exp in data["score_human"]: # each expert
            new_entry[task_id]['ins_yesno_100'].append([x*100 for x in scores_exp["instruction"]]) # reasoning to 100
            # import ipdb;ipdb.set_trace()
            new_entry[task_id]["ins_100"].append(scores_exp["ins_100"])
            if not data["task_id"].startswith("log_"): # 逻辑题不参加评分
                new_entry[task_id]["phy_100"].append(scores_exp["phy_100"])
            new_entry[task_id]["vis_100"].append(scores_exp["vis_100"])
            new_entry[task_id]["cons_100"].append(scores_exp["cons_100"])

    return new_entry

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def to_100(score, max_score):
    if score is None:
        return None
    return 100.0 * (score-1) / (max_score-1)

def load_gpt_scores(
    ins_yesno_json,
    reasoning_json,
    phy_json,
    image_json,
    consis_json,
    tasks,
):
    gpt = defaultdict(dict)
    if ins_yesno_json:
        for d in load_json(ins_yesno_json):
            for dq in d:
                if dq['task_id'] not in tasks:
                    continue
                if not gpt.get(dq['task_id'], None):
                    gpt[dq["task_id"]]["ins_yesno_100"] = []
                if dq['answer'].lower().strip() == "yes":
                    gpt[dq["task_id"]]["ins_yesno_100"].append(100)
                else:
                    gpt[dq["task_id"]]["ins_yesno_100"].append(0)
    if reasoning_json:
        for d in load_json(reasoning_json):
            if isinstance(d["score"], (int, float)):
                # reasoning dont need to -1
                gpt[d["task_id"]]["ins_100"] = d["score"]*100
                gpt[d["task_id"]]["ins_100_raw"] = d["score"]


    if phy_json:
        for d in load_json(phy_json):
            if d["task_id"].startswith("log_"):
                continue
            gpt[d["task_id"]]["phy_100"] = to_100(d["score"], TOTAL_PHY_VIS_CONS[0])
            gpt[d["task_id"]]["phy_100_raw"] = d["score"]

    if image_json:
        for d in load_json(image_json):
            if isinstance(d["score"], (int, float)):
                gpt[d["task_id"]]["vis_100"] = to_100(d["score"], TOTAL_PHY_VIS_CONS[1])
                gpt[d["task_id"]]["vis_100_raw"] = d["score"]

    if consis_json:
        for d in load_json(consis_json):
            if isinstance(d["Final Score"], (int, float)):
                gpt[d["task_id"]]["cons_100"] = to_100(d["Final Score"], TOTAL_PHY_VIS_CONS[-1])
                gpt[d["task_id"]]["cons_100_raw"] = d["Final Score"]

    return gpt
def compute_segmented_mae_simple(
        human_scores, gpt_scores,
        metric="ins_yesno_100",
        max_score=1
    ):
    all_mae = []
    for task_name, score_dict in gpt_scores.items():
        ins_yesno_gpt = score_dict[metric]
        col_sums = [sum(col) for col in zip(*human_scores[task_name][metric])]
        col_avg = [s / len(human_scores[task_name]["ins_yesno_100"]) for s in col_sums]
        mae_this_task = [abs(h-g) for h,g in zip(col_avg, ins_yesno_gpt)]
        all_mae.append(mae_this_task)
    flat_mae = [v for row in all_mae for v in row]

    return sum(flat_mae) / len(flat_mae)

def compute_segmented_mae(
    human_scores,
    gpt_scores,
    metric,
    max_score
):
    """
    human_scores[task_id][metric] = [h1_100, h2_100, ...]
    gpt_scores[task_id][metric]   = gpt_100

    metric: 'ins_100' | 'phy_100' | 'vis_100' | 'cons_100'
    max_score: GPT full score 1 / 3 / 5
    """

    buckets = defaultdict(list)

    for task_id, h in human_scores.items():
        if task_id not in gpt_scores:
            continue
        if metric not in gpt_scores[task_id]:
            continue
        if metric not in h:
            continue

        gpt_100 = gpt_scores[task_id][metric]
        if not isinstance(gpt_100, (int, float)):
            continue

        human_list = h[metric]
        if not isinstance(human_list, list) or len(human_list) == 0:
            continue

        # ---- task-level MAE（对专家取平均）----
        # import ipdb;ipdb.set_trace()
        task_mae = abs(gpt_100 - np.mean(human_list))

        # ---- GPT 原始分 → 离散桶 ----
        gpt_raw = gpt_scores[task_id][f"{metric}_raw"]
        buckets[gpt_raw].append(task_mae)

    # -------- 汇总统计 --------
    report = {}
    total = sum(len(v) for v in buckets.values())
    print(f"[INFO] Actual Items: {total}")

    for score in sorted(buckets.keys()):
        maes = buckets[score] # gpt原始分数对应所有题的mae
        report[score] = {
            "count": len(maes),
            "ratio": len(maes) / total if total > 0 else 0.0,
            "mae": float(np.mean(maes)), # 桶平均
            "std": float(np.std(maes)),
        }

    all_maes = [m for v in buckets.values() for m in v]
    report["overall"] = {
        "count": len(all_maes),
        "mae": float(np.mean(all_maes)) if all_maes else 0.0,
        "std": float(np.std(all_maes)) if all_maes else 0.0,
    }

    return report

def print_report(title, report):
    print(f"\n==== {title} ====")
    for k, v in report.items():
        if k == "overall":
            continue
        print(
            f"[GPT={k}] "
            f"count={v['count']:3d} | "
            f"ratio={v['ratio']*100:5.1f}% | "
            f"MAE={v['mae']:6.2f} | "
            f"STD={v['std']:6.2f}"
        )
    o = report["overall"]
    print(
        f"[OVERALL] "
        f"count={o['count']:3d} | "
        f"MAE={o['mae']:6.2f} | "
        f"STD={o['std']:6.2f}"
    )
def report_to_df(report):
    rows = []
    for k, v in report.items():
        if k == "overall":
            rows.append({
                "GPT_score": "overall",
                "count": v["count"],
                "ratio(%)": 100.0,
                "MAE": v["mae"],
                "STD": v["std"],
            })
        else:
            rows.append({
                "GPT_score": k,
                "count": v["count"],
                "ratio(%)": v["ratio"] * 100,
                "MAE": v["mae"],
                "STD": v["std"],
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    #### 输出结构，一个videoModel--1个rootDir, 下面是分mode的ablation
    # 用到的数据 v4.json或者子集（不用url）
    json_path = "outputs/hailuo_phy/v4_phy_hailuo_url.json" # "outputs/hailuo/v4_sample_100_sorted_hailuo_url.json" # input video json
    # gpt 中间结果,不用换算百分
    root_dir = "outputs/hailuo_phy/"
    gpt_under_root = "final_phy/shibei-easy-5-criti-ck"
    # 分制/有几个专家可以用
    TOTAL_PHY_VIS_CONS = [5,3,5] # 物理合理性/视觉质量/一致性的满分
    sheet_num = 5 # 选前几个sheet
    ####
    ins_yesno_json = ""
    ins_json = "" # "./outputs/hailuo/temp_reasoning_score.json"
    phy_json = os.path.join(root_dir, gpt_under_root, "phy_rationality_result.json")
    vis_json = "" # "./outputs/hailuo/temp_image_quality_result.json"
    cons_json = "" # "./outputs/hailuo/temp_consis_result.json"
    # fix name for human xlsx
    xlsx_path = os.path.join(root_dir, f"human_{TOTAL_PHY_VIS_CONS[0]}score_sheet{sheet_num}.xlsx")

    # output 不用动
    out_json  = os.path.join(root_dir, gpt_under_root, "MAE_human_gpt_100_maefix.json") # "outputs/hailuo/v4_sample_100_sorted_hailuo_url_human.json" # output json
    out_xlsx = os.path.join(root_dir, gpt_under_root, "MAE_human_gpt_100_maefix.xlsx")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    if not os.path.isfile(xlsx_path):
        raise FileNotFoundError(f"Required xlsx file not found: {xlsx_path}")
    print(f"\n*** [Human Scores] phy score {TOTAL_PHY_VIS_CONS[0]}, sheets to use {sheet_num} ***\n")

    score_dict = load_scores_from_xlsx(xlsx_path, sheet_num)

    write_scores_to_json(
        json_path,
        score_dict,
        out_json,
        field_name="score_human",
        TOTAL_PHY_VIS_CONS=TOTAL_PHY_VIS_CONS,
    )
    print(f"[Saved] {out_json}")
    
    # # human-wise MAE
    human_scores = get_human_scores_map(out_json)
    # mae_result = human_human_mae(human_scores)

    # for k, v in mae_result.items():
    #     print(f"[Human-Human MAE] {k}: {v:.3f}")

    # human gpt MAE
    gpt_scores = load_gpt_scores(
        ins_yesno_json,
        ins_json,
        phy_json,
        vis_json,
        cons_json,
        tasks=list(score_dict.keys())
    )
    # import ipdb;ipdb.set_trace()
    # reasoning_yesno
    if ins_yesno_json:
        reasoning_yesno_report = compute_segmented_mae_simple(
            human_scores, gpt_scores,
            metric="ins_yesno_100",
            max_score=1
        )
        print(f"==== yesno mae ====\n {reasoning_yesno_report}")
    # reasoning (0–1)
    reasoning_report = compute_segmented_mae(
        human_scores, gpt_scores,
        metric="ins_100",
        max_score=1
    )
    # phy (0–3)
    phy_report = compute_segmented_mae(
        human_scores, gpt_scores,
        metric="phy_100",
        max_score=TOTAL_PHY_VIS_CONS[0]
    )
    # image quality (0–3)
    image_report = compute_segmented_mae(
        human_scores, gpt_scores,
        metric="vis_100",
        max_score=TOTAL_PHY_VIS_CONS[1]
    )
    # consistency (0–5)
    consis_report = compute_segmented_mae(
        human_scores, gpt_scores,
        metric="cons_100",
        max_score=TOTAL_PHY_VIS_CONS[-1]
    )
    # ===== 打印所有指标 =====
    print_report("Instruction Reasoning", reasoning_report)
    print_report("Physical Rationality", phy_report)
    print_report("Image Quality", image_report)
    print_report("Consistency", consis_report)

    # human_gpt 写入xlsx
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        report_to_df(reasoning_report).to_excel(
            writer, sheet_name="Instruction_Reasoning", index=False
        )
        report_to_df(phy_report).to_excel(
            writer, sheet_name="Physical_Rationality", index=False
        )
        report_to_df(image_report).to_excel(
            writer, sheet_name="Image_Quality", index=False
        )
        report_to_df(consis_report).to_excel(
            writer, sheet_name="Consistency", index=False
        )

    print(f"[SAVE] MAE results saved to {out_xlsx}")