## To Calculate MAE
Package usage:

```python
python=3.10
numpy==1.24.4
pandas==2.0.3
```

## Data Structure
```python
    json_path = "outputs/hailuo_phy/v4_phy_hailuo_url.json" # Need "task_id"
    root_dir = "outputs/hailuo_phy/" # the dir of gpt and human results
    gpt_under_root = "final_phy/shibei-easy-5-criti-ck" # You can add sub directory HERE

    TOTAL_PHY_VIS_CONS = [5,3,5] # Full score for phy_rationality, visual_quality, consistency
    sheet_num = 5 # num of experts
    #### GPT output
    ins_yesno_json = ""
    ins_json = os.path.join(root_dir, gpt_under_root, "temp_reasoning_score.json") # 
    phy_json = os.path.join(root_dir, gpt_under_root, "phy_rationality_result.json")
    vis_json = os.path.join(root_dir, gpt_under_root, "temp_image_quality_result.json") # 
    cons_json = os.path.join(root_dir, gpt_under_root, "temp_consis_result.json") # 
    # fix name for human score xlsx "human_sheet{experts_number}.xlsx"
    xlsx_path = os.path.join(root_dir, f"human_sheet{sheet_num}.xlsx")
```

run
```python
python extract_human_xlsx.py
```

The output:
```python
out_json  = os.path.join(root_dir, gpt_under_root, "MAE_human_gpt_100_maefix.json")
out_xlsx = os.path.join(root_dir, gpt_under_root, "MAE_human_gpt_100_maefix.xlsx")
```