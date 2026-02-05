from relax import cal_relax
from strict import cal_strict
import os


result_root_list = [
    "/mnt/nas-new/home/yangxue/lmx/video/video_repo/frame_result/cogx15_gpt_result_folder",
    "/mnt/nas-new/home/yangxue/lmx/video/video_repo/frame_result/hailuo2_3_gpt_result_folder",
    "/mnt/nas-new/home/yangxue/lmx/video/video_repo/frame_result/hunyuan_distill_gpt_result_folder",
    "/mnt/nas-new/home/yangxue/lmx/video/video_repo/frame_result/hunyuan_gpt_result_folder",
    "/mnt/nas-new/home/yangxue/lmx/video/video_repo/frame_result/kling2_6_gpt_result_folder",
    "/mnt/nas-new/home/yangxue/lmx/video/video_repo/frame_result/seed1_5pro_gpt_result_folder",
    "/mnt/nas-new/home/yangxue/lmx/video/video_repo/frame_result/sora_gpt_result_folder",
    "/mnt/nas-new/home/yangxue/lmx/video/video_repo/frame_result/veo3_1_gpt_result_folder",
    "/mnt/nas-new/home/yangxue/lmx/video/video_repo/frame_result/wan2_2_14B_gpt_result_folder",
    "/mnt/nas-new/home/yangxue/lmx/video/video_repo/frame_result/wan2_2_result_folder",
    "/mnt/nas-new/home/yangxue/lmx/video/video_repo/frame_result/wan2_6_gpt_result_folder"
]
relax_save_root = "/mnt/nas-new/home/yangxue/lmx/video/video_repo/ana_res/debug/relax"
strcit_save_root = "/mnt/nas-new/home/yangxue/lmx/video/video_repo/ana_res/debug/strict"
os.makedirs(relax_save_root, exist_ok=True)
os.makedirs(strcit_save_root, exist_ok=True)
for root in result_root_list:
    cal_relax(root, relax_save_root)
    cal_strict(root, strcit_save_root)