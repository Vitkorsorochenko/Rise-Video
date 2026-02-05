import cv2
import numpy as np
import os
import json
from math import sqrt

def detect_object_bbox(frame, lower_hsv, upper_hsv):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    cx, cy = x + w // 2, y + h // 2
    return (cx, cy, w, h)

def detect_green(frame):
    lower = np.array([35, 80, 80])
    upper = np.array([85, 255, 255])
    return detect_object_bbox(frame, lower, upper)

def detect_red(frame):
    lower1 = np.array([0, 120, 70])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 120, 70])
    upper2 = np.array([180, 255, 255])
    result1 = detect_object_bbox(frame, lower1, upper1)
    result2 = detect_object_bbox(frame, lower2, upper2)
    if result1 and result2:
        return result1 if result1[2] * result1[3] > result2[2] * result2[3] else result2
    return result1 or result2

def draw_yellow_endpoint(canvas, bbox, shrink_ratio=1.):
    if not bbox:
        return canvas
    cx, cy, w, h = bbox
    min_side = min(w, h)

    r = sqrt((w/2.)**2 + (h/2.)**2)
    yellow_radius = int(r * shrink_ratio)
    # cv2.circle(canvas, (cx, cy), yellow_radius + 2, (0, 0, 0), -1) 
    cv2.circle(canvas, (cx, cy), yellow_radius, (0, 255, 255), -1)
    return canvas

def mask_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0.0

def get_goal_mask(hsv_img, dst_color="green"):
    if dst_color.lower() == "green":
        # green in hsv band
        color_mask = cv2.inRange(hsv_img, (40, 50, 50), (80, 255, 255))
    else:
        # red in hsv band
        mask1 = cv2.inRange(hsv_img, (0, 100, 100), (10, 255, 255))
        mask2 = cv2.inRange(hsv_img, (170, 100, 100), (180, 255, 255))
        color_mask = cv2.bitwise_or(mask1, mask2)

    # find contours
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.zeros_like(color_mask), None

    # smaxt contour
    c = max(contours, key=cv2.contourArea)

    # minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(c)
    center = (int(x), int(y))
    radius = int(radius)

    # circle mask
    circle_mask = np.zeros_like(color_mask)
    cv2.circle(circle_mask, center, radius, 255, -1)

    return circle_mask, (center, radius)

def test_maze(frame0_path, traj_frame_path, dst_color, iou_thres):
    # 
    img0 = cv2.imread(frame0_path)
    img_traj = cv2.imread(traj_frame_path)

    if img0 is None or img_traj is None:
        raise FileNotFoundError("Cannot find image")

    # to hsv
    hsv0 = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
    hsv_traj = cv2.cvtColor(img_traj, cv2.COLOR_BGR2HSV)

    # blue and black overlap
    blue_mask = cv2.inRange(hsv_traj, (100, 100, 100), (130, 255, 255))  # blue frame_traj
    black_mask = cv2.inRange(hsv0, (0, 0, 0), (180, 255, 50))             # black wall in frame0

    # blue and black overlap
    overlap_blue_black = cv2.bitwise_and(blue_mask, black_mask)
    blue_black_intersect = np.any(overlap_blue_black > 0)

    # draw yellow trajectory mask
    yellow_mask = cv2.inRange(hsv_traj, (20, 100, 100), (40, 255, 255))

    # 
    dst_mask, circle_info = get_goal_mask(hsv0, dst_color)

    # whether yellow reaches goal
    # overlap_yellow_dst = cv2.bitwise_and(yellow_mask, dst_mask)
    # yellow_reach_goal = np.any(overlap_yellow_dst > 0)
    iou = mask_iou(yellow_mask > 0, dst_mask > 0)
    yellow_reach_goal = iou > iou_thres  # Here 0

    #print(f"yellow-goal IoU={iou:.3f}, reach={yellow_reach_goal}")

    # 
    score = 0
    if (not blue_black_intersect) and yellow_reach_goal:
        score = 1
    elif (not blue_black_intersect) or yellow_reach_goal:
        score = 0.5
    else:
        score = 0

    # 
    return score, {
        "no_blue_black_intersect": (not blue_black_intersect),
        "yellow_reach_goal_iou": f"{iou:.3f}",
        "yellow_reach_goal": yellow_reach_goal,
    }

def draw_test_traj(video_path, out_path, iou_thres):
    os.makedirs(out_path, exist_ok=True)
    output_json = []
    video_name = os.path.basename(video_path)
    # for idx, video_colors in enumerate(video_whitelist):
    #     for video_name, colors in video_colors.items():
    out_path_ = os.path.join(out_path, video_name.split(".")[0])
    os.makedirs(out_path_, exist_ok=True)

    # 
    color_detector = detect_green
    line_color = (255, 0, 0)
    src_color_name = "Green"

    video_file = video_path
    cap = cv2.VideoCapture(video_file)
    success, first_frame = cap.read()
    if not success:
        print(f"Cannot Read Video: {video_file}")
        

    # ===== Save frame0 =====
    frame0_name = video_name.replace(".mp4", "_frame0.jpg")
    frame0_path = os.path.join(out_path_, frame0_name)
    cv2.imwrite(frame0_path, first_frame)
    #print(f"Save Frame0: {frame0_name}")

    canvas = first_frame.copy()
    traj = []
    last_bbox = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bbox = color_detector(frame)
        if bbox:
            pos = bbox[:2]
            traj.append(pos)
            last_bbox = bbox

    # ===== draw traj =====
    for i in range(1, len(traj)):
        cv2.line(canvas, traj[i - 1], traj[i], line_color, 5)

    if traj:
        cv2.circle(canvas, traj[0], 5, line_color, -1)
        canvas = draw_yellow_endpoint(canvas, last_bbox, shrink_ratio=1.)

    canvas_name = f"{video_name.split('.')[0]}_traj_src_Green.jpg"
    traj_frame_path = os.path.join(out_path_, canvas_name)
    cv2.imwrite(traj_frame_path, canvas)
    cap.release()
    #print(f"{src_color_name} Trajectory Saved To: {traj_frame_path}")
    # classic
    score, detail = test_maze(frame0_path, traj_frame_path, dst_color="red", iou_thres=iou_thres) # 
    # print(f"===== video name: {video_name} =====")
    # print(f"score: {score}")
    # print(f"detail: {detail}")

    item = {
        #"idx": idx,
        "video_path": video_file,
        "frame0_path": frame0_path,
        "merged_keyframes": [frame0_path, traj_frame_path],
        "merged_keyframes_len": 2,
        "iou_thres": iou_thres,
        "score": score,
        "detail": str(detail)
    }
    output_json.append(item)

    # ===== Save JSON =====
    with open(f'{out_path_}/maze_traj_frames.json', "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
    #print(f"\nJSON Saved To: {out_path_}/maze_traj_frames.json")

    single_score = {}
    single_score["task_id"] = ""
    single_score["score"] = score
    return single_score


    