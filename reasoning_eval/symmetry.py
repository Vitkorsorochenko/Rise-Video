import cv2
import numpy as np
import json
import os

def auto_crop_black_border(img, thresh=10): # 
    """
    Automatically remove the black borders (near-black pixels) around the image.
    thresh: 0 - 255. The larger the value, the more tolerant it is (10 - 20 are commonly used).
    """
    if img is None:
        raise ValueError("img is None")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Black = Pixel value is less than thresh
    mask = gray > thresh  # True=Not black

    # Find the smallest bounding box of non-black pixels
    coords = np.argwhere(mask)

    if coords.size == 0:
        # image all black
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # +1 for slicing

    cropped = img[y0:y1, x0:x1]

    return cropped
# HSV range: In OpenCV, H ∈ [0, 179], S, V ∈ [0, 255]
COLOR_RANGES = {
    "red": [
        # There are two areas of red (extending beyond 0 degrees)
        (np.array([0,   80, 80]),  np.array([10, 255, 255])),
        (np.array([160, 80, 80]),  np.array([179, 255, 255])),
    ],
    "yellow": [
        (np.array([20,  80, 80]),  np.array([35, 255, 255])),
    ],
    "green": [
        (np.array([40,  80, 80]),  np.array([85, 255, 255])),
    ],
    "blue": [
        (np.array([90,  80, 80]),  np.array([130, 255, 255])),
    ],
    "purple": [
        (np.array([135, 80, 80]),  np.array([160, 255, 255])),
    ],
    "orange": [
        (np.array([5,  80, 80]),  np.array([25, 255, 255])),
    ],
}


def draw_error_cells_and_save(
    last_img,
    fn_mask,
    fp_mask,
    grid_size,
    save_path,
    x_color_fn=(255, 0, 0),   # Red X → FN (GT is colored but pred is not)
    x_color_fp=(0, 0, 255),   # Blue X → FP (GT colorless but pred colored)
    thickness=2
):
    """
    Draw an X on the FN/FP grid on the "last_img" and save it. 
    Args:
    last_img (RGB np.ndarray): H x W x 3
    fn_mask (bool array): [rows, cols], GT is colored but pred is not
    fp_mask (bool array): [rows, cols], GT is not colored but pred is colored grid_size (tuple): (rows, cols)
    save_path (str): Saving path
    x_color_fn (tuple): Color used by FN (R, G, B)
    x_color_fp (tuple): Color used by FP (R, G, B)
    """
    rows, cols = grid_size
    h, w, _ = last_img.shape
    cell_h = h // rows
    cell_w = w // cols

    vis_img = last_img.copy()

    for i in range(rows):
        for j in range(cols):
            if not fn_mask[i, j] and not fp_mask[i, j]:
                continue

            # The center position of this grid
            y0, y1 = i * cell_h, (i + 1) * cell_h
            x0, x1 = j * cell_w, (j + 1) * cell_w
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2

            size = min(cell_h, cell_w) // 3  # The size of the cross symbol

            if fn_mask[i, j]:
                color = x_color_fn
            else:
                color = x_color_fp

            # Draw X. Note that this is RGB, so when writing it, you need to convert it to BGR.
            cv2.line(vis_img, (cx - size, cy - size), (cx + size, cy + size), color, thickness)
            cv2.line(vis_img, (cx - size, cy + size), (cx + size, cy - size), color, thickness)

    # Save (using BGR format in OpenCV)
    cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

def draw_error_cells_and_save(
    last_img,
    fn_mask,
    fp_mask,
    grid_size,
    save_path,
    shrink=1.0,
    x_color_fn=(255, 0, 0),   # Red X → FN (GT is colored but pred is not)
    x_color_fp=(0, 0, 255),   # Blue X → FP (GT colorless but pred colored)
    grid_color=(0, 255, 0),   # Green checkered border line
    grid_thickness=2,
    thickness=2
):
    """
    On the "last_img" image, draw an "X" on the FN/FP grids and save it, while also drawing the grid frame lines.
    """

    rows, cols = grid_size
    h, w, _ = last_img.shape
    cell_h = h // rows
    cell_w = w // cols

    vis_img = last_img.copy()

    # grid line
    for i in range(rows):
        for j in range(cols):
            y0 = i * cell_h
            y1 = (i + 1) * cell_h
            x0 = j * cell_w
            x1 = (j + 1) * cell_w

            # Draw a rectangular box: BGR. Note that OpenCV uses BGR color format.
            cv2.rectangle(
                vis_img,
                (x0, y0),
                (x1, y1),
                grid_color,
                grid_thickness
            )

    # False X
    for i in range(rows):
        for j in range(cols):

            if not fn_mask[i, j] and not fp_mask[i, j]:
                continue

            # Determine the position of the grid
            y0 = i * cell_h
            y1 = (i + 1) * cell_h
            x0 = j * cell_w
            x1 = (j + 1) * cell_w

            # Center point
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2

            size = int(min(cell_h, cell_w) * 0.3)

            # Error Type Color
            if fn_mask[i, j]:
                color = x_color_fn
            else:
                color = x_color_fp

            # Draw X
            cv2.line(vis_img, (cx - size, cy - size), (cx + size, cy + size), color, thickness)
            cv2.line(vis_img, (cx - size, cy + size), (cx + size, cy - size), color, thickness)

    cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

#

def get_colored_cells_from_gt(
    gt_img,
    grid_size,
    color_name,
    hsv_ranges=COLOR_RANGES,
    save_path=None,
    grid_color=(0, 255, 0),   # Theoretical grid box line color (RGB)
    grid_thickness=2,
    check_color=(0, 255, 0),  # Checkmark color (RGB)
    check_thickness=3,
):

    assert color_name in hsv_ranges, f"Unsupported color_name: {color_name}"
    ranges = hsv_ranges[color_name]

    h, w, _ = gt_img.shape
    rows, cols = grid_size
    cell_h = h // rows
    cell_w = w // cols

    colored_mask = np.zeros((rows, cols), dtype=bool)
    colored_coords = []

    # 
    vis_img = gt_img.copy()

    for i in range(rows):
        for j in range(cols):
            y0, y1 = i * cell_h, (i + 1) * cell_h
            x0, x1 = j * cell_w, (j + 1) * cell_w

            gt_cell = gt_img[y0:y1, x0:x1, :]  # RGB

            # Calculate the average RGB (0-255) of this grid.
            gt_avg = gt_cell.mean(axis=(0, 1))  # [R, G, B]

            # Convert the average RGB values to HSV (using a 1x1 image for conversion to facilitate maintaining the OpenCV standard range)
            avg_rgb_1x1 = np.uint8([[gt_avg]])  # shape (1,1,3)
            avg_hsv_1x1 = cv2.cvtColor(avg_rgb_1x1, cv2.COLOR_RGB2HSV)
            h_val, s_val, v_val = avg_hsv_1x1[0, 0]

            # Determine whether this average HSV value falls within any of the HSV ranges of the target color.
            is_colored = False
            for lower, upper in ranges:
                if (
                    lower[0] <= h_val <= upper[0] and
                    lower[1] <= s_val <= upper[1] and
                    lower[2] <= v_val <= upper[2]
                ):
                    is_colored = True
                    break

            if is_colored:
                colored_mask[i, j] = True
                colored_coords.append((i, j))
                if save_path:
                    # Draw a more obvious check mark (√) inside this grid.
                    cx = int((x0 + x1) / 2)
                    cy = int((y0 + y1) / 2)
                    size = int(min(cell_h, cell_w) * 0.4)  # 

                    # Check mark with three dots: bottom left → slightly below center → top right
                    p1 = (int(cx - size * 0.6), int(cy + size * 0.2))
                    p2 = (int(cx - size * 0.1), int(cy + size * 0.7))
                    p3 = (int(cx + size * 0.7), int(cy - size * 0.7))

                    # Red
                    ck_color = (255, 0, 0)  # (R,G,B)
                    ck_th = max(check_thickness, 3)

                    cv2.line(vis_img, p1, p2, ck_color, ck_th)
                    cv2.line(vis_img, p2, p3, ck_color, ck_th)

    # drow gt grid
    if save_path:
        for i in range(rows):
            for j in range(cols):
                y0, y1 = i * cell_h, (i + 1) * cell_h
                x0, x1 = j * cell_w, (j + 1) * cell_w
                cv2.rectangle(
                    vis_img,
                    (x0, y0),
                    (x1, y1),
                    grid_color,
                    grid_thickness
                )

        # 
        if save_path is not None:
            cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

    return colored_mask, colored_coords

def get_nonwhite_cells_from_grid(
    img_rgb,
    grid_size,
    white_sat_thresh=30,
    white_val_thresh=200,
    shrink = 0.8
):
    """
    Traverse the image in a grid pattern, calculate the average color for each grid cell. If the average HSV value of a grid cell does not fall within the "white" range,
    then it is considered to be a "colored (not white)" grid cell. 
    Args:
    img_rgb (np.ndarray): RGB image (H, W, 3) grid_size (tuple): (rows, cols)
    white_sat_thresh (int): If S < threshold, it is classified as white
    white_val_thresh (int): If V > threshold, it is classified as white 
    Returns:
    colored_mask (np.ndarray[bool]): shape=(rows, cols), True indicates that the cell is not a white cell
    colored_coords (list[(i,j)]): all the coordinates of the non-white cells
    """

    h, w, _ = img_rgb.shape
    rows, cols = grid_size
    cell_h = h // rows
    cell_w = w // cols

    colored_mask = np.zeros((rows, cols), dtype=bool)
    colored_coords = []

    for i in range(rows):
        for j in range(cols):
            y0_raw, y1_raw = i * cell_h, (i + 1) * cell_h
            x0_raw, x1_raw = j * cell_w, (j + 1) * cell_w

            sub_h = int(cell_h * shrink)   # shrink ratio
            sub_w = int(cell_w * shrink)

            cy = (y0_raw + y1_raw) // 2
            cx = (x0_raw + x1_raw) // 2

            y0 = cy - sub_h // 2
            y1 = cy + sub_h // 2
            x0 = cx - sub_w // 2
            x1 = cx + sub_w // 2

            cell = img_rgb[y0:y1, x0:x1, :]

            # Calculate Avg RGB
            avg_rgb = cell.mean(axis=(0, 1))  # [R,G,B]
            avg_rgb_1x1 = np.uint8([[avg_rgb]])

            # To HSV（OpenCV: H[0,179], S,V[0,255]）
            avg_hsv = cv2.cvtColor(avg_rgb_1x1, cv2.COLOR_RGB2HSV)[0, 0]
            h_val, s_val, v_val = avg_hsv

            # Definition of White:
            #   S is very low (close to achromatic)
            #   and V is very bright (close to white or light gray)
            is_white = (s_val < white_sat_thresh) and (v_val > white_val_thresh)

            if not is_white:
                colored_mask[i, j] = True
                colored_coords.append((i, j))

    return colored_mask, colored_coords

# ----------------- Main -----------------
def compute_colored_match(
    gt_path,
    last_frame_path,
    color_name="red",
    grid_size=(3, 3),
    cell_color_ratio_thresh=0.3,
    save_path=None,
    shrink=0.8
):
    """
    1. In GT, use the HSV range of color_name to detect the "colored grid" (of this color).
    2. In last-frame, consider "non-white" as colored.
    3. Perform binary classification comparison over the entire grid: - GT_colored vs last_colored
    Calculate TP / FP / FN / TN and various indicators. 
    Returns:
    metrics (dict): {
    "num_cells", "tp", "fp", "fn", "tn",
    "num_gt_colored", "num_last_colored",
    "precision", "recall", "f1", "accuracy"
    }
    gt_colored_cells (bool array): [rows, cols]
    last_colored_cells (bool array): [rows, cols]
    """
    assert color_name in COLOR_RANGES, f"Unsupported color_name: {color_name}"

    # 
    gt_img_bgr = cv2.imread(gt_path)
    gt_img_bgr = cv2.resize(gt_img_bgr, (800, 500), interpolation=cv2.INTER_AREA)
    last_img_bgr = cv2.imread(last_frame_path)
    # no crop
    # cropped = last_img_bgr
    # cropped
    cropped = auto_crop_black_border(last_img_bgr, thresh=10)
    cv2.imwrite(os.path.join(os.path.dirname(last_frame_path), "last_frame_cropped.jpg"), cropped)
    last_img_bgr = cv2.resize(cropped, (800, 500), interpolation=cv2.INTER_AREA)
    # import ipdb;ipdb.set_trace()

    if gt_img_bgr is None:
        raise ValueError(f"Failed to read gt image: {gt_path}")
    if last_img_bgr is None:
        raise ValueError(f"Failed to read last-frame image: {last_frame_path}")

    gt_img = cv2.cvtColor(gt_img_bgr, cv2.COLOR_BGR2RGB)
    last_img = cv2.cvtColor(last_img_bgr, cv2.COLOR_BGR2RGB)

    h, w, _ = gt_img.shape
    rows, cols = grid_size
    # cell_h = h // rows
    # cell_w = w // cols

    gt_colored_mask, gt_colored_coords = get_colored_cells_from_gt(
        gt_img,
        grid_size=grid_size,
        color_name=color_name,   # or "red"/"yellow"...
        save_path=os.path.join(os.path.dirname(save_path), "marked_gt.jpg")
    )

    colored_mask, colored_coords = get_nonwhite_cells_from_grid(
        last_img,
        grid_size=(10, 16),
        shrink=shrink,
    )
    # -------- 2 wrong types--------
    # miss
    fn_mask = gt_colored_mask & (~colored_mask)

    # misplace
    fp_mask = (~gt_colored_mask) & colored_mask

    num_fn = int(fn_mask.sum())  # miss
    num_fp = int(fp_mask.sum())  # misplace


    draw_error_cells_and_save(
        last_img=last_img,
        fn_mask=fn_mask,
        fp_mask=fp_mask,
        grid_size=grid_size,
        save_path=save_path
    )
    return num_fn, num_fp

def compute_grid_score_step(num_fn, num_fp, grid_size):
    """
    Calculate the overall error rate using FN + FP, and then map the accuracy to the range of 1 to 5 through nonlinear segmentation. 
    Return:
    num_err: Total number of error cells
    error_rate: Error rate = num_err / Total number of cells accuracy:   = 1 - error_rate
    score:      A step-based rating from 1 to 5 
    """
    rows, cols = grid_size
    total_cells = rows * cols

    num_err = num_fn + num_fp
    error_rate = num_err / total_cells
    accuracy = 1.0 - error_rate

    # Scoring
    if accuracy < 0.85: # 32 wrong
        score = 0
    elif accuracy < 1.: # 8 wrong
        score = 0.5
    else:
        score = 1
    
    single_score = {}
    single_score["task_id"] = ""
    single_score["score"] = score
    return num_err, error_rate, accuracy, single_score

