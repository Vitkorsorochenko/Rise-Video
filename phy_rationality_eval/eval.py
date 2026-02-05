import os
import json
import logging
import time
import base64
import re
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
from openai import OpenAI

PHYSICS_EVAL_PROMPT_TEMPLATE = """
**Role:** You are a rigorous physics and visual effects analyst.

**Objective:** Evaluate the physical correctness of the provided video frames.

### Evaluation Rubric (Amplitude-Aware)

**1 (Scene Broken):** Scene jumps to unrelated content. Common-sense continuity of both the main subject and the background is lost.

**2 (Severe & Large-Amplitude Errors):** Persistent, large-amplitude physical failures in the main subject or core interaction (e.g., deep clipping, structural break, rigid bodies melting, sudden appearing/vanishing). Immediately breaks immersion.

**3 (Noticeable & Medium/Large Amplitude):** Medium to large-amplitude physical violations in the main subject **or background** (e.g., clear distortion, unnatural fluid, objects popping in/out, abrupt trajectory/velocity change). Semantics still understandable, realism reduced.

**4 (Minor & Small Amplitude, Needs Review):** Small-amplitude physical artifacts in the main subject **or background** (e.g., slight texture shimmering/flicker, minor liquid jitter). Does not block understanding, often requires replay to confirm.

**5 (Physically Seamless):** No perceivable physical errors. Motion, contact, fluidity, object permanence, and material state transitions feel naturally continuous.

### Requirements

- Respond with **one valid JSON object**:

**Example Output:**

{{
    "score": 2,
    "justification": "The object clipped deeply through the surface and cast no shadow."
}}
"""

def parse_response(response_content):
    try:
        pattern = r"```(?:json)?\s*(\{.*?\})\s*```|(\{.*\})"
        match = re.search(pattern, response_content, re.IGNORECASE | re.DOTALL)
        json_string = match.group(1) if match and match.group(1) else (match.group(2) if match else None)
        
        if not json_string:
            json_string = response_content.strip()

        data = json.loads(json_string)
        return data.get("score"), data.get("justification")

    except Exception as e:
        return None, response_content

def load_frames_from_video(video_path, num_segments=16, reverse=False):
    if not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    base64_images = []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return []

        indices = np.linspace(0, total_frames - 1, min(num_segments, total_frames), dtype=int)
        target_indices_set = set(indices)
        max_target_idx = indices[-1] 
        current_frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or current_frame_idx > max_target_idx:
                break

            if current_frame_idx in target_indices_set:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    buffered = BytesIO()
                    pil_img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    base64_images.append(f"data:image/jpeg;base64,{img_str}")
                except Exception as e:
                    print(f"Error processing frame {current_frame_idx}: {e}")

            current_frame_idx += 1

    finally:
        cap.release()
    
    if reverse:
        base64_images = base64_images[::-1]
    
    return base64_images

def evaluate_intent_physics_single(video_path, intent, reverse, client, model_name):
    frames = load_frames_from_video(video_path, num_segments=16, reverse=reverse)
    
    if not frames:
        return None, "Frame extraction failed"

    prompt = PHYSICS_EVAL_PROMPT_TEMPLATE.format(PROVIDED_INTENT=intent)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": [
            {"type": "text", "text": "Evaluate based on the provided intent."}
        ] + [{"type": "image_url", "image_url": {"url": f}} for f in frames]}
    ]


    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0
    )
    content = completion.choices[0].message.content
    return parse_response(content)


def eval_phy_rationality(data_item, result_json_path, gpt_url, gpt_key, gpt_model):

    client = OpenAI(base_url=gpt_url, api_key=gpt_key)
    
    task_id = data_item.get('task_id')
    video_path = data_item.get('video_path') or data_item.get('video')
    
    intent = data_item.get('intent_classification', 'REALISTIC')
    
    raw_reverse = data_item.get('reverse', False)
    if isinstance(raw_reverse, str):
        is_reverse = raw_reverse.upper() == 'TRUE'
    else:
        is_reverse = bool(raw_reverse)

    if not video_path or not os.path.exists(video_path):
        print(f"  [PhyIntent] Video not found: {video_path}")
        return None
    
    score, justification = evaluate_intent_physics_single(
        video_path, intent, is_reverse, client, gpt_model
    )
    
    new_record = {
        "task_id": task_id,
        "video_path": video_path,
        "intent_classification": intent,
        "reverse": is_reverse,
        "score": score,
        "justification": justification
    }
    
    try:
        with open(f"{result_json_path}/{data_item['task_id']}.json", 'w', encoding='utf-8') as f:
            json.dump(new_record, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"  [PhyIntent] Save Error: {e}")

    return score