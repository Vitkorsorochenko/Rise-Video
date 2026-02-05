from openai import OpenAI
import simplejson as json
import base64
import cv2
import os
import numpy as np

def uniform_indices(T, num=16):
    if T <= 0:
        return np.array([], dtype=int)
    num = min(num, T)               
    idx = np.linspace(0, T-1, num=num, dtype=float)
    idx = np.round(idx).astype(int) 
    idx = np.clip(idx, 0, T-1)
    uniq = np.unique(idx)
    if len(uniq) < num:
        extra = np.setdiff1d(idx, uniq, assume_unique=False)
        idx = np.concatenate([uniq, extra])[:num]
    else:
        idx = uniq[:num]
    return idx
def sample_frames(data, frame_counts = 16, img_ext=".jpg", quality=90, save_root = "consis_debug", save = False):
    video_path = data["video_path"]
    cap = cv2.VideoCapture(video_path)
    name = os.path.basename(video_path)
    name = os.path.splitext(name)[0]
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
    idxs = uniform_indices(T, frame_counts)
    
    frames_b64 = []

    frame_count = 0
    encode_params = []
    if img_ext.lower() in (".jpg", ".jpeg"):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count in idxs:
            if save:
                output_dir = os.path.join(save_root, data["task_id"])
                os.makedirs(output_dir, exist_ok = True)
                frame_filename = os.path.join(output_dir, f"consis_{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
            ok, buf = cv2.imencode(img_ext, frame, encode_params)
            if ok:
                b64 = base64.b64encode(buf).decode("utf-8")  # buf 本身就是 np.ndarray of uint8
                frames_b64.append(b64)
        frame_count += 1
    cap.release()
    return frames_b64 


def eval_consist(data, result_json, GPT_URL, GPT_KEY, GPT_MODEL):
    client = OpenAI(
                base_url=GPT_URL,
                api_key=GPT_KEY)

    if "video_path" in data:
        name = data['task_id']
        #sub_task = data['sub_task']
        instruct = data['text']
        prompt_system = f"""
            # Video Object Consistency Evaluation Instruction

            You are a highly skilled **video evaluator**. You will receive a video clip and a specific **instruction**. The video may depict an evolving scene, but your task is **ONLY** to evaluate whether the **objects remain visually and semantically consistent across frames**, **except** for changes that are explicitly required or implied by the instruction. 

            ## Task

            Evaluate **object consistency throughout the video** using the following 1–5 scale:

            - **5 (Perfect Consistency)**  
            Apart from changes required by the instruction (e.g., motion, action, time progression), all other details—object identity, personal features, colors, shapes, background, and spatial layout—remain completely stable across all frames.

            - **4 (Minor Differences)**  
            Mostly consistent with only one minor temporal discrepancy not implied by the instruction (e.g., brief lighting flicker, a momentarily missing accessory).

            - **3 (Noticeable Differences)**  
            One **noticeable inconsistency** across frames (e.g., attribute shifts briefly, background element jumps).

            - **2 (Significant Differences)**  
            **Two or more** inconsistencies (e.g., appearance changes and environment changes, an object identity briefly swaps/disappears, or appearance of unexpected new objects).

            - **1 (Severe Differences)**  
            Visual/semantic continuity repeatedly breaks. Key identities or scene attributes (e.g., major appearance features, background layout) change drastically, clearly deviating from intended continuity.

            ## Example

            **Instruction:** Two women—one in a black dress and one in a white dress—are sitting on a bench. The woman in the black dress stands up.

            - **Score 5 — Perfect Consistency**  
            Both women’s clothing, hairstyles, skin tones, and body shapes remain stable; the bench texture and background stay unchanged; only the black-dress woman smoothly transitions from sitting to standing with no flicker or jumps.

            - **Score 4 — Minor Differences**  
            Overall consistent; the black-dress woman stands normally. There is a single brief exposure flicker (or the white-dress woman’s earring is momentarily occluded for one frame) that immediately returns to normal, without affecting identity or layout stability.

            - **Score 3 — Noticeable Differences**  
            The stand-up motion is correct, but during a segment the black dress shifts slightly toward gray for a few frames and then reverts; identities and scene layout remain stable, with only this one brief, localized inconsistency.

            - **Score 2 — Significant Differences**  
            Two issues or more prolonged issue: the black-dress woman’s hair length repeatedly shortens and returns over many frames, and the bench wood grain changes at several moments; identities are still recognizable and the scene is not fundamentally reconfigured.

            - **Score 1 — Severe Differences**  
            Identity- or scene-level failures: the black-dress woman morphs into a different person or swaps dress colors with the white-dress woman, the white-dress woman disappears or teleports, and the background jumps from a park bench to an indoor hallway—continuity is clearly broken.

            ## Notes

            - **Ignore** changes explicitly stated or implied by the **instruction**.  
            - Focus on unintended issues: identity drift, texture flicker, background jump, spatial discontinuity, or attribute change (e,g, color, size, count and so on).
            - **DO NOT** judge whether the video follows the instructions. Only evaluate based on object consistency for scoring.

            ## Input

            **Instruction:** {instruct}

            ## Output Format (**strict JSON**)
            {{
                "Instruction": "Repeat the instruction you received",
                "Final Score": 1–5,
                "Reason": "A concise 1-2 sentence analysis to support your score"
            }}
            Do NOT wrap the JSON output in markdown code blocks (no ```json, no ```).
            Return only a valid **JSON dictionary**.
            """
        frames = sample_frames(data)

        messages = [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": [
                    *[{"type": "image_url",  "image_url": {"url": f"data:image/jpeg;base64,{path}"}} for path in frames]
                ]}
            ]
        flag = 1
        
        result = {}
        while(flag <= 5):
            try:
                response = client.chat.completions.create(
                        model=GPT_MODEL,
                        messages=messages,
                        temperature=0.0
                    )
                raw_response = response.choices[0].message.content
                #print(raw_response)
                output = json.loads(raw_response)
            
                result['task_id'] = name
                for key in ['Instruction', 'Final Score', 'Reason']:
                    res = output[key]
                    result[key] = res
                with open(f"{result_json}/{data['task_id']}.json", 'w',encoding='utf-8') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                
                flag = 10

                
            except Exception as e:
                flag += 1
                print(f"{data['task_id']} consist eval {flag} time fail: {e}")
            
        if flag == 6:
            print(f"{data['task_id']} fail over 5 times!")
            result['id'] = name
            result['Instruction'] = instruct
            result['Final Score'] = "bad response"
            result['Reason'] = "bad response"
        
        
        
        return result["Final Score"]
    else: 
        print(f"no video for {data['task_id']}")