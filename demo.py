import streamlit as st
import cv2
from ultralytics import YOLO
from config import *
import time
import tempfile
from collections import defaultdict
import pandas as pd
from datetime import datetime
import os

DEBOUNCE_SEC = 3
ESCALATE_SEC = 7


violation_state = defaultdict(lambda: {
    "start_time": None,
    "missing_ppe": [],
    "alerted": False
})

events_log = []

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

def get_box_track_id(box):
    if box.id is None:
        return None

    try:
        return int(box.id.item())
    except Exception:
        return int(box.id)


def process_detections(results, model, frame_id, timestamp, frame=None, annotated_frame=None):

    detected_violations = {}

    if results is None or len(results) == 0 or results[0].boxes is None:
        return []

    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        cls_name = model.names[cls_id]

        if cls_name in VIOLATION_CLASSES:
            track_id = get_box_track_id(box)

            if track_id is None:
                continue

            detected_violations.setdefault(track_id, []).append(cls_name)

    new_alerts = []

    for track_id, missing in detected_violations.items():
        state = violation_state[track_id]

        if state["start_time"] is None:
            state["start_time"] = timestamp
            state["missing_ppe"] = missing
            state["alerted"] = False
        else:
            state["missing_ppe"] = missing

        duration = (timestamp - state["start_time"]).total_seconds()

        if duration >= ESCALATE_SEC:
            severity = "HIGH"
        elif duration >= DEBOUNCE_SEC:
            severity = "MEDIUM"
        else:
            severity = None

        if severity and not state["alerted"]:
            state["alerted"] = True

            alert = {
                "event_id": f"EVT-{len(events_log) + len(new_alerts) + 1:03d}",
                "object_id": f"OBJ-{track_id:03d}",
                "missing_ppe": ", ".join(sorted(set(missing))),
                "started_at": state["start_time"].strftime("%H:%M:%S"),
                "duration_sec": round(duration),
                "severity": severity,
                "resolved_at": None,
                "frame_id": frame_id
            }

            new_alerts.append(alert)

    for track_id, state in list(violation_state.items()):
        if track_id not in detected_violations and state["start_time"] is not None:
            if state["alerted"]:
                resolved_event = {
                    "event_id": f"EVT-{len(events_log) + len(new_alerts) + 1:03d}",
                    "object_id": f"OBJ-{track_id:03d}",
                    "missing_ppe": ", ".join(sorted(set(state["missing_ppe"]))),
                    "started_at": state["start_time"].strftime("%H:%M:%S"),
                    "duration_sec": round((timestamp - state["start_time"]).total_seconds()),
                    "severity": "RESOLVED",
                    "resolved_at": timestamp.strftime("%H:%M:%S"),
                    "frame_id": frame_id
                }

                new_alerts.append(resolved_event)

            violation_state[track_id] = {
                "start_time": None,
                "missing_ppe": [],
                "alerted": False
            }

    events_log.extend(new_alerts)
    if len(new_alerts) > 0:
        for alert in new_alerts:
            if alert["severity"] in ["MEDIUM", "HIGH"]:
                image_name = f"{alert['event_id']}_{alert['object_id']}_frame_{frame_id}.jpg"
                image_path = os.path.join(FRAME_DIR, image_name)

                if annotated_frame is not None:
                    cv2.imwrite(image_path, annotated_frame)
                elif frame is not None:
                    cv2.imwrite(image_path, frame)
                else:
                    image_path = None

                alert["image_path"] = image_path

        df_events = pd.DataFrame(events_log)
        df_events.to_csv(
            LOG_PATH,
            index=False,
            encoding="utf-8-sig"
        )
    return new_alerts

st.set_page_config(
    page_title="PPE Detection Demo",
    layout="wide"
)

st.sidebar.header("PPE Detection")

model_selectbox = st.sidebar.selectbox(
    "Choose model",
    ("rtdetr", "yolov8", "yolov5"),
    index=1
)

mode_selectbox = st.sidebar.selectbox(
    "Choose mode",
    ("Camera", "Video"),
    index=0
)

conf_thres = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

iou_thres = st.sidebar.slider(
    "IoU threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

imgsz = st.sidebar.selectbox(
    "Image size",
    (416, 512, 640, 768),
    index=2
)

video_source = mode_dict.get(mode_selectbox)

if mode_selectbox == "Video":
    uploaded_file = st.sidebar.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_source = tfile.name
    else:
        video_source = VIDEO_PATH

run = st.sidebar.checkbox("Start")

@st.cache_resource
def load_model(model_name):
    return YOLO(model_dict.get(model_name))


model = load_model(model_selectbox)

# model = YOLO(model_dict.get(model_selectbox))
left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Original")

with right_column:
    st.subheader("Detection")

original_frame = left_column.image([])
pred_frame = right_column.image([])

metrics_placeholder = st.empty()
event_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(video_source)
    frame_id = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                st.warning("Video ended or camera frame could not be read.")
                break

            frame_id += 1
            timestamp = datetime.now()

            original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            start_time = time.time()

            results = model.track(
                frame,
                persist=True,
                conf=conf_thres,
                iou=iou_thres,
                imgsz=imgsz,
                tracker="bytetrack.yaml",
                verbose=False
            )

            total_time = time.time() - start_time
            fps = 1 / total_time if total_time > 0 else 0

            annotated = results[0].plot()
            new_alerts = process_detections(
                results=results,
                model=model,
                frame_id=frame_id,
                timestamp=timestamp,
                frame=frame,
                annotated_frame=annotated
            )

            for alert in new_alerts:
                if alert["severity"] in ["MEDIUM", "HIGH"]:
                    st.toast(
                        f"{alert['object_id']} | {alert['missing_ppe']} | "
                        f"{alert['duration_sec']}s | {alert['severity']}"
                    )

            latency_ms = results[0].speed.get("inference", 0)

            try:
                device_type = "GPU" if model.device.type == "cuda" else "CPU"
            except Exception:
                device_type = "Unknown"

            
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            metrics_placeholder.markdown(
                f"""
                **FPS:** {fps:.2f} &nbsp;|&nbsp;
                **Latency:** {latency_ms:.2f} ms &nbsp;|&nbsp;
                **Device:** {device_type} &nbsp;|&nbsp;
                **Frame:** {frame_id}
                """
            )

            original_frame.image(
                original_rgb,
                channels="RGB",
                width="stretch"
            )

            pred_frame.image(
                annotated_rgb,
                channels="RGB",
                width="stretch"
            )

            if len(events_log) > 0:
                df_events = pd.DataFrame(events_log)
                event_placeholder.dataframe(
                    df_events.tail(10),
                    width='stretch'
                )

    except Exception as e:
        st.error(f"Bị ngắt kết nối hoặc có lỗi: {e}")

    finally:
        cap.release()

else:
    st.info("Chọn model, chọn nguồn video/camera rồi bấm Start để chạy demo.")