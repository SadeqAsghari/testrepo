#!/usr/bin/env python3
import os
import sys
import cv2
import time
import math
import datetime
import numpy as np
import depthai as dai
import pytesseract
import re

# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = "footages/input.mp4"  # your video file

# Blobs (update these paths!)
DET_BLOB_PATH = "models/yolov8n_vehicles_peds_640x352.blob"  # main detector
LP_BLOB_PATH  = "models/yolov8n_lp_640x640.blob"             # license plate detector

OUTPUT_ROOT = "output_logs"  # root for rilevazione_YYYY-MM-DD.tt + fg_YYYY-MM-DD/

# Detector input sizes (must match blob training/export)
DET_INPUT_W, DET_INPUT_H = 640, 352
LP_INPUT_W,  LP_INPUT_H  = 640, 640

CONF_THRESH_DET = 0.4
CONF_THRESH_LP  = 0.4

# Adapt these to your detector's class indices
CLS_PERSON  = 0
CLS_VEHICLE = [1, 2, 3, 5, 7]  # e.g. car, moto, bus, truck, etc.

PIXEL_TO_CM = 1.0   # placeholder; calibrate later
LINE_BLINK_SEC = 0.6

# Tesseract OCR config (Italian plates)
TESSERACT_CONFIG = (
    "--oem 1 --psm 6 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)
CAR_PLATE_RE  = re.compile(r"[A-Z]{2}[0-9]{3}[A-Z]{2}")  # AA000AA
MOTO_PLATE_RE = re.compile(r"[A-Z]{2}[0-9]{5}")          # AA00000

# ============================================================
# SECTION 1: DEVICE PIPELINE (RAW RGB888i)
# ============================================================

def build_device_pipeline():
    pipeline = dai.Pipeline()

    # ---- Main detection branch (host RGB -> ImageManip -> YOLO -> ObjectTracker) ----
    xin_bgr = pipeline.create(dai.node.XLinkIn)
    xin_bgr.setStreamName("in_bgr")

    det_manip = pipeline.create(dai.node.ImageManip)
    det_manip.initialConfig.setResize(DET_INPUT_W, DET_INPUT_H)
    det_manip.initialConfig.setKeepAspectRatio(False)
    det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888i)
    det_manip.setMaxOutputFrameSize(DET_INPUT_W * DET_INPUT_H * 3)
    xin_bgr.out.link(det_manip.inputImage)

    det_nn = pipeline.create(dai.node.YoloDetectionNetwork)
    det_nn.setBlobPath(DET_BLOB_PATH)
    det_nn.setConfidenceThreshold(CONF_THRESH_DET)
    det_nn.input.setBlocking(False)
    det_nn.input.setQueueSize(2)
    det_manip.out.link(det_nn.input)

    tracker = pipeline.create(dai.node.ObjectTracker)
    tracker.setTrackerType(dai.TrackerType.SHORT_TERM_KCF)
    tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)

    det_nn.passthrough.link(tracker.inputTrackerFrame)
    det_nn.out.link(tracker.inputDetections)

    xout_track = pipeline.create(dai.node.XLinkOut)
    xout_track.setStreamName("track_out")
    tracker.out.link(xout_track.input)

    xout_dbg = pipeline.create(dai.node.XLinkOut)
    xout_dbg.setStreamName("debug_rgb")
    det_nn.passthrough.link(xout_dbg.input)

    # ---- LP detection branch (host RGB crop -> ImageManip -> LP YOLO) ----
    xin_lp = pipeline.create(dai.node.XLinkIn)
    xin_lp.setStreamName("lp_in")

    lp_manip = pipeline.create(dai.node.ImageManip)
    lp_manip.initialConfig.setResize(LP_INPUT_W, LP_INPUT_H)
    lp_manip.initialConfig.setKeepAspectRatio(False)
    lp_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888i)
    lp_manip.setMaxOutputFrameSize(LP_INPUT_W * LP_INPUT_H * 3)
    xin_lp.out.link(lp_manip.inputImage)

    lp_nn = pipeline.create(dai.node.YoloDetectionNetwork)
    lp_nn.setBlobPath(LP_BLOB_PATH)
    lp_nn.setConfidenceThreshold(CONF_THRESH_LP)
    lp_nn.input.setBlocking(False)
    lp_nn.input.setQueueSize(4)
    lp_manip.out.link(lp_nn.input)

    xout_lp = pipeline.create(dai.node.XLinkOut)
    xout_lp.setStreamName("lp_out")
    lp_nn.out.link(xout_lp.input)

    return pipeline


class DetectionModule:
    def __init__(self):
        self.device = None
        self.q_in_bgr = None
        self.q_track = None
        self.q_debug = None
        self.q_lp_in = None
        self.q_lp_out = None

    def start(self):
        pipeline = build_device_pipeline()
        print("[DetectionModule] Starting device...")
        self.device = dai.Device(pipeline)

        self.q_in_bgr = self.device.getInputQueue("in_bgr", maxSize=4,  blocking=False)
        self.q_lp_in  = self.device.getInputQueue("lp_in",  maxSize=8,  blocking=False)
        self.q_track  = self.device.getOutputQueue("track_out", maxSize=16, blocking=False)
        self.q_debug  = self.device.getOutputQueue("debug_rgb", maxSize=4,  blocking=False)
        self.q_lp_out = self.device.getOutputQueue("lp_out",   maxSize=8,  blocking=False)

        print("[DetectionModule] Queues ready.")
        return {
            "in_bgr": self.q_in_bgr,
            "track_out": self.q_track,
            "debug_rgb": self.q_debug,
            "lp_in": self.q_lp_in,
            "lp_out": self.q_lp_out,
        }

# ============================================================
# SECTION 2: TRACKING MODULE
# ============================================================

class TrackState:
    def __init__(self, is_vehicle, det_label):
        self.is_vehicle = is_vehicle
        self.det_label = det_label
        self.progressive = None     # "V0000001" or "P0000001"
        self.positions = []         # list of (t, x, y)
        self.total_distance_cm = 0.0
        self.first_time = None
        self.last_time = None
        self.last_pos = None
        self.position_index = 0
        self.lines_crossed = set()
        self.lp_text = None         # recognized plate
        self.nationality = ""       # X(3)
        self.vehicle_type = "A"     # default type; adapt if needed

class TrackManager:
    def __init__(self):
        self.tracks = {}
        self.next_vehicle_id = 1
        self.next_ped_id = 1

    def _format_progressive(self, prefix, num):
        return f"{prefix}{num:07d}"

    def ensure_track(self, trk):
        tid = trk.id
        if tid not in self.tracks:
            label = trk.label
            is_vehicle = label in CLS_VEHICLE
            ts = TrackState(is_vehicle=is_vehicle, det_label=label)

            if is_vehicle:
                prog = self._format_progressive("V", self.next_vehicle_id)
                self.next_vehicle_id += 1
            else:
                prog = self._format_progressive("P", self.next_ped_id)
                self.next_ped_id += 1

            ts.progressive = prog
            self.tracks[tid] = ts
        return self.tracks[tid]

    def update_position(self, tid, x, y, t_sec):
        ts = self.tracks.get(tid)
        if ts is None:
            return None, None, None

        if ts.first_time is None:
            ts.first_time = t_sec
            ts.last_time = t_sec
            ts.last_pos = (x, y)
            ts.position_index = 1
            ts.positions.append((t_sec, x, y))
            return ts, 0.0, 0.0

        prev_t = ts.last_time
        prev_x, prev_y = ts.last_pos
        dt = max(t_sec - prev_t, 1e-3)
        dx = x - prev_x
        dy = y - prev_y
        dist_px = math.sqrt(dx*dx + dy*dy)
        dist_cm = dist_px * PIXEL_TO_CM
        ts.total_distance_cm += dist_cm

        speed_m_s = (dist_cm / 100.0) / dt
        speed_kmh = speed_m_s * 3.6
        speed_hundredths = speed_kmh * 100.0

        ts.last_pos = (x, y)
        ts.last_time = t_sec
        ts.position_index += 1
        ts.positions.append((t_sec, x, y))

        total_time = t_sec - ts.first_time
        return ts, speed_hundredths, total_time

# ============================================================
# SECTION 3: LOGGING MODULE (per spec)
# ============================================================

class LoggingManager:
    def __init__(self, output_root=OUTPUT_ROOT):
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)
        self.current_date = None
        self.log_file = None
        self.fg_dir = None

    def _ensure_today(self):
        today = datetime.date.today().isoformat()
        if today == self.current_date and self.log_file is not None:
            return
        self.current_date = today
        log_name = os.path.join(self.output_root, f"rilevazione_{today}.tt")
        self.fg_dir = os.path.join(self.output_root, f"fg_{today}")
        os.makedirs(self.fg_dir, exist_ok=True)
        if self.log_file:
            self.log_file.close()
        self.log_file = open(log_name, "a", encoding="utf-8")

    def _pad_num(self, value, length):
        return f"{int(value):0{length}d}"

    def _pad_text(self, value, length):
        s = (value or "")
        if len(s) > length:
            s = s[:length]
        return s.ljust(length, " ")

    def write_line(self, s: str):
        self._ensure_today()
        self.log_file.write(s + "\n")
        self.log_file.flush()

    # 2.5.1 New object identification
    def log_identification(self, ts: TrackState):
        pv = "V" if ts.is_vehicle else "P"
        prog_num = ts.progressive[1:]
        tipoVeicolo = ts.vehicle_type if ts.is_vehicle else " "
        naz = self._pad_text(ts.nationality, 3) if ts.is_vehicle else "   "
        targa = self._pad_text(ts.lp_text or "", 9) if ts.is_vehicle else " " * 9
        line = f"{pv}{prog_num}{tipoVeicolo}{naz}{targa}"
        self.write_line(line)

    # 2.5.2 New location detected
    def log_position(self, ts: TrackState, x, y, speed_hundredths, total_time_sec):
        pv = "V" if ts.is_vehicle else "P"
        prog_num = ts.progressive[1:]
        num_pos = self._pad_num(ts.position_index, 3)
        coord_x = self._pad_num(int(x), 5)
        coord_y = self._pad_num(int(y), 5)
        vel = self._pad_num(int(round(speed_hundredths)), 5)
        dist_cm = self._pad_num(int(round(ts.total_distance_cm)), 6)
        tot_time = self._pad_num(int(round(total_time_sec)), 5)
        line = f"{pv}{prog_num}{num_pos}{coord_x}{coord_y}{vel}{dist_cm}{tot_time}"
        self.write_line(line)

    # 2.5.3 Crossing line
    def log_crossing(self, ts: TrackState, line_code: str, crossing_time_unix: int, frame_bgr):
        self._ensure_today()
        tipoRiga = "A"
        prog_num = ts.progressive[1:]
        instant = self._pad_num(crossing_time_unix, 10)
        cod_linea = self._pad_text(line_code, 5)
        line = f"{tipoRiga}{prog_num}{instant}{cod_linea}"
        self.write_line(line)
        filename = f"{prog_num}_{crossing_time_unix}.jpeg"
        cv2.imwrite(os.path.join(self.fg_dir, filename), frame_bgr)

# ============================================================
# SECTION 4: GEOMETRY / UI / OCR / MAIN
# ============================================================

def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)):
            t = (y - y1) / (y2 - y1 + 1e-9)
            x_intersect = x1 + t * (x2 - x1)
            if x_intersect > x:
                inside = not inside
    return inside

def segments_intersect(p, p2, q, q2):
    def ccw(a, b, c):
        return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
    return (ccw(p, q, q2) != ccw(p2, q, q2)) and (ccw(p, p2, q) != ccw(p, p2, q2))

class DrawState:
    def __init__(self, frame):
        self.base = frame.copy()
        self.zone_pts = []
        self.lines = []  # (p1, p2, code)
        self.tmp_line_pts = []
        self.mode = "zone"

def zone_and_lines_ui(first_frame):
    st = DrawState(first_frame)
    win = "Draw zone & lines"
    cv2.namedWindow(win)

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if st.mode == "zone":
                st.zone_pts.append((x, y))
                print("[UI] Zone point:", x, y)
            else:
                st.tmp_line_pts.append((x, y))
                if len(st.tmp_line_pts) == 2:
                    p1, p2 = st.tmp_line_pts
                    code = f"L{len(st.lines)+1:03d}"
                    st.lines.append((p1, p2, code))
                    st.tmp_line_pts = []
                    print("[UI] Line added:", p1, p2, "code:", code)

    cv2.setMouseCallback(win, mouse_cb)

    print("Zone mode: left-click to add vertices. Press 's' to save zone & switch to line mode.")
    print("Line mode: left-click 2 points per line. Press 'p' to finish. 'q' to abort.")

    while True:
        vis = st.base.copy()
        if len(st.zone_pts) > 1:
            cv2.polylines(vis, [np.array(st.zone_pts, np.int32)], True, (0,255,255), 2)
        for p in st.zone_pts:
            cv2.circle(vis, p, 4, (0,255,255), -1)
        for (p1, p2, code) in st.lines:
            cv2.line(vis, p1, p2, (255,0,255), 2)
            cv2.putText(vis, code, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(vis, now_str, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow(win, vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord('s') and st.mode == "zone":
            if len(st.zone_pts) >= 3:
                print("Zone saved, switching to line mode.")
                st.mode = "lines"
            else:
                print("Need at least 3 points for zone.")
        elif k == ord('p') and st.mode == "lines":
            print("Line mode finished.")
            break
        elif k == ord('q'):
            print("Aborted drawing.")
            break

    cv2.destroyWindow(win)
    return st.zone_pts, st.lines

def preprocess_plate_for_ocr(bgr):
    h, w = bgr.shape[:2]
    scale = max(1.0, 150 / min(h, w))
    new_w = int(w * scale)
    new_h = int(h * scale)
    bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    if np.mean(bw) > 127:
        bw = 255 - bw

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    return bw

def extract_plate_text_tesseract(plate_bgr):
    bw = preprocess_plate_for_ocr(plate_bgr)
    raw = pytesseract.image_to_string(bw, config=TESSERACT_CONFIG)
    if not raw:
        return None

    raw = raw.upper().strip().replace("\n", " ")
    cleaned = "".join(ch if (ch.isalnum() or ch == " ") else " " for ch in raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    nospace = re.sub(r"\s+", "", cleaned)

    m = CAR_PLATE_RE.search(nospace)
    if m:
        return m.group(0)

    m = MOTO_PLATE_RE.search(nospace)
    if m:
        return m.group(0)

    seqs = re.findall(r"[A-Z0-9]{5,8}", nospace)
    if seqs:
        return max(seqs, key=len)
    return None

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video:", VIDEO_PATH)
        sys.exit(1)

    ret, frame0 = cap.read()
    if not ret:
        print("Cannot read first frame.")
        sys.exit(1)

    first_small = cv2.resize(frame0, (DET_INPUT_W, DET_INPUT_H))
    zone_poly, lines = zone_and_lines_ui(first_small)
    if not zone_poly:
        print("No zone defined. Exiting.")
        return
    print("[Main] Zone:", zone_poly)
    print("[Main] Lines:", lines)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    det_mod = DetectionModule()
    q = det_mod.start()
    q_in_bgr = q["in_bgr"]
    q_track  = q["track_out"]
    q_debug  = q["debug_rgb"]
    q_lp_in  = q["lp_in"]
    q_lp_out = q["lp_out"]

    track_mgr = TrackManager()
    log_mgr   = LoggingManager()

    lp_req_id = 1
    pending_lp = {}
    line_blink_until = {code: 0.0 for (_,_,code) in lines}

    start_time = time.time()
    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)

    print("[Main] Running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        t_now = time.time()
        t_el  = t_now - start_time

        frame_small_bgr = cv2.resize(frame, (DET_INPUT_W, DET_INPUT_H))
        frame_small_rgb = cv2.cvtColor(frame_small_bgr, cv2.COLOR_BGR2RGB)

        msg = dai.ImgFrame()
        msg.setType(dai.RawImgFrame.Type.RGB888i)
        msg.setWidth(frame_small_rgb.shape[1])
        msg.setHeight(frame_small_rgb.shape[0])
        msg.setData(frame_small_rgb.flatten())
        try:
            q_in_bgr.send(msg)
        except RuntimeError as e:
            print("Error sending frame:", e)
            continue

        if q_debug.has():
            dbg_msg = q_debug.get()
            dbg_frame = dbg_msg.getCvFrame()  # converted to BGR for display
        else:
            dbg_frame = frame_small_bgr.copy()

        # ---- Tracker / detection outputs ----
        while q_track.has():
            trk_msg = q_track.get()
            for trk in trk_msg.tracklets:
                tid = trk.id
                label = trk.label
                roi = trk.roi.denormalize(DET_INPUT_W, DET_INPUT_H)
                x1 = int(roi.topLeft().x)
                y1 = int(roi.topLeft().y)
                x2 = int(roi.bottomRight().x)
                y2 = int(roi.bottomRight().y)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if not point_in_polygon((cx, cy), zone_poly):
                    continue

                ts = track_mgr.ensure_track(trk)
                ts, speed_hundredths, total_time = track_mgr.update_position(
                    tid, cx, cy, t_el
                )

                if ts.first_time == t_el and ts.position_index == 1:
                    if not ts.is_vehicle:
                        log_mgr.log_identification(ts)
                else:
                    log_mgr.log_position(ts, cx, cy, speed_hundredths, total_time)

                # schedule LP detection
                if ts.is_vehicle and ts.lp_text is None:
                    pad = 5
                    xx1 = max(x1 - pad, 0)
                    yy1 = max(y1 - pad, 0)
                    xx2 = min(x2 + pad, DET_INPUT_W-1)
                    yy2 = min(y2 + pad, DET_INPUT_H-1)
                    crop_bgr = frame_small_bgr[yy1:yy2, xx1:xx2]
                    if crop_bgr.size > 0:
                        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                        msg_lp = dai.ImgFrame()
                        msg_lp.setType(dai.RawImgFrame.Type.RGB888i)
                        msg_lp.setWidth(crop_rgb.shape[1])
                        msg_lp.setHeight(crop_rgb.shape[0])
                        msg_lp.setData(crop_rgb.flatten())
                        req_id = lp_req_id
                        lp_req_id += 1
                        msg_lp.setSequenceNum(req_id)
                        pending_lp[req_id] = (tid, t_el, crop_bgr.copy())
                        try:
                            q_lp_in.send(msg_lp)
                        except RuntimeError as e:
                            print("Error sending LP crop:", e)

                # line crossing
                if len(ts.positions) >= 2:
                    p_prev = ts.positions[-2][1:]
                    p_curr = ts.positions[-1][1:]
                    for (p1, p2, code) in lines:
                        if code in ts.lines_crossed:
                            continue
                        if segments_intersect(p_prev, p_curr, p1, p2):
                            ts.lines_crossed.add(code)
                            unix_ts = int(time.time())
                            log_mgr.log_crossing(ts, code, unix_ts, dbg_frame)
                            line_blink_until[code] = time.time() + LINE_BLINK_SEC

                # draw bounding box + ID
                color = (0,255,0) if ts.is_vehicle else (255,255,0)
                cv2.rectangle(dbg_frame, (x1, y1), (x2, y2), color, 2)
                label_txt = ts.progressive
                if ts.lp_text:
                    label_txt += " " + ts.lp_text
                cv2.putText(dbg_frame, label_txt, (x1, max(0,y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # ---- LP detector outputs -> OCR ----
        while q_lp_out.has():
            lp_msg = q_lp_out.get()
            dets = lp_msg.detections
            try:
                seq = lp_msg.getSequenceNum()
            except Exception:
                seq = None

            if seq is None or seq not in pending_lp:
                continue

            tid, t_el_req, crop_bgr = pending_lp.pop(seq)
            if not dets or tid not in track_mgr.tracks:
                continue

            ts = track_mgr.tracks[tid]
            if not ts.is_vehicle or ts.lp_text is not None:
                continue

            best = max(dets, key=lambda d: d.confidence)
            h, w = crop_bgr.shape[:2]
            px1 = int(best.xmin * w)
            py1 = int(best.ymin * h)
            px2 = int(best.xmax * w)
            py2 = int(best.ymax * h)
            px1 = max(px1, 0); py1 = max(py1, 0)
            px2 = min(px2, w-1); py2 = min(py2, h-1)
            plate_crop_bgr = crop_bgr[py1:py2, px1:px2]
            if plate_crop_bgr.size == 0:
                continue

            text = extract_plate_text_tesseract(plate_crop_bgr)
            if text:
                ts.lp_text = text
                ts.nationality = ""  # or "I  " if you want to encode Italy
                log_mgr.log_identification(ts)
                print(f"[OCR] Track {ts.progressive} -> {text}")

        # draw zone & lines & clock
        cv2.polylines(
            dbg_frame,
            [np.array(zone_poly, np.int32)],
            True, (0,255,255), 2
        )

        now_t = time.time()
        for (p1, p2, code) in lines:
            col = (255,0,255)
            thick = 2
            if now_t < line_blink_until.get(code, 0):
                col = (0,255,255)
                thick = 3
            cv2.line(dbg_frame, p1, p2, col, thick)
            cv2.putText(dbg_frame, code, p1,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        clock_txt = f"{int(t_el)}s"
        cv2.putText(dbg_frame, clock_txt, (DET_INPUT_W - 80, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("debug", dbg_frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Main] Done.")


if __name__ == "__main__":
    main()
