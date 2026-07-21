import base64
import hashlib
import io
import os
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

from src.detector import PCBDetector

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image as PDFImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


APP_NAME = "PCBInspect AI"
MODEL_NAME = "YOLOv8"
DEVELOPER_NAME = "Keerthy GS"
GITHUB_URL = "https://github.com/keerthy-gs"

st.set_page_config(
    page_title=f"{APP_NAME} | Industrial Inspection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def safe_html(html: str, sidebar: bool = False) -> None:
    """Render HTML without Markdown showing the tags as code."""
    cleaned = dedent(html).strip()
    target = st.sidebar if sidebar else st

    if hasattr(target, "html"):
        target.html(cleaned)
    else:
        compact = " ".join(line.strip() for line in cleaned.splitlines())
        target.markdown(compact, unsafe_allow_html=True)


safe_html(
    """
    <style>
        :root {
            --bg: #edf3fa;
            --card: rgba(255, 255, 255, 0.97);
            --soft: #f7faff;
            --text: #0f172a;
            --muted: #64748b;
            --border: #dbe5f1;
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --shadow: 0 10px 28px rgba(15, 23, 42, 0.09);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --bg: #07111f;
                --card: rgba(15, 23, 42, 0.97);
                --soft: #111d30;
                --text: #e8eef8;
                --muted: #9fb0c8;
                --border: #273850;
                --shadow: 0 12px 30px rgba(0, 0, 0, 0.30);
            }
        }

        html, body, [class*="css"] {
            font-family: Inter, "Segoe UI", Arial, sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(37,99,235,.10), transparent 26%),
                var(--bg);
        }

        .block-container {
            max-width: 1840px;
            padding-top: .8rem;
            padding-bottom: 1rem;
        }

        #MainMenu, footer, header { visibility: hidden; }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f2f62, #081b38);
            border-right: 1px solid rgba(255,255,255,.08);
        }

        [data-testid="stSidebar"] * { color: #f8fafc; }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label { color: #d8e5f7 !important; }

        .sidebar-brand, .sidebar-card {
            border: 1px solid rgba(255,255,255,.12);
            background: rgba(255,255,255,.075);
            border-radius: 16px;
        }

        .sidebar-brand {
            padding: 1rem;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, rgba(59,130,246,.27), rgba(29,78,216,.10));
        }

        .sidebar-title { font-size: 1.25rem; font-weight: 850; }
        .sidebar-subtitle { color: #bdd1ed; font-size: .8rem; line-height: 1.45; }
        .sidebar-card { margin: .75rem 0; padding: .85rem .95rem; }
        .sidebar-label {
            color: #94b6e0;
            font-size: .67rem;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: .35rem;
        }
        .sidebar-value { color: white; font-size: .84rem; font-weight: 650; line-height: 1.65; }
        .sidebar-value a { color: #93c5fd !important; text-decoration: none; }

        .app-header {
            padding: 1.15rem 1.35rem;
            margin-bottom: .75rem;
            border-radius: 22px;
            color: white;
            background:
                radial-gradient(circle at 88% 10%, rgba(191,219,254,.35), transparent 21%),
                linear-gradient(115deg, #0b397d, #1d4ed8 55%, #2563eb);
            box-shadow: 0 16px 34px rgba(29,78,216,.24);
        }

        .header-row { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
        .header-title { margin: 0; font-size: clamp(1.65rem, 2vw, 2.25rem); font-weight: 900; }
        .header-subtitle { margin-top: .2rem; color: #dbeafe; font-size: .9rem; }
        .badge-row { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: .4rem; }
        .badge {
            padding: .4rem .7rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,.25);
            background: rgba(255,255,255,.14);
            font-size: .72rem;
            font-weight: 750;
        }

        div[data-testid="stMetric"] {
            min-height: 112px;
            padding: .85rem .9rem;
            border-radius: 17px;
            background: var(--card);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
        }

        div[data-testid="stMetricLabel"] { color: var(--muted); }
        div[data-testid="stMetricValue"] { color: var(--text); }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--border);
            border-radius: 18px;
            background: var(--card);
            box-shadow: var(--shadow);
        }

        div[data-testid="stDataFrame"] {
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 13px;
        }

        .stButton > button,
        .stDownloadButton > button {
            width: 100%;
            min-height: 43px;
            border-radius: 12px;
            font-weight: 800;
            transition: transform .18s ease, box-shadow .18s ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 9px 18px rgba(37,99,235,.18);
        }

        .image-frame img {
            height: 300px;
            width: 100%;
            object-fit: contain;
            background: #06101f;
            border: 1px solid var(--border);
            border-radius: 14px;
        }

        .footer-bar {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: .8rem;
            margin-top: .75rem;
            padding: .72rem .9rem;
            border-radius: 14px;
            color: #dbeafe;
            background: linear-gradient(100deg, #081b38, #0f2f62);
            font-size: .7rem;
        }

        .footer-bar a { color: #93c5fd !important; text-decoration: none; font-weight: 750; }

        @media (max-width: 1100px) {
            .header-row { align-items: flex-start; flex-direction: column; }
            .badge-row { justify-content: flex-start; }
            .image-frame img { height: 230px; }
        }
    </style>
    """
)


@st.cache_resource(show_spinner=False)
def load_detector():
    return PCBDetector()


def to_pil_rgb(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if hasattr(image, "shape"):
        if len(image.shape) == 2:
            return Image.fromarray(image).convert("RGB")
        if image.shape[2] == 4:
            return Image.fromarray(image[:, :, :3]).convert("RGB")
        return Image.fromarray(image).convert("RGB")

    raise TypeError("Unsupported image returned by the detector.")


def image_to_png_bytes(image):
    output = io.BytesIO()
    to_pil_rgb(image).save(output, format="PNG")
    return output.getvalue()


def format_file_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1024**2:.2f} MB"


def normalise_confidence(value):
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0

    if 0 <= confidence <= 1:
        confidence *= 100

    return max(0.0, min(confidence, 100.0))


def first_value(data, keys, default=None):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def format_bbox(detection):
    bbox = first_value(
        detection,
        ["bbox", "box", "bounding_box", "coordinates", "xyxy"],
    )

    if bbox is None and all(key in detection for key in ("x1", "y1", "x2", "y2")):
        bbox = [detection["x1"], detection["y1"], detection["x2"], detection["y2"]]

    if isinstance(bbox, dict):
        bbox = [bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")]

    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            return ", ".join(str(round(float(value), 1)) for value in bbox[:4])
        except (TypeError, ValueError):
            return ", ".join(str(value) for value in bbox[:4])

    return ""


def severity_from_confidence(confidence):
    if confidence >= 80:
        return "High"
    if confidence >= 50:
        return "Medium"
    return "Low"


def build_detection_table(detections):
    rows = []

    for number, detection in enumerate(detections or [], start=1):
        if not isinstance(detection, dict):
            detection = {"defect_type": str(detection), "confidence": 0}

        defect_type = first_value(
            detection,
            ["defect_type", "defect", "class_name", "class", "label", "name"],
            "Unknown Defect",
        )
        confidence = normalise_confidence(
            first_value(detection, ["confidence", "conf", "score", "probability"], 0)
        )

        rows.append(
            {
                "Serial Number": number,
                "Defect Type": str(defect_type),
                "Confidence (%)": round(confidence, 2),
                "Severity": severity_from_confidence(confidence),
                "Bounding Box (x1, y1, x2, y2)": format_bbox(detection),
            }
        )

    table = pd.DataFrame(rows)

    if table.empty:
        return pd.DataFrame(
            columns=["Serial Number", "Defect Type", "Confidence (%)", "Severity"]
        )

    bbox_column = "Bounding Box (x1, y1, x2, y2)"
    if not table[bbox_column].astype(bool).any():
        table = table.drop(columns=[bbox_column])

    return table


def calculate_health(defect_count):
    if defect_count == 0:
        return 100, "Excellent"
    if defect_count == 1:
        return 90, "Good"
    if defect_count == 2:
        return 80, "Needs Inspection"
    if defect_count == 3:
        return 70, "Needs Inspection"
    return 60, "Critical"


def style_detection_row(row):
    if row["Severity"] == "High":
        style = "background-color:rgba(220,38,38,.13);color:#991b1b;font-weight:650;"
    elif row["Severity"] == "Medium":
        style = "background-color:rgba(217,119,6,.13);color:#92400e;font-weight:650;"
    else:
        style = "background-color:rgba(37,99,235,.10);color:#1e40af;font-weight:650;"
    return [style] * len(row)


def create_bar_chart(counts):
    figure, axis = plt.subplots(figsize=(7.2, 3.0))
    labels = list(counts.keys())
    values = list(counts.values())
    palette = ["#2563eb", "#0ea5e9", "#7c3aed", "#14b8a6", "#f59e0b", "#ef4444"]

    bars = axis.bar(labels, values, color=palette[: len(labels)], edgecolor="white")
    axis.set_title("Detected Defects by Type", fontweight="bold")
    axis.set_ylabel("Count")
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="x", rotation=12, labelsize=8)

    for bar in bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            str(int(bar.get_height())),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    figure.tight_layout()
    return figure


def create_pie_chart(counts):
    figure, axis = plt.subplots(figsize=(7.2, 3.0))
    palette = ["#2563eb", "#0ea5e9", "#7c3aed", "#14b8a6", "#f59e0b", "#ef4444"]

    axis.pie(
        list(counts.values()),
        labels=list(counts.keys()),
        autopct=lambda value: f"{value:.1f}%",
        startangle=90,
        colors=palette[: len(counts)],
        wedgeprops={"linewidth": 1.1, "edgecolor": "white"},
        textprops={"fontsize": 8},
    )
    axis.set_title("Defect Percentage Distribution", fontweight="bold")
    axis.axis("equal")
    figure.tight_layout()
    return figure


def build_pdf_report(result):
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=32,
        leftMargin=32,
        topMargin=28,
        bottomMargin=30,
        title=f"{APP_NAME} Inspection Report",
        author=DEVELOPER_NAME,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=colors.HexColor("#0f2f62"),
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=8.5,
        textColor=colors.HexColor("#64748b"),
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        textColor=colors.HexColor("#1d4ed8"),
        spaceBefore=8,
        spaceAfter=6,
    )

    story = [
        Paragraph(APP_NAME, title_style),
        Paragraph("AI Powered PCB Quality Inspection Report", subtitle_style),
        Paragraph("Inspection Summary", section_style),
    ]

    timestamp = result["timestamp"]
    summary_rows = [
        ["Date", timestamp.strftime("%d %B %Y"), "Time", timestamp.strftime("%I:%M:%S %p")],
        ["Image Source", result["source_name"], "File Name", result["filename"]],
        ["File Size", result["file_size_text"], "Resolution", result["resolution"]],
        ["Model", MODEL_NAME, "Threshold", f'{result["threshold"]:.0%}'],
        ["Processing", f'{result["elapsed"]:.3f} sec', "Total Defects", str(result["total_defects"])],
        ["Top Confidence", f'{result["highest_confidence"]:.2f}%', "Health", f'{result["health_score"]}/100 - {result["status"]}'],
    ]

    summary_table = Table(
        summary_rows,
        colWidths=[1.15 * inch, 1.55 * inch, 1.25 * inch, 1.65 * inch],
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dbe5f1")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([summary_table, Spacer(1, 10), Paragraph("Image Comparison", section_style)])

    image_table = Table(
        [
            [Paragraph("<b>Original PCB</b>", styles["BodyText"]), Paragraph("<b>Detection Result</b>", styles["BodyText"])],
            [
                PDFImage(io.BytesIO(result["original_bytes"]), width=3.18 * inch, height=2.12 * inch),
                PDFImage(io.BytesIO(result["detection_bytes"]), width=3.18 * inch, height=2.12 * inch),
            ],
        ],
        colWidths=[3.28 * inch, 3.28 * inch],
    )
    image_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dbe5f1")),
            ]
        )
    )
    story.extend([image_table, Spacer(1, 10), Paragraph("Detection Details", section_style)])

    table = result["table"]
    if table.empty:
        story.append(Paragraph("No defects were detected.", styles["BodyText"]))
    else:
        headers = list(table.columns)
        rows = [headers]
        for _, row in table.iterrows():
            values = []
            for column in headers:
                value = row[column]
                if column == "Confidence (%)":
                    value = f"{value:.2f}%"
                values.append(str(value))
            rows.append(values)

        widths = [0.6 * inch, 2.1 * inch, 1.1 * inch, 0.9 * inch]
        if len(headers) == 5:
            widths.append(2.0 * inch)

        detail_table = Table(rows, repeatRows=1, colWidths=widths)
        detail_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d4ed8")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#dbe5f1")),
                ]
            )
        )
        story.append(detail_table)

    story.extend(
        [
            Spacer(1, 14),
            Paragraph(
                f"Generated by {APP_NAME} | Developed by {DEVELOPER_NAME} | YOLOv8 | OpenCV | Python | Streamlit",
                subtitle_style,
            ),
        ]
    )

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def summary_item(label, value):
    """Native Streamlit summary item. No HTML is used."""
    with st.container(border=True):
        st.caption(label)
        st.markdown(f"**{value}**")


def render_summary(result):
    """Native summary panel, preventing HTML from ever appearing as code."""
    with st.container(border=True):
        st.subheader("🧾 Inspection Summary")

        rows = [
            ("Inspection Date", result["timestamp"].strftime("%d %b %Y")),
            ("Inspection Time", result["timestamp"].strftime("%I:%M:%S %p")),
            ("Image Source", result["source_name"]),
            ("File Name", result["filename"]),
            ("File Size", result["file_size_text"]),
            ("Resolution", result["resolution"]),
            ("Model Name", MODEL_NAME),
            ("Confidence Threshold", f'{result["threshold"]:.0%}'),
            ("Processing Time", f'{result["elapsed"]:.3f} sec'),
            ("Total Defects", str(result["total_defects"])),
            ("Highest Confidence", f'{result["highest_confidence"]:.2f}%'),
            ("PCB Health Score", f'{result["health_score"]}/100'),
        ]

        for index in range(0, len(rows), 2):
            columns = st.columns(2)
            with columns[0]:
                summary_item(*rows[index])
            with columns[1]:
                summary_item(*rows[index + 1])

        status = result["status"]
        if status in {"Excellent", "Good"}:
            st.success(f"PCB status: {status}", icon="✅")
        elif status == "Needs Inspection":
            st.warning("PCB status: Needs Inspection", icon="⚠️")
        else:
            st.error("PCB status: Critical", icon="🚨")


if "inspection_result" not in st.session_state:
    st.session_state.inspection_result = None
if "input_signature" not in st.session_state:
    st.session_state.input_signature = None


safe_html(
    f"""
    <div class="sidebar-brand">
        <div class="sidebar-title">🔍 {APP_NAME}</div>
        <div class="sidebar-subtitle">Industrial AI vision for automated PCB quality inspection.</div>
    </div>
    """,
    sidebar=True,
)

st.sidebar.markdown("### ⚙ Inspection Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.25,
    step=0.05,
)

safe_html(
    f"""
    <div class="sidebar-card">
        <div class="sidebar-label">Model Information</div>
        <div class="sidebar-value">Engine: {MODEL_NAME}<br>Task: PCB Defect Detection<br>Mode: Real-Time Inference</div>
    </div>
    <div class="sidebar-card">
        <div class="sidebar-label">Application</div>
        <div class="sidebar-value">Interface: Industrial Dashboard<br>Input: Upload or Camera<br>Reports: Image, CSV and PDF</div>
    </div>
    <div class="sidebar-card">
        <div class="sidebar-label">About</div>
        <div class="sidebar-value">Developed by {DEVELOPER_NAME}<br><a href="{GITHUB_URL}" target="_blank">Open GitHub Profile ↗</a></div>
    </div>
    """,
    sidebar=True,
)

if not REPORTLAB_AVAILABLE:
    st.sidebar.warning("Install reportlab to enable PDF reports.", icon="⚠️")


safe_html(
    f"""
    <div class="app-header">
        <div class="header-row">
            <div>
                <h1 class="header-title">🔍 {APP_NAME}</h1>
                <div class="header-subtitle">AI Powered PCB Quality Inspection Dashboard</div>
            </div>
            <div class="badge-row">
                <span class="badge">🤖 {MODEL_NAME}</span>
                <span class="badge">● Real-Time Detection</span>
                <span class="badge">📷 Upload or Camera</span>
            </div>
        </div>
    </div>
    """
)


with st.container(border=True):
    st.subheader("📥 Select PCB Image Source")
    st.caption("Upload an existing image or capture a new photo using your camera.")

    input_mode = st.radio(
        "Image Source",
        ["Upload Image", "Live Camera Capture"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if input_mode == "Upload Image":
        selected_file = st.file_uploader(
            "Upload PCB Image",
            type=["jpg", "jpeg", "png"],
        )
        source_name = "Uploaded Image"
    else:
        selected_file = st.camera_input("Capture PCB Image")
        source_name = "Live Camera Capture"

    start_inspection = st.button(
        "🚀 Start Inspection",
        type="primary",
        use_container_width=True,
        disabled=selected_file is None,
    )


selected_bytes = None
if selected_file is not None:
    selected_bytes = selected_file.getvalue()
    signature = hashlib.md5(selected_bytes).hexdigest()

    if st.session_state.input_signature not in {None, signature}:
        st.session_state.inspection_result = None

    st.session_state.input_signature = signature


if start_inspection and selected_file is not None:
    try:
        original_image = Image.open(io.BytesIO(selected_bytes)).convert("RGB")
    except Exception as error:
        st.error(f"Unable to read the selected image: {error}")
        st.stop()

    temporary_path = None
    try:
        original_name = getattr(selected_file, "name", "camera_capture.jpg")
        suffix = Path(original_name).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            suffix = ".jpg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            if suffix == ".png":
                original_image.save(temporary_file.name, format="PNG")
            else:
                original_image.save(temporary_file.name, format="JPEG", quality=95)
            temporary_path = temporary_file.name

        detector = load_detector()
        started_at = time.perf_counter()

        with st.spinner("Running YOLOv8 PCB inspection..."):
            annotated_image, detections = detector.predict(
                temporary_path,
                confidence_threshold,
            )

        elapsed_time = time.perf_counter() - started_at

    except Exception as error:
        st.error(f"PCB inspection failed: {error}")
        st.stop()

    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)

    detection_table = build_detection_table(detections)
    total_defects = len(detection_table)
    highest_confidence = (
        float(detection_table["Confidence (%)"].max())
        if not detection_table.empty
        else 0.0
    )
    health_score, status = calculate_health(total_defects)
    width, height = original_image.size
    timestamp = datetime.now()

    filename = getattr(selected_file, "name", None)
    if not filename or input_mode == "Live Camera Capture":
        filename = f"camera_capture_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"

    result = {
        "filename": filename,
        "source_name": source_name,
        "file_size_text": format_file_size(len(selected_bytes)),
        "resolution": f"{width} × {height} px",
        "threshold": confidence_threshold,
        "elapsed": elapsed_time,
        "timestamp": timestamp,
        "table": detection_table,
        "total_defects": total_defects,
        "highest_confidence": highest_confidence,
        "health_score": health_score,
        "status": status,
        "original_bytes": image_to_png_bytes(original_image),
        "detection_bytes": image_to_png_bytes(annotated_image),
    }
    result["pdf_bytes"] = build_pdf_report(result)
    st.session_state.inspection_result = result


result = st.session_state.inspection_result

if result is None:
    with st.container(border=True):
        st.markdown("## 🧠 Inspection workspace ready")
        st.write(
            "Select an image source, provide a PCB image and click "
            "**Start Inspection** to generate results."
        )
else:
    table = result["table"]

    kpi_columns = st.columns(5)
    kpi_columns[0].metric("🔍 Total Defects", result["total_defects"], "Objects found")
    kpi_columns[1].metric("🎯 Highest Confidence", f'{result["highest_confidence"]:.2f}%', "Top prediction")
    kpi_columns[2].metric("⚡ Processing Time", f'{result["elapsed"]:.3f}s', "Inference")
    kpi_columns[3].metric("🤖 Model", MODEL_NAME, "Object detection")
    kpi_columns[4].metric("🛡️ PCB Health", f'{result["health_score"]}/100', result["status"])

    st.progress(result["health_score"] / 100, text=f'PCB Health: {result["status"]}')

    original_column, detection_column = st.columns(2)

    with original_column:
        with st.container(border=True):
            st.subheader("📷 Original PCB Image")
            st.image(result["original_bytes"], use_container_width=True)
            info_columns = st.columns(3)
            info_columns[0].caption(f'**Resolution**\n\n{result["resolution"]}')
            info_columns[1].caption(f'**File Size**\n\n{result["file_size_text"]}')
            info_columns[2].caption(f'**Source**\n\n{result["source_name"]}')

    with detection_column:
        with st.container(border=True):
            st.subheader("🎯 Detection Result")
            st.image(result["detection_bytes"], use_container_width=True)
            info_columns = st.columns(3)
            info_columns[0].caption(f'**Objects**\n\n{result["total_defects"]}')
            info_columns[1].caption(f'**Threshold**\n\n{result["threshold"]:.0%}')
            info_columns[2].caption(f'**Inference**\n\n{result["elapsed"]:.3f} sec')

    table_column, summary_column = st.columns([1.55, 0.95])

    with table_column:
        with st.container(border=True):
            st.subheader("📋 Detection Table")

            if table.empty:
                st.success("No defects were detected.", icon="✅")
            else:
                styled_table = (
                    table.style
                    .apply(style_detection_row, axis=1)
                    .format({"Confidence (%)": "{:.2f}%"})
                )
                st.dataframe(
                    styled_table,
                    use_container_width=True,
                    hide_index=True,
                    height=min(300, 38 + len(table) * 35),
                )

    with summary_column:
        render_summary(result)

    chart_left, chart_right = st.columns(2)

    if table.empty:
        with chart_left:
            with st.container(border=True):
                st.subheader("📊 Defect Distribution")
                st.info("No defect counts are available.")
        with chart_right:
            with st.container(border=True):
                st.subheader("◉ Defect Composition")
                st.info("The PCB passed the current threshold.")
    else:
        counts = Counter(table["Defect Type"])

        with chart_left:
            with st.container(border=True):
                st.subheader("📊 Defect Distribution")
                bar_chart = create_bar_chart(counts)
                st.pyplot(bar_chart, use_container_width=True)
                plt.close(bar_chart)

        with chart_right:
            with st.container(border=True):
                st.subheader("◉ Defect Composition")
                pie_chart = create_pie_chart(counts)
                st.pyplot(pie_chart, use_container_width=True)
                plt.close(pie_chart)

    with st.container(border=True):
        st.subheader("⬇ Download Center")
        download_columns = st.columns(3)
        csv_bytes = table.to_csv(index=False).encode("utf-8")

        download_columns[0].download_button(
            "📥 Download Detection Image",
            data=result["detection_bytes"],
            file_name="pcb_detection_result.png",
            mime="image/png",
            use_container_width=True,
        )
        download_columns[1].download_button(
            "📄 Export CSV",
            data=csv_bytes,
            file_name="pcb_detection_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if result["pdf_bytes"] is not None:
            download_columns[2].download_button(
                "🧾 Generate PDF Report",
                data=result["pdf_bytes"],
                file_name="pcb_inspection_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            download_columns[2].button(
                "🧾 Install ReportLab for PDF",
                disabled=True,
                use_container_width=True,
            )


safe_html(
    f"""
    <div class="footer-bar">
        <div><b>{APP_NAME}</b> · Industrial PCB Quality Inspection</div>
        <div>
            Developed by <b>{DEVELOPER_NAME}</b> ·
            <a href="{GITHUB_URL}" target="_blank">GitHub Profile</a> ·
            YOLOv8 · OpenCV · Python · Streamlit
        </div>
    </div>
    """
)