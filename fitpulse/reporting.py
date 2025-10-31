"""
Reporting utilities for FitPulse: CSV and PDF export of anomaly results and summaries.
"""

import io
from typing import Dict, Any, Optional

import pandas as pd

# Optional dependency: reportlab (used for PDF export)
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except ImportError:  # pragma: no cover
    REPORTLAB_AVAILABLE = False


def build_anomaly_summary_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a flat dataframe summarizing anomaly counts and rates per metric.
    Expects results from EnhancedFitnessPipeline.run_comprehensive_analysis or
    similar with an 'anomaly_detection' key.
    """
    rows = []
    anomaly_detection = results.get("anomaly_detection", {}) if isinstance(results, dict) else {}
    for metric, info in anomaly_detection.items():
        anomalies = info.get("anomalies")
        if anomalies is None:
            continue
        total = int(len(anomalies))
        count = int(anomalies.sum())
        rate = (count / total * 100.0) if total > 0 else 0.0
        rows.append({
            "metric": metric,
            "anomaly_count": count,
            "anomaly_rate_pct": round(rate, 2),
        })
    return pd.DataFrame(rows)


def dataframe_to_csv_bytes(df: pd.DataFrame, index: bool = False) -> bytes:
    return df.to_csv(index=index).encode("utf-8")


def build_pdf_report(
    title: str,
    overview: Dict[str, Any],
    anomaly_summary_df: Optional[pd.DataFrame] = None,
) -> bytes:
    """
    Create a simple PDF report and return as bytes for download in Streamlit.
    """
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(title, styles["Title"]))
    elements.append(Spacer(1, 12))

    # Overview section
    elements.append(Paragraph("Overview", styles["Heading2"]))
    for k, v in overview.items():
        elements.append(Paragraph(f"<b>{k.replace('_', ' ').title()}:</b> {v}", styles["BodyText"]))
    elements.append(Spacer(1, 12))

    # Anomaly summary table
    if anomaly_summary_df is not None and not anomaly_summary_df.empty:
        elements.append(Paragraph("Anomaly Summary", styles["Heading2"]))
        table_data = [list(anomaly_summary_df.columns)] + anomaly_summary_df.values.tolist()
        table = Table(table_data, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(table)

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


