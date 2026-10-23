"""PDF report generation with ReportLab (tables + a native vector chart; no browser/kaleido needed)."""
from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from . import eda
from .forecasting import SelectionResult

LIMITATIONS = [
    "Only about 2.4 years (29 monthly points) of history: seasonal estimates rest on two November peaks.",
    "The hold-out period (3 months) is tiny and contains no November; hold-out scores are indicative only.",
    "Prediction intervals were narrower than nominal in rolling validation; treat them as optimistic.",
    "Regressors are calendar flags only; price, promotion, macro or capacity drivers are not in the data.",
    "Patterns described are associations in this dataset; no causal claims are made.",
    "Revenue basis excludes the configured statuses (default: Cancelled); other non-shipped statuses are included.",
]


def _tbl(df: pd.DataFrame, col_widths=None, font: int = 7) -> Table:
    data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2a3c")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), font),
                           ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9aa4b5")),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f8")]),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


def _chart(history: pd.Series, fc: pd.DataFrame) -> Drawing:
    d = Drawing(480, 220)
    lp = LinePlot()
    lp.x, lp.y, lp.width, lp.height = 40, 25, 420, 170
    h = history.tail(30)
    xs = list(range(len(h)))
    fx = [len(h) - 1 + i for i in range(0, len(fc) + 1)]
    last = float(h.iloc[-1])
    lp.data = [list(zip(xs, h.values)),
               list(zip(fx, [last] + list(fc["forecast"].values))),
               list(zip(fx[1:], fc["upper_95"].values)), list(zip(fx[1:], fc["lower_95"].values))]
    for i, c in enumerate((colors.HexColor("#222222"), colors.HexColor("#ff4b2b"), colors.HexColor("#ffa28f"),
                           colors.HexColor("#ffa28f"))):
        lp.lines[i].strokeColor, lp.lines[i].strokeWidth = c, 1.6 if i < 2 else 0.8
    lp.xValueAxis.valueMin, lp.xValueAxis.valueMax = 0, len(h) + len(fc)
    lp.xValueAxis.visibleLabels = False
    lp.yValueAxis.labelTextFormat = "%.0f"
    lp.yValueAxis.labels.fontSize = 6
    d.add(lp)
    d.add(String(40, 205, "History (black), forecast (red) and 95% prediction interval (light red)", fontSize=7))
    return d


def build_pdf(df: pd.DataFrame, audit: dict, audit_log: list[dict], sel: SelectionResult | None,
              forecast: pd.DataFrame | None, notes: list[str], target_label: str) -> bytes:
    """Create the PDF report and return its bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm,
                            bottomMargin=1.5 * cm, title="AutoCast AI report")
    ss = getSampleStyleSheet()
    h1, h2 = ss["Heading1"], ss["Heading2"]
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, textColor=colors.HexColor("#555555"))
    el: list = [Paragraph("AutoCast AI - Sales Forecasting Report", h1),
                Paragraph(f"Generated {datetime.now():%Y-%m-%d %H:%M}. Target: {target_label}.", small), Spacer(1, 8)]

    k = eda.kpis(df)
    dmin, dmax = audit["date_range"]
    el += [Paragraph("1. Dataset summary", h2),
           Paragraph(f"{audit['shape'][0]:,} order lines x {audit['shape'][1]} columns; orders from {dmin:%d %b %Y} to "
                     f"{dmax:%d %b %Y}. Gross revenue ${k['revenue']:,.0f}; {k['units']:,} units; {k['orders']:,} distinct "
                     f"orders; {k['customers']} customers in {k['countries']} countries. Revenue, units and orders are "
                     "different measures and are never mixed.", body)]

    el += [Paragraph("2. Data quality findings", h2), _tbl(audit["suspicious"])]
    for n in audit.get("notes", []):
        el.append(Paragraph("- " + n, body))
    ch = pd.DataFrame(audit_log)[["step", "detail", "rows_affected"]]
    el += [Spacer(1, 6), Paragraph("Cleaning audit log", ss["Heading3"]), _tbl(ch, [3.5 * cm, 11.5 * cm, 2.2 * cm])]

    el += [Paragraph("3. Exploratory analysis", h2)]
    for o in eda.observations(df):
        el.append(Paragraph("- " + o, body))

    if sel is not None:
        ev = sel.seasonal_evidence
        adf = sel.adf
        el += [PageBreak(), Paragraph("4. Time-series diagnostics", h2),
               Paragraph(f"Monthly series, {len(sel.series)} observations. ADF test (development set): statistic "
                         f"{adf['statistic']:.2f}, p={adf['p_value']:.4f} -> "
                         f"{'stationary' if adf['stationary_5pct'] else 'non-stationary'} at 5%. {ev['reason']}", body)]
        el += [Paragraph("5. Model evaluation", h2),
               Paragraph(f"Rolling-origin validation: {len(sel.origins)} expanding-window folds, "
                         f"{sel.settings.val_horizon}-month horizon; hold-out of {sel.settings.test_periods} months kept "
                         "untouched until selection.", body)]
        fb = sel.family_best.copy()
        cols = [c for c in ("family", "model", "mae", "rmse", "wape", "mape", "coverage_95", "aic", "status") if c in fb]
        fb = fb[cols].copy()
        for c in ("mae", "rmse", "aic"):
            if c in fb:
                fb[c] = fb[c].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}")
        for c in ("wape", "mape", "coverage_95"):
            if c in fb:
                fb[c] = fb[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.3f}")
        el += [Paragraph("Validation (best candidate per family)", ss["Heading3"]), _tbl(fb, font=6.5)]
        tt = sel.test_table[["family", "model", "mae", "rmse", "wape", "mape"]].copy()
        for c in ("mae", "rmse"):
            tt[c] = tt[c].map(lambda v: f"{v:,.0f}")
        for c in ("wape", "mape"):
            tt[c] = tt[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.3f}")
        el += [Spacer(1, 6), Paragraph("Hold-out test (reported after selection)", ss["Heading3"]), _tbl(tt, font=6.5),
               Spacer(1, 6), Paragraph(sel.rationale, body)]
        for n in sel.notes:
            el.append(Paragraph("- " + n, body))
        if forecast is not None and len(forecast):
            f = forecast.copy()
            f.index = [d.strftime("%b %Y") for d in f.index]
            f = f.round(0).astype(int).reset_index().rename(columns={"index": "month", "month": "month"})
            el += [PageBreak(), Paragraph("6. Forecast with 95% prediction intervals", h2), _tbl(f),
                   Spacer(1, 8), _chart(sel.series, forecast)]
    el += [Paragraph("7. Limitations", h2)] + [Paragraph("- " + x, body) for x in LIMITATIONS + notes]
    doc.build(el)
    return buf.getvalue()
