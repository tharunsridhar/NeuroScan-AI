from __future__ import annotations

import os
from datetime import datetime

import cv2
import numpy as np

from utils.config import MRI_PARAMS, REPORT_DIR
from utils.io_utils import safe

def build_overlays(orig_bgr: np.ndarray, mask: np.ndarray, heatmap_2d: np.ndarray, size_info: dict) -> dict:
    mask_vis = np.uint8(mask * 255)
    mask_col = cv2.applyColorMap(mask_vis, cv2.COLORMAP_HOT)
    mask_ov = cv2.addWeighted(orig_bgr, 0.7, mask_col, 0.3, 0)
    if size_info.get('bbox'):
        b = size_info['bbox']
        cv2.rectangle(mask_ov, (b['x'], b['y']), (b['x'] + b['w'], b['y'] + b['h']), (0, 255, 0), 2)
        cv2.putText(mask_ov, f"{size_info['diameter_cm']}cm", (b['x'], b['y'] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    hmap_c = cv2.applyColorMap(np.uint8(255 * heatmap_2d), cv2.COLORMAP_JET)
    gcam_ov = cv2.addWeighted(orig_bgr, 0.6, hmap_c, 0.4, 0)
    return {'mask_ov': mask_ov, 'gcam_ov': gcam_ov, 'hmap_r': heatmap_2d}


def make_risk_chart(size_info, risk_info, shape_info, mass_info) -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    DARK = '#14284A'
    RED = '#C0392B'
    ORG = '#E67E22'
    GRN = '#27AE60'
    BLU = '#2980B9'
    PUR = '#8E44AD'
    TEA = '#16A085'
    BG = '#F5F7FA'

    def rc(v):
        return RED if v > 0.6 else ORG if v > 0.3 else GRN
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.2))
    fig.patch.set_facecolor('#FFFFFF')
    tp = max(min(size_info['tumor_percent'], 100), 0)
    ax = axes[0]
    ax.set_facecolor('#FFFFFF')
    _, _, ats = ax.pie([tp, 100 - tp], colors=[RED, GRN], autopct='%1.1f%%', startangle=90, pctdistance=0.78, wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2), textprops={'fontsize': 7.5, 'color': '#333333'})
    for at in ats:
        at.set_fontsize(7)
        at.set_fontweight('bold')
        at.set_color('white')
    ax.text(0, 0, f'{tp:.1f}%\nTumor', ha='center', va='center', fontsize=8, fontweight='bold', color=DARK)
    ax.set_title('Tumor vs Brain Coverage', fontsize=9, fontweight='bold', color=DARK, pad=8)
    ax.legend(handles=[mpatches.Patch(facecolor=RED, label='Tumor'), mpatches.Patch(facecolor=GRN, label='Healthy Brain')], fontsize=6.5, loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax = axes[1]
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color('#DDDDDD')
    cats = ['Mass Effect', 'Shape', 'Size', 'Confidence']
    vals = [round(min(mass_info['shift_mm'] / 10.0, 1.0), 2), round(shape_info['irregularity'] if shape_info else 0, 2), round(min(size_info['tumor_percent'] / 30.0, 1.0), 2), round(min(risk_info['score'] * 1.2, 1.0), 2)]
    bars = ax.barh(cats, vals, color=[rc(v) for v in vals], height=0.45, edgecolor='white', linewidth=0.8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Score (0-1)', fontsize=7, color='#555555')
    ax.tick_params(axis='y', labelsize=7.5)
    ax.tick_params(axis='x', labelsize=6.5)
    ax.set_title('Risk Factor Analysis', fontsize=9, fontweight='bold', color=DARK, pad=8)
    for x in [0.3, 0.6]:
        ax.axvline(x, color='#CCCCCC', linewidth=0.8, linestyle='--')
    for bar, v in zip(bars, vals):
        ax.text(min(v + 0.03, 0.97), bar.get_y() + bar.get_height() / 2, f'{v:.2f}', va='center', ha='left', fontsize=7, fontweight='bold', color='#333333')
    ax.legend(handles=[mpatches.Patch(facecolor=RED, label='High > 0.6'), mpatches.Patch(facecolor=ORG, label='Mod  > 0.3'), mpatches.Patch(facecolor=GRN, label='Low <= 0.3')], fontsize=6, loc='lower right', frameon=True, framealpha=0.85)
    ax = axes[2]
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color('#DDDDDD')
    if shape_info:
        lbls = ['Irregularity', 'Compactness\n(norm)', 'Eccentricity', 'Inv.Convexity']
        vals2 = [shape_info['irregularity'], min(shape_info['compactness'] / 3.5, 1.0), shape_info['eccentricity'], round(1.0 - shape_info['convexity'], 3)]
        b2 = ax.bar(range(len(lbls)), vals2, color=[PUR, BLU, TEA, ORG], width=0.55, edgecolor='white', linewidth=0.8)
        ax.set_xticks(range(len(lbls)))
        ax.set_xticklabels(lbls, fontsize=6.2)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score', fontsize=7, color='#555555')
        ax.tick_params(axis='y', labelsize=6.5)
        ax.axhline(0.5, color='#AAAAAA', linewidth=0.8, linestyle='--', label='Threshold')
        ax.legend(fontsize=6, loc='upper right', frameon=True, framealpha=0.85)
        for bar, v in zip(b2, vals2):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.025, f'{v:.2f}', ha='center', fontsize=7, fontweight='bold', color='#333333')
    else:
        ax.text(0.5, 0.5, 'No tumor data', ha='center', va='center', fontsize=10, color='#888888', transform=ax.transAxes)
    ax.set_title('Tumor Shape Metrics', fontsize=9, fontweight='bold', color=DARK, pad=8)
    fig.suptitle('Tumor Risk Analytics', fontsize=11, fontweight='bold', color=DARK, y=1.02)
    plt.tight_layout()
    os.makedirs('/tmp/rpt', exist_ok=True)
    p = '/tmp/rpt/chart.jpg'
    plt.savefig(p, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close()
    return p


NAV = (20, 40, 74)


TXT = (30, 35, 45)


LGY = (200, 205, 210)


MUT = (100, 110, 125)


BGC = (245, 247, 250)


WHT = (255, 255, 255)


def _header(pdf, title, sub1, sub2=''):
    pdf.set_fill_color(*NAV)
    pdf.rect(0, 0, 210, 14, 'F')
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(*WHT)
    pdf.set_xy(10, 2)
    pdf.cell(125, 10, safe(title))
    pdf.set_font('Helvetica', '', 7)
    pdf.set_text_color(180, 200, 230)
    pdf.set_xy(135, 3)
    pdf.cell(65, 4, safe(sub1), align='R')
    pdf.set_xy(135, 8)
    pdf.cell(65, 4, safe(sub2), align='R')
    pdf.set_text_color(*TXT)


def _footer(pdf, pg, total=3):
    pdf.set_xy(10, 287)
    pdf.set_font('Helvetica', 'I', 6.5)
    pdf.set_text_color(*MUT)
    pdf.cell(155, 4, 'AI-generated report. Must be reviewed by a qualified medical professional before clinical use.')
    pdf.set_font('Helvetica', 'B', 7)
    pdf.set_xy(175, 287)
    pdf.cell(25, 4, f'Page {pg} / {total}', align='R')
    pdf.set_text_color(*TXT)


def _secbar(pdf, title, y):
    pdf.set_fill_color(*NAV)
    pdf.rect(10, y, 190, 6, 'F')
    pdf.set_font('Helvetica', 'B', 7.5)
    pdf.set_text_color(*WHT)
    pdf.set_xy(13, y + 0.9)
    pdf.cell(184, 4.5, safe(f'  {title}'))
    pdf.set_text_color(*TXT)
    return y + 7


def _card(pdf, x, y, w, h):
    pdf.set_fill_color(*BGC)
    pdf.set_draw_color(*LGY)
    pdf.rect(x, y, w, h, 'FD')


def _divider(pdf, y):
    pdf.set_draw_color(*LGY)
    pdf.line(10, y, 200, y)


def _kv(pdf, x, y, k, v, kw=32, vw=36, bold_val=False, vc=None):
    pdf.set_font('Helvetica', 'B', 7)
    pdf.set_text_color(*MUT)
    pdf.set_xy(x, y)
    pdf.cell(kw, 4.5, safe(f'{k}:'), ln=False)
    pdf.set_font('Helvetica', 'B' if bold_val else '', 7.5)
    pdf.set_text_color(*(vc if vc else TXT))
    pdf.set_xy(x + kw, y)
    pdf.cell(vw, 4.5, safe(str(v)))
    pdf.set_text_color(*TXT)


def _badge(pdf, x, y, text, color):
    tw = max(len(text) * 2.0 + 6, 18)
    pdf.set_fill_color(*color)
    pdf.rect(x, y, tw, 5.5, 'F')
    pdf.set_font('Helvetica', 'B', 7.5)
    pdf.set_text_color(*WHT)
    pdf.set_xy(x, y)
    pdf.cell(tw, 5.5, safe(text), align='C')
    pdf.set_text_color(*TXT)


def _pt_banner(pdf, patient_name, patient_id='', y_after_header=14):
    if not patient_name:
        return y_after_header
    pdf.set_fill_color(12, 22, 50)
    pdf.rect(0, y_after_header, 210, 6, 'F')
    pdf.set_font('Helvetica', '', 7.5)
    pdf.set_text_color(150, 180, 230)
    pdf.set_xy(10, y_after_header + 1.2)
    id_part = f'   ID: {patient_id}' if patient_id else ''
    pdf.cell(190, 4, safe(f"  Patient: {patient_name}{id_part}   |   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
    pdf.set_text_color(*TXT)
    return y_after_header + 6


def _llm_page(pdf, llm_report, title, sub1, sub2, pg, total, patient_name='', patient_id=''):
    pdf.add_page()
    _header(pdf, title, sub1, sub2)
    y = _pt_banner(pdf, patient_name, patient_id)
    y = max(y, 14) + 2
    _divider(pdf, y)
    y += 4
    for line in llm_report.split('\n'):
        line = safe(line.strip())
        if not line:
            y += 2
            continue
        if y > 278:
            break
        is_sec = any((line.startswith(h) for h in ['1.', '2.', '3.', '4.', '5.', '6.']))
        if is_sec:
            y += 2
            pdf.set_fill_color(*NAV)
            pdf.rect(10, y, 4, 7, 'F')
            pdf.set_fill_color(228, 234, 252)
            pdf.rect(14, y, 186, 7, 'F')
            pdf.set_font('Helvetica', 'B', 8.5)
            pdf.set_text_color(*NAV)
            pdf.set_xy(16, y + 0.8)
            pdf.cell(182, 5.5, safe(f' {line}'))
            pdf.set_text_color(*TXT)
            y += 9.5
        else:
            pdf.set_font('Helvetica', '', 8)
            pdf.set_text_color(45, 50, 60)
            pdf.set_xy(14, y)
            pdf.multi_cell(184, 4.8, line)
            y = pdf.get_y() + 1.5
    _footer(pdf, pg, total)


def pdf_tumor(filename, image_path, label, confidence, size_info, shape_info, mass_info, risk_info, cal_info, severity, clinical, rano, llm_report, orig, mask_overlay, heatmap, gcam_overlay, report_dir, patient_name='', patient_id='', fusion=None, quality=None, overlap=None, comparison=None, reliability=None):
    from fpdf import FPDF
    SC = (180, 30, 30) if severity in ('Severe', 'Moderate') else (20, 110, 50)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    _header(pdf, 'AI Brain Tumor Analysis Report', f'File: {os.path.basename(filename)}', f"Sequence: {MRI_PARAMS['Sequence']}")
    y = _pt_banner(pdf, patient_name, patient_id)
    y = max(y, 14)
    sc_color = (140, 0, 0) if severity == 'Severe' else (155, 75, 0) if severity == 'Moderate' else (0, 110, 50)
    pdf.set_fill_color(*sc_color)
    pdf.rect(10, y, 190, 8, 'F')
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(*WHT)
    banner = f"  SEVERITY: {severity.upper()}   |   PREDICTION: {label.upper()}   |   CONFIDENCE: {confidence:.2%}   |   GROWTH RISK: {risk_info['risk'].upper()}"
    pdf.set_xy(10, y + 2)
    pdf.cell(190, 4, safe(banner), align='C')
    pdf.set_text_color(*TXT)
    y += 10
    y = _secbar(pdf, 'ANALYSIS SUMMARY', y)
    _card(pdf, 10, y, 190, 38)
    ROW = 7.2
    c1 = [(12, 'Tumor Type', label.upper()), (12, 'Confidence', f'{confidence:.2%}'), (12, 'Uncertainty', cal_info['uncertainty']), (12, 'Calibration', str(cal_info['cal_score'])), (12, 'Severity', severity)]
    c2 = [(76, 'Tumor Area', f"{size_info['area_cm2']} cm2"), (76, 'Diameter', f"{size_info['diameter_cm']} cm"), (76, 'Est. Volume', f"{size_info['volume_cm3']} cm3"), (76, 'Brain Coverage', f"{size_info['tumor_percent']}%"), (76, 'Bounding Box', f"{size_info['bbox']['width_cm']} x {size_info['bbox']['height_cm']} cm" if size_info.get('bbox') else 'N/A')]
    c3 = [(140, 'Laterality', mass_info['laterality']), (140, 'Midline Shift', f"{mass_info['shift_mm']} mm"), (140, 'Compression', mass_info['compression']), (140, 'Sulcal Eff.', mass_info['sulcal']), (140, 'Aggressiveness', f"{risk_info['score']:.3f} / 1.0")]
    for col in [c1, c2, c3]:
        for i, (cx, k, v) in enumerate(col):
            _kv(pdf, cx, y + 2 + i * ROW, k, v, kw=30, vw=34)
    y += 40
    if reliability:
        y = _secbar(pdf, 'DIAGNOSTIC RELIABILITY INDEX (DRI)', y)
        DH = 22
        _card(pdf, 10, y, 190, DH)
        dri_v = reliability.get('dri', 0.0)
        dri_t = reliability.get('dri_tier', '')
        dri_col = (0, 130, 55) if dri_t == 'HIGH' else (155, 100, 0) if dri_t == 'MODERATE' else (160, 30, 30)
        _kv(pdf, 12, y + 2, 'DRI Score', f'{dri_v:.4f}  [{dri_t}]', kw=24, vw=40, bold_val=True, vc=dri_col)
        _kv(pdf, 12, y + 11, 'Gate Status', reliability.get('acceptance_status', ''), kw=24, vw=40, bold_val=True, vc=dri_col)
        _kv(pdf, 110, y + 2, 'Lesion Trust', str(fusion.get('lesion_trust_multiplier', 'N/A') if fusion else 'N/A'), kw=26, vw=20)
        _kv(pdf, 110, y + 11, 'Flags', str(len(reliability.get('escalation_reasons', []))), kw=26, vw=20)
        y += DH + 3
    if fusion:
        y = _secbar(pdf, 'LESION-AWARE MULTI-MODEL FUSION', y)
        FH = 20
        _card(pdf, 10, y, 190, FH)
        _kv(pdf, 12, y + 2, 'Decision', fusion['decision_logic'], kw=30, vw=50)
        _kv(pdf, 12, y + 10, 'Agreement', f"{fusion['agreement_score'] * 100:.0f}%", kw=30, vw=20)
        fw = 45
        for i, (mn, mv) in enumerate(fusion['model_votes'].items()):
            _kv(pdf, 80 + i * fw, y + 2, mn, mv.replace('_', ' ').title(), kw=24, vw=20)
        if quality:
            _kv(pdf, 80, y + 10, 'Scan Quality', str(quality['quality_score']), kw=26, vw=20)
            _kv(pdf, 126, y + 10, 'Reliability', str(risk_info['reliability_score']), kw=24, vw=18)
        if overlap:
            _kv(pdf, 162, y + 10, 'XAI Consistency', str(overlap['explainability_consistency']), kw=30, vw=16)
        y += FH + 3
    y = _secbar(pdf, 'VISUAL ANALYSIS', y)
    tmp = '/tmp/rpt'
    os.makedirs(tmp, exist_ok=True)

    def sv(arr, name, cm_=None):
        p = f'{tmp}/{name}.jpg'
        if cm_ is None:
            cv2.imwrite(p, arr)
        else:
            import matplotlib.pyplot as plt
            plt.imsave(p, arr, cmap=cm_)
        return p
    imgs = [sv(orig, 'orig'), sv(mask_overlay, 'mask'), sv(heatmap, 'hmap', cm_='jet'), sv(gcam_overlay, 'gcam')]
    lbls = ['Original MRI', 'Segmentation', 'GradCAM Heatmap', 'GradCAM Overlay']
    IW, IH = (44, 40)
    GAP = (190 - 4 * IW) / 5
    _card(pdf, 10, y, 190, IH + 12)
    for i, (p, l) in enumerate(zip(imgs, lbls)):
        ix = 10 + GAP + i * (IW + GAP)
        pdf.image(p, x=ix, y=y + 2, w=IW, h=IH)
        pdf.set_font('Helvetica', 'B', 6.5)
        pdf.set_text_color(*MUT)
        pdf.set_xy(ix, y + IH + 3)
        pdf.cell(IW, 4, safe(l), align='C')
    pdf.set_text_color(*TXT)
    y += IH + 16
    y = _secbar(pdf, 'SHAPE ANALYSIS   |   MASS EFFECT   |   RISK SCORE', y)
    SH = 32
    _card(pdf, 10, y, 190, SH)
    RH = 5.8
    pdf.set_draw_color(*LGY)
    pdf.line(74, y + 2, 74, y + SH - 2)
    pdf.line(138, y + 2, 138, y + SH - 2)
    if shape_info:
        _kv(pdf, 12, y + 2, 'Irregularity', str(shape_info['irregularity']), kw=24, vw=18)
        _kv(pdf, 12, y + 2 + RH, 'Compactness', str(shape_info['compactness']), kw=24, vw=18)
        _kv(pdf, 12, y + 2 + RH * 2, 'Convexity', str(shape_info['convexity']), kw=24, vw=18)
        _kv(pdf, 12, y + 2 + RH * 3, 'Border', shape_info['border_def'], kw=24, vw=18)
        _kv(pdf, 12, y + 2 + RH * 4, 'Roughness', shape_info['roughness'], kw=24, vw=18)
    _kv(pdf, 78, y + 2, 'Laterality', mass_info['laterality'], kw=26, vw=28)
    _kv(pdf, 78, y + 2 + RH, 'Shift', f"{mass_info['shift_mm']} mm", kw=26, vw=28)
    _kv(pdf, 78, y + 2 + RH * 2, 'Compression', mass_info['compression'], kw=26, vw=28)
    _kv(pdf, 78, y + 2 + RH * 3, 'Sulcal', mass_info['sulcal'], kw=26, vw=28)
    _kv(pdf, 142, y + 2, 'Aggress. Score', f"{risk_info['score']:.3f} / 1.0", kw=28, vw=22, bold_val=True, vc=SC)
    _kv(pdf, 142, y + 2 + RH, 'Growth Risk', risk_info['risk'], kw=28, vw=22, bold_val=True, vc=SC)
    _kv(pdf, 142, y + 2 + RH * 2, 'Progression', risk_info['progression_risk'], kw=28, vw=22)
    _kv(pdf, 142, y + 2 + RH * 3, 'Uncertainty', cal_info['uncertainty'], kw=28, vw=22)
    y += SH + 3
    if comparison and comparison.get('prior_available'):
        y = _secbar(pdf, 'PRIOR-CASE COMPARISON', y)
        CH = 14
        _card(pdf, 10, y, 190, CH)
        flag = comparison['progression_flag']
        pct = comparison['change_percent']
        fc = (180, 30, 30) if flag == 'Progression' else (0, 110, 50) if flag == 'Regression' else (30, 80, 150)
        _kv(pdf, 12, y + 2, 'Progression Flag', flag, kw=28, vw=24, bold_val=True, vc=fc)
        _kv(pdf, 90, y + 2, 'Area Change', f'{pct:+.1f}%', kw=22, vw=20, bold_val=True, vc=fc)
        y += CH + 3
    if rano:
        y = _secbar(pdf, 'RANO CRITERIA ASSESSMENT', y)
        RAN_H = 22
        _card(pdf, 10, y, 190, RAN_H)
        _kv(pdf, 12, y + 2, 'Size Category', rano['size_cat'], kw=28, vw=28)
        _kv(pdf, 12, y + 11, 'Enhancement', rano['enhancement'], kw=28, vw=28)
        _kv(pdf, 80, y + 2, 'Necrosis', rano['necrosis'], kw=22, vw=44)
        pdf.set_xy(80, y + 11)
        pdf.set_font('Helvetica', 'B', 7)
        pdf.set_text_color(*MUT)
        pdf.cell(22, 4.5, 'Grade:', ln=False)
        pdf.set_font('Helvetica', 'B', 7)
        pdf.set_text_color(*SC)
        pdf.multi_cell(88, 4.5, safe(rano['grade']))
        pdf.set_text_color(*TXT)
        y += RAN_H + 3
    y = _secbar(pdf, 'CLINICAL DECISION SUPPORT', y)
    n = len(clinical['steps'])
    mid = (n + 1) // 2
    ch = mid * 5.8 + 14
    _card(pdf, 10, y, 190, ch)
    pdf.set_xy(12, y + 3)
    pdf.set_font('Helvetica', 'B', 7.5)
    pdf.set_text_color(*MUT)
    pdf.cell(28, 5, 'Urgency Level:', ln=False)
    _badge(pdf, 42, y + 3, f"  {clinical['urgency']}  ", SC)
    for i, step in enumerate(clinical['steps']):
        col = 0 if i < mid else 1
        row = i if i < mid else i - mid
        pdf.set_xy(14 + col * 96, y + 12 + row * 5.8)
        pdf.set_font('Helvetica', '', 7.5)
        pdf.set_text_color(*TXT)
        pdf.cell(90, 4.8, safe(f'  >>  {step}'))
    _footer(pdf, 1, 3)
    chart = make_risk_chart(size_info, risk_info, shape_info, mass_info)
    pdf.add_page()
    _header(pdf, 'Risk Analytics Dashboard', f'File: {os.path.basename(filename)}', f'Prediction: {label.upper()}  |  Confidence: {confidence:.2%}  |  Severity: {severity}')
    y = _pt_banner(pdf, patient_name, patient_id)
    y = max(y, 14) + 2
    pdf.set_font('Helvetica', 'I', 7.5)
    pdf.set_text_color(*MUT)
    pdf.set_xy(10, y)
    pdf.cell(190, 5, safe('Quantitative breakdown of tumor characteristics derived from segmentation and classification models.'))
    pdf.set_text_color(*TXT)
    y += 8
    _card(pdf, 10, y, 190, 96)
    pdf.image(chart, x=12, y=y + 3, w=186, h=90)
    y += 100
    y = _secbar(pdf, 'KEY METRICS AT A GLANCE', y)
    STRIP = 24
    _card(pdf, 10, y, 190, STRIP)
    metrics = [('Tumor Area', f"{size_info['area_cm2']} cm2"), ('Diameter', f"{size_info['diameter_cm']} cm"), ('Volume', f"{size_info['volume_cm3']} cm3"), ('Brain Cov.', f"{size_info['tumor_percent']}%"), ('Midline Shift', f"{mass_info['shift_mm']} mm"), ('Risk Score', f"{risk_info['score']:.3f} / 1.0")]
    cw = 190 / len(metrics)
    for i, (mk, mv) in enumerate(metrics):
        cx = 10 + i * cw
        if i > 0:
            pdf.set_draw_color(*LGY)
            pdf.line(cx, y + 3, cx, y + STRIP - 3)
        pdf.set_font('Helvetica', 'B', 6.5)
        pdf.set_text_color(*MUT)
        pdf.set_xy(cx, y + 4)
        pdf.cell(cw, 4, safe(mk), align='C')
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(*NAV)
        pdf.set_xy(cx, y + 10)
        pdf.cell(cw, 7, safe(mv), align='C')
    pdf.set_text_color(*TXT)
    y += STRIP + 4
    if shape_info:
        y = _secbar(pdf, 'SHAPE DETAIL', y)
        SH2 = 28
        _card(pdf, 10, y, 190, SH2)
        shape_rows = [('Irregularity', shape_info['irregularity'], 'Higher = more irregular boundary'), ('Compactness', round(min(shape_info['compactness'] / 3.5, 1.0), 3), 'Normalised compactness (0-1)'), ('Eccentricity', shape_info['eccentricity'], 'Elongation of fitted ellipse'), ('Convexity', shape_info['convexity'], 'How convex the tumor contour is'), ('Border', shape_info['border_def'], 'Border definition quality'), ('Roughness', shape_info['roughness'], 'Surface texture assessment')]
        scw = 190 / 3
        for i, (sk, sv2, desc) in enumerate(shape_rows):
            ci = i % 3
            ri = i // 3
            cx2 = 10 + ci * scw + 3
            cy2 = y + 3 + ri * 11
            pdf.set_font('Helvetica', 'B', 7)
            pdf.set_text_color(*MUT)
            pdf.set_xy(cx2, cy2)
            pdf.cell(scw - 3, 4.5, safe(f'{sk}:'), ln=False)
            pdf.set_font('Helvetica', 'B', 8)
            pdf.set_text_color(*NAV)
            pdf.set_xy(cx2 + 28, cy2)
            pdf.cell(28, 4.5, safe(str(sv2)), ln=False)
            pdf.set_font('Helvetica', 'I', 6)
            pdf.set_text_color(*MUT)
            pdf.set_xy(cx2, cy2 + 5.5)
            pdf.cell(scw - 3, 4, safe(desc))
        pdf.set_text_color(*TXT)
        y += SH2 + 4
    y = max(y, 248)
    pdf.set_fill_color(255, 248, 225)
    pdf.set_draw_color(230, 175, 60)
    pdf.rect(10, y, 190, 16, 'FD')
    pdf.set_font('Helvetica', 'B', 7.5)
    pdf.set_text_color(150, 90, 0)
    pdf.set_xy(13, y + 2.5)
    pdf.cell(10, 5, '(!)', ln=False)
    pdf.set_font('Helvetica', '', 7)
    pdf.set_xy(21, y + 2.5)
    pdf.multi_cell(176, 5, safe('IMPORTANT: This AI-generated analysis is intended for screening and decision-support purposes only. All findings must be reviewed and validated by a qualified radiologist or neurosurgeon before clinical use. Histopathological confirmation is required for definitive diagnosis.'))
    pdf.set_text_color(*TXT)
    _footer(pdf, 2, 3)
    _llm_page(pdf, llm_report, 'AI Generated Imaging Summary', f'File: {os.path.basename(filename)}', 'Note: AI-generated imaging findings. Histopathological confirmation required.', 3, 3, patient_name, patient_id)
    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    sn = os.path.splitext(os.path.basename(filename))[0]
    pt = f"_{patient_name.replace(' ', '_')}" if patient_name else ''
    out = os.path.join(report_dir, f'{sn}{pt}_{ts}_mri_report.pdf')
    pdf.output(out)
    return out


def pdf_normal(filename, image_path, confidence, llm_report, orig, report_dir, patient_name='', patient_id='', fusion=None, quality=None, reliability=None):
    from fpdf import FPDF
    GRN_C = (0, 120, 55)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    _header(pdf, 'AI Brain MRI Analysis Report', f'File: {os.path.basename(filename)}', f"Sequence: {MRI_PARAMS['Sequence']}")
    y = _pt_banner(pdf, patient_name, patient_id)
    y = max(y, 14)
    pdf.set_fill_color(*GRN_C)
    pdf.rect(10, y, 190, 8, 'F')
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(*WHT)
    pdf.set_xy(10, y + 2)
    pdf.cell(190, 4, safe(f'  RESULT: NORMAL   |   NO TUMOR DETECTED   |   CONFIDENCE: {confidence:.2%}   |   SEVERITY: NONE'), align='C')
    pdf.set_text_color(*TXT)
    y += 10
    if reliability:
        y = _secbar(pdf, 'DIAGNOSTIC RELIABILITY INDEX (DRI)', y)
        DH = 14
        _card(pdf, 10, y, 190, DH)
        dri_v = reliability.get('dri', 0.0)
        dri_t = reliability.get('dri_tier', '')
        dri_col = (0, 130, 55) if dri_t == 'HIGH' else (155, 100, 0) if dri_t == 'MODERATE' else (160, 30, 30)
        _kv(pdf, 12, y + 2, 'DRI', f'{dri_v:.4f}  [{dri_t}]', kw=18, vw=36, bold_val=True, vc=dri_col)
        _kv(pdf, 100, y + 2, 'Gate', reliability.get('acceptance_status', ''), kw=18, vw=36, bold_val=True, vc=dri_col)
        y += DH + 3
    if fusion:
        y = _secbar(pdf, 'MULTI-MODEL FUSION RESULT', y)
        FH = 14
        _card(pdf, 10, y, 190, FH)
        _kv(pdf, 12, y + 2, 'Fused Confidence', f"{fusion['fused_confidence'] * 100:.1f}%", kw=28, vw=20)
        _kv(pdf, 80, y + 2, 'Agreement', f"{fusion['agreement_score'] * 100:.0f}%", kw=20, vw=16)
        if quality:
            _kv(pdf, 140, y + 2, 'Scan Quality', str(quality['quality_score']), kw=26, vw=20)
        y += FH + 3
    y = _secbar(pdf, 'ANALYSIS SUMMARY', y)
    _card(pdf, 10, y, 190, 36)
    rows = [('Tumor Type', 'NO TUMOR DETECTED'), ('Classification Confidence', f'{confidence:.2%}'), ('Severity', 'None'), ('Growth Risk', 'None'), ('Risk Score', '0.000 / 1.0'), ('Recommended Action', 'Routine follow-up in 12 months')]
    for i, (k, v) in enumerate(rows):
        col = 0 if i < 3 else 1
        row = i if i < 3 else i - 3
        cx = 12 + col * 96
        _kv(pdf, cx, y + 2 + row * 11, k, v, kw=46, vw=46, bold_val=True, vc=GRN_C if k == 'Tumor Type' else TXT)
    y += 38
    y = _secbar(pdf, 'ORIGINAL MRI SCAN', y)
    tmp = '/tmp/rpt'
    os.makedirs(tmp, exist_ok=True)
    orig_p = f'{tmp}/orig_normal.jpg'
    cv2.imwrite(orig_p, orig)
    img_h = 85
    _card(pdf, 10, y, 190, img_h + 8)
    pdf.image(orig_p, x=10 + (190 - 100) / 2, y=y + 4, w=100, h=img_h)
    y += img_h + 12
    y = _secbar(pdf, 'CLINICAL DECISION SUPPORT', y)
    cal = {'steps': ['Routine follow-up MRI in 12 months if clinically indicated', 'No neurosurgical or oncological intervention required', 'Correlate with clinical symptoms and patient history', 'Maintain regular neurological check-ups'], 'urgency': 'LOW'}
    n = len(cal['steps'])
    mid = (n + 1) // 2
    ch = mid * 5.8 + 14
    _card(pdf, 10, y, 190, ch)
    pdf.set_xy(12, y + 3)
    pdf.set_font('Helvetica', 'B', 7.5)
    pdf.set_text_color(*MUT)
    pdf.cell(28, 5, 'Urgency Level:', ln=False)
    _badge(pdf, 42, y + 3, '  LOW  ', GRN_C)
    for i, step in enumerate(cal['steps']):
        col = 0 if i < mid else 1
        row = i if i < mid else i - mid
        pdf.set_xy(14 + col * 96, y + 12 + row * 5.8)
        pdf.set_font('Helvetica', '', 7.5)
        pdf.set_text_color(*TXT)
        pdf.cell(90, 4.8, safe(f'  >>  {step}'))
    y += ch + 4
    y = max(y, 248)
    pdf.set_fill_color(230, 255, 235)
    pdf.set_draw_color(60, 180, 80)
    pdf.rect(10, y, 190, 16, 'FD')
    pdf.set_font('Helvetica', 'B', 7.5)
    pdf.set_text_color(0, 110, 40)
    pdf.set_xy(13, y + 2.5)
    pdf.cell(10, 5, '(!)', ln=False)
    pdf.set_font('Helvetica', '', 7)
    pdf.set_xy(21, y + 2.5)
    pdf.multi_cell(176, 5, safe('IMPORTANT: This AI-generated result is intended for screening and decision-support purposes only. All findings must be reviewed by a qualified radiologist before clinical use. If the patient has symptoms, further investigation by a medical professional is strongly advised.'))
    pdf.set_text_color(*TXT)
    _footer(pdf, 1, 2)
    _llm_page(pdf, llm_report, 'AI Generated Imaging Summary', f'File: {os.path.basename(filename)}', 'Note: AI-generated imaging findings - No tumor detected.', 2, 2, patient_name, patient_id)
    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    sn = os.path.splitext(os.path.basename(filename))[0]
    pt = f"_{patient_name.replace(' ', '_')}" if patient_name else ''
    out = os.path.join(report_dir, f'{sn}{pt}_{ts}_normal_report.pdf')
    pdf.output(out)
    return out
