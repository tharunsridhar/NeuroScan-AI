from __future__ import annotations

import base64

from groq import Groq

from utils.io_utils import clean_text

def groq_tumor_report(image_path, label, confidence, size_info, shape_info, mass_info, risk_info, severity, rano, groq_api_key, patient_name='') -> str:
    from groq import Groq
    client = Groq(api_key=groq_api_key)
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    bbox_t = f"{size_info['bbox']['width_cm']} x {size_info['bbox']['height_cm']} cm" if size_info.get('bbox') else 'N/A'
    rano_t = '' if not rano else f"\n- RANO Size: {rano['size_cat']}\n- Enhancement: {rano['enhancement']}\n- Necrosis: {rano['necrosis']}"
    shape_t = '' if not shape_info else f"\n- Irregularity: {shape_info['irregularity']}\n- Border: {shape_info['border_def']}"
    pt_line = f'\nPatient Name: {patient_name}' if patient_name else ''
    prompt = f"You are a senior neuroradiology AI assistant generating a formal imaging decision-support report.\n{pt_line}\nAI ANALYSIS FINDINGS:\n- Tumor Type: {label.upper()} | Confidence: {confidence:.2%} | Severity: {severity}\n- Tumor Area: {size_info['area_cm2']} cm2 | Diameter: {size_info['diameter_cm']} cm | Est. Volume: {size_info['volume_cm3']} cm3\n- Brain Coverage: {size_info['tumor_percent']}% | Bounding Box: {bbox_t}\n- Laterality: {mass_info['laterality']} | Midline Shift: {mass_info['shift_mm']} mm\n- Risk Score: {risk_info['score']}/1.0 | Growth Risk: {risk_info['risk']}{shape_t}{rano_t}\n\nGenerate a structured imaging decision-support report with EXACTLY these 6 numbered sections.\nWrite each section as 2-4 complete professional sentences.\n\n1. CLINICAL INDICATION\n2. IMAGING TECHNIQUE\n3. FINDINGS\n4. IMPRESSION\n5. RISK STRATIFICATION\n6. RECOMMENDATIONS\n\nSTRICT OUTPUT RULES:\n- Use plain ASCII text only. No markdown, no bold, no bullets, no special characters.\n- Frame the output as screening and decision support, not a definitive diagnosis.\n- Never definitively confirm tumor grade.\n- Always state that histopathological confirmation is required.\n- Total report must be under 350 words.\n- Start each section with its number and title on its own line."
    resp = client.chat.completions.create(model='meta-llama/llama-4-scout-17b-16e-instruct', messages=[{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img_b64}'}}, {'type': 'text', 'text': prompt}]}], max_tokens=1200)
    return clean_text(resp.choices[0].message.content)


def groq_normal_report(image_path, confidence, groq_api_key, patient_name='') -> str:
    from groq import Groq
    client = Groq(api_key=groq_api_key)
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    pt_line = f'\nPatient Name: {patient_name}' if patient_name else ''
    prompt = f'You are a senior neuroradiology AI assistant generating a formal imaging decision-support report.\n{pt_line}\nAI CLASSIFICATION RESULT:\n- Result: NO TUMOR DETECTED\n- Classifier Confidence: {confidence:.2%}\n- Severity: NONE\n\nGenerate a structured imaging decision-support report with EXACTLY these 6 numbered sections.\nEach section must reflect that NO tumor was found. Write 2-3 complete professional sentences per section.\n\n1. CLINICAL INDICATION\n2. IMAGING TECHNIQUE\n3. FINDINGS\n4. IMPRESSION\n5. RISK STRATIFICATION\n6. RECOMMENDATIONS\n\nSTRICT OUTPUT RULES:\n- Use plain ASCII text only. No markdown, no bold, no bullets, no special characters.\n- Frame the output as screening and decision support, not a definitive diagnosis.\n- Total report must be under 300 words.\n- Start each section with its number and title on its own line.'
    resp = client.chat.completions.create(model='meta-llama/llama-4-scout-17b-16e-instruct', messages=[{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img_b64}'}}, {'type': 'text', 'text': prompt}]}], max_tokens=900)
    return clean_text(resp.choices[0].message.content)
