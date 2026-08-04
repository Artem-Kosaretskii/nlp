from openai import OpenAI
from pydantic import BaseModel
from enum import Enum
from typing import List, Optional
import json


class SeverityEnum(str, Enum):
    moderate = "moderate"
    mild = "mild"


class Finding(BaseModel):
    region: str
    observation: str
    severity: SeverityEnum


class ChestXrayReport(BaseModel):
    study_id: str
    findings: List[Finding]
    conclusion: str
    recommendations: List[str]


API_KEY = "apikey"
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
model_id = 'nvidia/nemotron-3-ultra-550b-a55b:free'

protocol = """
Chest X-ray Protocol
Examination: Chest X-ray, AP and lateral projections
Examination Date: August 1, 2026
Examination ID: XR-2026-0801-001
Description:
In the right lung, in the lower lobe, an area of moderate infiltration measuring up to 4 cm is detected.
In the left lung, in the projection of the lower lobe, a small amount of pleural fluid is noted.
The heart and pulmonary roots are normal.
The trachea is located in the midline.
Conclusion:
The X-ray image is consistent with focal infiltrative changes in the lower lobe of the right lung; on the left, there are signs of minimal pleural effusion.
Recommendations:
Computerized tomography of the chest to clarify the nature of the infiltration.
Consult a pulmonologist.
Follow-up examination in 10-14 days.
"""

prompt_no_format_description = f"""Form a structured conclusion based on the following protocol.{protocol}"""

prompt_format_description = f"""Form a structured conclusion based on the following protocol.{protocol}
Each conclusion must include:

- study identifier (study_id);
- list of findings (findings), where each finding is described by the fields: region (e.g., “right lung”), observation (e.g., “infiltrate”), severity (category: mild / moderate / severe);
- doctor's conclusion, a line with a brief summary;
- optional recommendations (recommendation), a list of text elements..

The result must be in json format.
"""

for prompt in [prompt_no_format_description, prompt_format_description]:
    completion = client.chat.completions.create(
      extra_body={},
      model=model_id,
      messages=[
        {
          "role": "user",
          "content": prompt,
        }
      ]
    )
    print("Completion")
    print(completion.choices[0].message.content)

    completion_pydantic = client.chat.completions.parse(
      extra_body={},
      model=model_id,
      messages=[
        {
          "role": "user",
          "content": prompt,
        }
      ],
      response_format=ChestXrayReport
    )
    print("Completion Pydantic")
    print(ChestXrayReport.model_validate_json(completion_pydantic.choices[0].message.content))

"""
Completion
**RADIOLOGY REPORT: CHEST X-RAY CONCLUSION**
**Examination:** Chest X-ray, AP and Lateral Projections
**Date:** August 1, 2026
**Accession/Exam ID:** XR-2026-0801-001
---
### **I. FINDINGS**
**Lungs:**
*   **Right Lower Lobe:** Area of moderate infiltration identified, measuring up to **4 cm** in maximum dimension.
*   **Left Lower Lobe:** Small volume of pleural fluid noted in the projection of the lower lobe (minimal pleural effusion).
*   **Remaining Lung Fields:** Clear. No additional focal lesions, masses, or consolidations detected.
**Mediastinum & Hila:**
*   **Cardiac Silhouette:** Normal size and contour (cardiothoracic ratio within normal limits).
*   **Pulmonary Roots (Hila):** Normal position, structure, and caliber.
*   **Trachea:** Midline position. No deviation or narrowing.
**Pleura:**
*   **Right:** No pleural effusion or thickening.
*   **Left:** Minimal pleural effusion (lower lobe projection).
**Bony Thorax & Soft Tissues:** No acute osseous abnormalities or soft tissue masses noted (within limits of examination).
---
### **II. IMPRESSION / CONCLUSION**
1.  **Focal infiltrative process, lower lobe of the right lung** (measuring approx. 4 cm). Differential diagnosis includes infectious pneumonia (bacterial, viral, fungal), organizing pneumonia, neoplastic process, or inflammatory etiology.
2.  **Minimal left-sided pleural effusion** (left lower lobe projection). Likely reactive/parapneumonic given the contralateral infiltrate, though alternative etiologies (metastatic, transudative) cannot be excluded solely on radiographs.
---
### **III. RECOMMENDATIONS**
1.  **Chest Computed Tomography (CT) with IV contrast:**
    *   *Indication:* To characterize the morphology, margins, and internal architecture of the right lower lobe infiltrate; to evaluate for underlying endobronchial lesion or lymphadenopathy; and to quantify/characterize the left pleural effusion.
2.  **Pulmonology Consultation:**
    *   *Indication:* Clinical correlation (fever, cough, leukocytosis, risk factors), determination of need for diagnostic thoracentesis (if effusion enlarges), sputum cultures, or bronchoscopy.
3.  **Follow-up Chest X-ray in 10–14 days:**
    *   *Indication:* To assess resolution of the infiltrate and effusion following initiation of therapy (if prescribed), or to monitor for interval change if managed conservatively.
---
**Reporting Radiologist:** _________________________
**Date Signed:** August 1, 2026
**Time Signed:** _________________________

Completion Pydantic
study_id='XR-2026-0801-001' findings=[Finding(region='Right Lung, Lower Lobe', observation='Area of moderate infiltration measuring up to 4 cm.', severity=<SeverityEnum.moderate: 'moderate'>), Finding(region='Left Lung, Lower Lobe (Pleura)', observation='Small amount of pleural fluid (minimal pleural effusion).', severity=<SeverityEnum.mild: 'mild'>), Finding(region='Heart and Pulmonary Roots', observation='Normal size and contour.', severity=<SeverityEnum.mild: 'mild'>), Finding(region='Trachea', observation='Midline position.', severity=<SeverityEnum.mild: 'mild'>)] conclusion='X-ray findings are consistent with focal infiltrative changes in the lower lobe of the right lung (measuring up to 4 cm) and signs of minimal pleural effusion on the left.' recommendations=['Chest CT scan to characterize the nature of the right lower lobe infiltration.', 'Pulmonology consultation.', 'Follow-up chest X-ray in 10–14 days to assess resolution or progression.']


With format description:
Completion
{
  "study_id": "XR-2026-0801-001",
  "findings": [
    {
      "region": "right lung, lower lobe",
      "observation": "infiltrate",
      "severity": "moderate"
    },
    {
      "region": "left lung, lower lobe",
      "observation": "pleural effusion",
      "severity": "mild"
    }
  ],
  "conclusion": "Focal infiltrative changes in the lower lobe of the right lung; minimal pleural effusion on the left.",
  "recommendations": [
    "Computerized tomography of the chest to clarify the nature of the infiltration.",
    "Consult a pulmonologist.",
    "Follow-up examination in 10-14 days."
  ]
}

Completion Pydantic
study_id='XR-2026-0801-001' findings=[Finding(region='right lower lobe', observation='infiltration', severity=<SeverityEnum.moderate: 'moderate'>), Finding(region='left lower lobe', observation='pleural effusion', severity=<SeverityEnum.mild: 'mild'>)] conclusion='The X-ray image is consistent with focal infiltrative changes in the lower lobe of the right lung; on the left, there are signs of minimal pleural effusion.' recommendations=['Computerized tomography of the chest to clarify the nature of the infiltration.', 'Consult a pulmonologist.', 'Follow-up examination in 10-14 days.']

"""
