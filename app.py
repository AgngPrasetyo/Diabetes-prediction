from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from pathlib import Path
import requests
import json
import os



app = Flask(__name__)


ROOT_DIR = Path(__file__).parent.resolve()
MODEL_DIR = ROOT_DIR / "best models"



GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


GROQ_MODEL = "llama-3.3-70b-versatile" 

ensemble = joblib.load(MODEL_DIR / "ngboost_ensemble_model.joblib")
ebm = joblib.load(MODEL_DIR / "ebm_explainer.joblib")

model1 = ensemble["model1"]
model2 = ensemble["model2"]
w1, w2 = ensemble["weights"]
threshold = ensemble["threshold"]


FEATURE_COLS = [
    "HighBP",
    "HighChol",
    "Smoker",
    "Stroke",
    "HeartDisease",
    "PhysActivity",
    "Alcohol",
    "Sex",
    "BMI",
    "GenHealth",
    "MentalHealth",
    "PhysicalHealth",
    "Age",
    "BMI_HighBP_Interaction",
    "Age_GenHealth_Interaction"
]


#  CLINICAL GUARDRAIL

def clinical_guardrail(input_data, risk_prob, risk_label):
    """
    Apply clinical rules to adjust risk assessment
    """
    clinically_protective = (
        input_data["Age"] <= 5 and
        input_data["BMI"] < 25 and
        input_data["HighBP"] == 0 and
        input_data["HighChol"] == 0 and
        input_data["PhysActivity"] == 1 and
        input_data["HeartDisease"] == 0 and
        input_data["Stroke"] == 0
    )

    if clinically_protective:
        return min(risk_prob, 0.20), "Low Risk (Clinically Adjusted)", True

    return risk_prob, risk_label, False



def build_ai_payload(
    risk_label,
    risk_probability,
    clinical_adjustment,
    ebm_contribution,
    top_k=5,  #  top 5 faktor untuk analisis lebih detail
    threshold=0.03 
):
    """
    Prepare structured data for AI explanation
    """
    risk_factors = []
    protective_factors = []

    for feature, value in ebm_contribution.items():
        if value >= threshold:
            risk_factors.append({
                "factor": feature,
                "contribution": round(value, 4)
            })
        elif value <= -threshold:
            protective_factors.append({
                "factor": feature,
                "contribution": round(value, 4)
            })

    risk_factors = sorted(
        risk_factors, key=lambda x: x["contribution"], reverse=True
    )[:top_k]

    protective_factors = sorted(
        protective_factors, key=lambda x: x["contribution"]
    )[:top_k]

    return {
        "final_risk_label": risk_label,
        "final_risk_probability": round(risk_probability, 4),
        "clinical_adjustment": clinical_adjustment,
        "dominant_risk_factors": risk_factors,
        "dominant_protective_factors": protective_factors
    }



# BUILD GROQ PROMPT

def build_groq_prompt(ai_payload):
    """
    Generate optimized prompt for Groq/LLaMA model
    """
    return f"""Anda adalah dokter spesialis penyakit dalam dengan keahlian khusus dalam diabetes melitus. Analisis hasil prediksi diabetes berikut dan berikan penjelasan medis yang komprehensif.

DATA PASIEN (dari Model Machine Learning):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Kategori Risiko: {ai_payload["final_risk_label"]}
Probabilitas Risiko: {ai_payload["final_risk_probability"]} ({ai_payload["final_risk_probability"]*100:.2f}%)
Penyesuaian Klinis: {"Ya - Risiko disesuaikan berdasarkan profil protektif" if ai_payload["clinical_adjustment"] else "Tidak"}

FAKTOR PENINGKAT RISIKO (berdasarkan kontribusi statistik):
{json.dumps(ai_payload["dominant_risk_factors"], indent=2, ensure_ascii=False)}

FAKTOR PROTEKTIF (berdasarkan kontribusi statistik):
{json.dumps(ai_payload["dominant_protective_factors"], indent=2, ensure_ascii=False)}

TUGAS ANDA:
Berikan analisis medis lengkap dalam format JSON berikut (HANYA JSON, tanpa teks tambahan):

{{
  "interpretasi_klinis": "Penjelasan singkat dan jelas tentang hasil prediksi, termasuk arti dari probabilitas risiko dan apakah ada penyesuaian klinis yang diterapkan (3-4 kalimat, bahasa medis tapi mudah dipahami)",
  
  "kesimpulan_klinis": "Kesimpulan profesional yang menyatakan status risiko pasien secara keseluruhan, dengan mempertimbangkan semua faktor yang ada (3-4 kalimat, tegas namun empatik)",
  
  "faktor_utama": [
    "Penjelasan DETAIL faktor risiko/protektif pertama: nama faktor, mengapa penting, bagaimana pengaruhnya terhadap diabetes, dan nilai kontribusinya",
    "Penjelasan DETAIL faktor risiko/protektif kedua dengan cara yang sama",
    "Penjelasan DETAIL faktor risiko/protektif ketiga dengan cara yang sama",
    "Penjelasan DETAIL faktor risiko/protektif keempat dengan cara yang sama (jika ada)",
    "Penjelasan DETAIL faktor risiko/protektif kelima dengan cara yang sama (jika ada)"
  ],
  
  "saran_tindak_lanjut": [
    "Saran medis spesifik pertama berdasarkan hasil analisis",
    "Saran medis spesifik kedua berdasarkan hasil analisis",
    "Saran medis spesifik ketiga berdasarkan hasil analisis",
    "Saran medis spesifik keempat berdasarkan hasil analisis",
    "Saran medis spesifik kelima (jika diperlukan)"
  ],
  
  "penjelasan_ai_detail": "Penjelasan komprehensif dan mendalam (5-7 kalimat) yang menjelaskan: (1) Bagaimana faktor-faktor risiko dan protektif saling berinteraksi, (2) Mengapa model memberikan probabilitas seperti ini, (3) Apa implikasi klinis dari kombinasi faktor-faktor tersebut, (4) Perspektif medis tentang kondisi pasien secara holistik. Gunakan bahasa profesional namun mudah dipahami."
}}

PEDOMAN PENTING:
✓ Gunakan bahasa Indonesia yang profesional, medis, namun mudah dipahami pasien awam
✓ Jelaskan istilah medis dengan bahasa sederhana
✓ JANGAN mengubah atau mempertanyakan nilai probabilitas yang diberikan model
✓ JANGAN memberikan diagnosis pasti - gunakan kata "indikasi", "kemungkinan", "risiko"
✓ Sesuaikan tingkat urgency saran dengan tingkat risiko (High Risk = lebih urgent)
✓ Referensikan data kontribusi statistik secara spesifik dalam penjelasan
✓ Fokus pada edukasi preventif dan pemberdayaan pasien
✓ Berikan HANYA JSON yang valid - tidak ada markdown, backticks, atau teks pembuka/penutup

CONTOH faktor_utama yang BAIK:
"BMI (Indeks Massa Tubuh) menunjukkan kontribusi positif sebesar 0.156 terhadap risiko diabetes. BMI yang tinggi berkaitan langsung dengan resistensi insulin karena jaringan lemak berlebih mengganggu kerja insulin dalam mengatur gula darah. Ini adalah faktor risiko yang dapat dimodifikasi melalui program penurunan berat badan."

Response Anda:"""


# =====================================================
# 🤖 CALL GROQ API
# =====================================================
def call_groq_api(prompt):
    """
    Call Groq API with error handling and retry logic
    """
    # Check API key
    if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        raise Exception("Groq API key belum dikonfigurasi. Dapatkan di https://console.groq.com/keys")
    
    try:
        print(f"🤖 Calling Groq API with model: {GROQ_MODEL}")
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "Anda adalah dokter spesialis yang memberikan penjelasan medis dalam format JSON yang valid."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,  # Balanced creativity and consistency
                "max_tokens": 3000,  # Enough for detailed medical explanation
                "top_p": 0.9
            },
            timeout=30
        )
        
        print(f"📡 Groq API Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            print(f"✓ Received response from Groq ({len(text)} chars)")
            
            # Clean and parse JSON
            text = text.strip()
            
            # Remove markdown code blocks if present
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            # Parse JSON
            parsed_data = json.loads(text)
            print("✓ JSON parsed successfully")
            
            return parsed_data
            
        elif response.status_code == 401:
            raise Exception("Groq API key tidak valid. Periksa kembali API key Anda.")
        elif response.status_code == 429:
            raise Exception("Rate limit terlampaui. Tunggu beberapa saat atau upgrade ke akun berbayar.")
        else:
            error_detail = response.text
            raise Exception(f"Groq API error {response.status_code}: {error_detail}")
            
    except requests.exceptions.Timeout:
        raise Exception("Request timeout. Groq API tidak merespons dalam 30 detik.")
    
    except json.JSONDecodeError as e:
        print(f"✗ JSON parse error: {e}")
        print(f"Raw response: {text[:500] if 'text' in locals() else 'No response'}")
        raise Exception(f"Groq mengembalikan response yang bukan JSON valid: {str(e)}")
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")


# =====================================================
# ROUTE: HOME
# =====================================================
@app.route("/")
def index():
    return render_template("index.html")


# =====================================================
# ROUTE: PREDICT
# =====================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        X = pd.DataFrame([data])

        # Create interaction features
        X["BMI_HighBP_Interaction"] = X["BMI"] * X["HighBP"]
        X["Age_GenHealth_Interaction"] = X["Age"] * X["GenHealth"]
        X = X[FEATURE_COLS]
        
        # Get predictions
        p1 = model1.predict_proba(X)[:, 1]
        p2 = model2.predict_proba(X)[:, 1]

        # Calculate weighted ensemble prediction
        raw_risk_prob = float((w1 * p1[0] + w2 * p2[0]) / (w1 + w2))
        raw_risk_label = "High Risk" if raw_risk_prob >= threshold else "Low Risk"

        # Apply clinical guardrail
        risk_prob, risk_label, guardrail_applied = clinical_guardrail(
            data, raw_risk_prob, raw_risk_label
        )

        # Get EBM explanation
        explanation = ebm.explain_local(X)
        scores = explanation.data(0)["scores"]

        # Build contribution dictionary
        ebm_contribution = {
            FEATURE_COLS[i]: float(scores[i])
            for i in range(len(FEATURE_COLS))
        }

        # Sort by absolute contribution
        ebm_contribution = dict(
            sorted(ebm_contribution.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return jsonify({
            "risk_probability": risk_prob,
            "risk_label": risk_label,
            "raw_risk_probability": raw_risk_prob,
            "raw_risk_label": raw_risk_label,
            "threshold": threshold,
            "clinical_adjustment": guardrail_applied,
            "ebm_contribution": ebm_contribution
        })
    
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Prediction error: {str(e)}"
        }), 500


# =====================================================
# 🤖 ROUTE: AI EXPLANATION (GROQ)
# =====================================================
@app.route("/ai-explanation", methods=["POST"])
def ai_explanation():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate required fields
        required_fields = ["risk_label", "risk_probability", "clinical_adjustment", "ebm_contribution"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Build AI payload
        ai_payload = build_ai_payload(
            risk_label=data["risk_label"],
            risk_probability=data["risk_probability"],
            clinical_adjustment=data["clinical_adjustment"],
            ebm_contribution=data["ebm_contribution"]
        )

        # Generate prompt
        prompt = build_groq_prompt(ai_payload)

        # Call Groq API
        ai_response = call_groq_api(prompt)

        # Validate response structure
        required_keys = ["interpretasi_klinis", "kesimpulan_klinis", "faktor_utama", "saran_tindak_lanjut", "penjelasan_ai_detail"]
        for key in required_keys:
            if key not in ai_response:
                print(f"⚠ Missing key in response: {key}")
                ai_response[key] = f"Data {key} tidak tersedia"

        print("✓ AI explanation generated successfully")
        return jsonify(ai_response)
    
    except Exception as e:
        print(f"✗ AI explanation error: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "error": f"AI explanation error: {str(e)}",
            "details": "Periksa konfigurasi API key dan koneksi internet"
        }), 500


# =====================================================
# RUN APP
# =====================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" SISTEM PREDIKSI DIABETES - GROQ AI POWERED")
    print("=" * 60)
    print(f" LLM Engine: Groq ({GROQ_MODEL})")
    print(f" API Key Status: {'✓ Configured' if GROQ_API_KEY != 'YOUR_GROQ_API_KEY_HERE' else '✗ Not Configured'}")
    print(f" Server: http://localhost:5000")
    print("=" * 60)
    
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        print("\n  WARNING: Groq API key belum dikonfigurasi!")
        print(" Dapatkan API key gratis di: https://console.groq.com/keys")
        print(" Edit baris 23 di app.py dan masukkan API key Anda\n")
    
app.run(host="0.0.0.0", port=5000)