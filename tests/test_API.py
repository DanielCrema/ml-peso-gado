import pandas as pd
import requests
import json
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ===============================
# CONFIG
# ===============================
API_URL = "http://localhost:8000/predict"  # replace with your deployed endpoint if needed
API_TOKEN = os.getenv("PREDICTION_API_TOKEN")
TEST_CSV_PATH = "../data/gado_test.csv"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

# ===============================
# READ TEST DATA
# ===============================
test_df = pd.read_csv(TEST_CSV_PATH)

# ===============================
# HELPER TO LOG JSON LINES
# ===============================
def log_result(result_dict):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4, sort_keys=True)
        f.write("\n\n")  # separate each log entry with a newline

# ===============================
# ITERATE OVER TEST DATA
# ===============================
for idx, row in test_df.iterrows():
    payload = {
        "raca": row["raca"],
        "idade_meses": int(row["idade_meses"]),
        "altura_cm": float(row["altura_cm"]),
        "comprimento_corpo_cm": float(row["comprimento_corpo_cm"]),
        "circunferencia_peito_cm": float(row["circunferencia_peito_cm"]),
        "cor_pelagem": row["cor_pelagem"],
        "sexo": row["sexo"]
    }

    try:
        headers = {"Authorization": API_TOKEN}
        response = requests.post(API_URL, json=payload, headers=headers, timeout=100)

        if response.status_code != 200:
            log_result({
                "row_index": idx,
                "payload": payload,
                "error": f"HTTP {response.status_code}",
                "response_text": response.text
            })
            print(f"[ERROR] Row {idx}: HTTP {response.status_code}")
            continue

        data = response.json()

        # Check for internal fallback error in the LLM analysis
        if data.get("analise_IA", {}).get("insights", {}).get("comparacao_raca") == "Erro na análise":
            log_result({
                "row_index": idx,
                "payload": payload,
                "error": "Fallback triggered in LLM analysis",
                "response_data": data
            })
            print(f"[WARNING] Row {idx}: Fallback triggered in LLM analysis")
        else:
            log_result({
                "row_index": idx,
                "payload": payload,
                "response_data": data
            })
            print(f"[OK] Row {idx}: peso_estimado={data['peso_estimado_kg']} kg")

    except requests.exceptions.RequestException as e:
        log_result({
            "row_index": idx,
            "payload": payload,
            "error": f"RequestException: {str(e)}"
        })
        print(f"[ERROR] Row {idx}: RequestException -> {str(e)}")

    except Exception as e:
        log_result({
            "row_index": idx,
            "payload": payload,
            "error": f"InternalException: {str(e)}"
        })
        print(f"[ERROR] Row {idx}: InternalException -> {str(e)}")

print(f"Test finished. Log saved to {LOG_FILE}")