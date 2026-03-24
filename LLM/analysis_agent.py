import json
from google import genai
from API.api_interface import PredictRequest

# =========================
# FUNÇÃO PARA GARANTIR JSON
# =========================
def extract_json(text: str):
    try:
        return json.loads(text)
    except:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])

# =========================
# CHAMADA DO GEMINI
# =========================
def generate_llm_analysis(data: PredictRequest, peso: float, API_KEY: str, model_name: str):
    prompt = f"""
Você é um especialista em pecuária de corte.

Analise os dados do animal abaixo e gere uma avaliação objetiva.

IMPORTANTE:
- Responda APENAS com JSON válido
- NÃO inclua texto fora do JSON

Formato esperado:
{{
  "classificacao_peso": "baixo|adequado|alto",
  "nivel_risco": "baixo|medio|alto",
  "recomendacoes": ["string"],
  "insights": {{
    "comparacao_raca": "string",
    "observacoes": "string"
  }}
}}

Dados do animal:
- Raça: {data.raca}
- Idade: {data.idade_meses} meses
- Peso estimado: {peso} kg
- Altura: {data.altura_cm} cm
- Comprimento: {data.comprimento_corpo_cm} cm
- Circunferência do peito: {data.circunferencia_peito_cm} cm
- Sexo: {data.sexo}
"""

    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt],
        config=genai.types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=300
        )
    )

    return extract_json(response.text)