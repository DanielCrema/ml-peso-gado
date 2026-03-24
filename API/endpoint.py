import os
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from API.api_interface import PredictRequest, PredictResponse, Analise, Insights
from API.predict import predict_weight
from LLM.analysis_agent import generate_llm_analysis

load_dotenv()
app = FastAPI()

MODEL_ARTIFACT_NAME = os.getenv("MODEL_ARTIFACT_NAME")
PREDICTION_API_TOKEN = os.getenv("PREDICTION_API_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

if not PREDICTION_API_TOKEN:
    raise RuntimeError("PREDICTION_PREDICTION_API_TOKEN não configurado!")

@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest, authorization: str = Header(None)):
    """
    Endpoint para estimativa de peso de gado.

    Retorna:
    - peso_estimado_kg: valor contínuo previsto pelo modelo
    - versao_modelo: identificação da versão do modelo
    """

    if authorization != PREDICTION_API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Chave de API do Gemini não configurada")
    
    if not MODEL_ARTIFACT_NAME:
        raise HTTPException(status_code=500, detail="Modelo preditivo não configurado")

    # Executa inferência
    result = predict_weight(MODEL_ARTIFACT_NAME, dict(data))

    peso_estimado = round(float(np.float64(result["peso_estimado_kg"])), 2)
    versao_modelo = result["versao_modelo"]

    try:
        analise_dict = generate_llm_analysis(data, peso_estimado, GEMINI_API_KEY, GEMINI_MODEL_NAME)
        # converte dict em Pydantic
        insights_obj = Insights(**analise_dict["insights"])
        analise_obj = Analise(
            classificacao_peso=analise_dict["classificacao_peso"],
            nivel_risco=analise_dict["nivel_risco"],
            recomendacoes=analise_dict["recomendacoes"],
            insights=insights_obj
        )
    except Exception as e:
        # fallback seguro
        insights_obj = Insights(
            comparacao_raca="Erro na análise",
            observacoes=str(e)
        )
        analise_obj = Analise(
            classificacao_peso="baixo",
            nivel_risco="baixo",
            recomendacoes=[],
            insights=insights_obj
        )

    return PredictResponse(
        peso_estimado_kg=peso_estimado,
        versao_modelo=versao_modelo,
        analise_IA=analise_obj
    )