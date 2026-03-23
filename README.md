# 🐄 CattleWeight — Estimativa Inteligente de Peso de Gado

O **CattleWeight** é um projeto de *Machine Learning* e *AI Automation* que estima o **peso de gado com base em características físicas**, expondo essa capacidade através de uma **API REST pronta para uso**.

A solução simula um cenário real do setor agropecuário, onde a pesagem manual pode gerar **imprecisão, inconsistência e baixa eficiência operacional**.

---

## 🚀 O que este projeto entrega

- 📊 Modelo de Machine Learning para regressão (peso em kg)
- ⚡ API REST para predição em tempo real
- 🤖 Integração com LLM para análise inteligente do animal
- 🔄 Base para automação com n8n
- 🧠 Pipeline completo de Data Science documentado

---

## 🔌 Como usar a API

### Endpoint

```
POST /predict
```

---

### 📥 Exemplo de requisição

```json
{
  "raca": "Angus",
  "idade_meses": 24,
  "altura_cm": 140,
  "comprimento_corpo_cm": 160,
  "circunferencia_peito_cm": 190,
  "cor_pelagem": "preta",
  "sexo": "macho"
}
```

---

### 📤 Exemplo de resposta

```json
{
  "peso_estimado_kg": 520.4,
  "versao_modelo": "v1",
  "analise": {
    "classificacao_peso": "adequado",
    "nivel_risco": "baixo",
    "recomendacoes": [
      "Manter alimentação equilibrada",
      "Monitorar crescimento mensalmente",
      "Garantir acesso a água limpa"
    ],
    "insights": {
      "comparacao_raca": "O peso está dentro da média esperada para a raça Angus com essa idade",
      "observacoes": "O animal apresenta bom desenvolvimento, sem sinais de sobrepeso ou baixo peso"
    }
  }
}

---

### 🔐 Autenticação

A API utiliza autenticação via header:

```
Authorization: <PREDICTION_API_TOKEN>
```

---

### 🧪 Exemplo em Python

```python
import requests

url = "http://localhost:8000/predict"

headers = {
    "Authorization": "seu_token_aqui"
}

data = {
    "raca": "Angus",
    "idade_meses": 24,
    "altura_cm": 140,
    "comprimento_corpo_cm": 160,
    "circunferencia_peito_cm": 190,
    "cor_pelagem": "preta",
    "sexo": "macho"
}

response = requests.post(url, json=data, headers=headers)

print(response.json())
```

---

## 🎯 Objetivo do Projeto

Criar um modelo de regressão que, a partir de características físicas do gado, estima o **peso do animal em quilogramas (kg)** de forma automatizada e consistente.

---

## 🧠 Pipeline de Data Science

O [**notebook do projeto**](main.ipynb) cobre as principais etapas:

### 1. ETL e limpeza dos dados
- Carregamento dos datasets
- Padronização e tipagem

### 2. Análise exploratória (EDA)
- Distribuições das variáveis
- Correlações
- Relações com a variável alvo

### 3. Engenharia de variáveis
- Encoding de variáveis categóricas
- Pipeline automatizado com Oracle AutoMLx

### 4. Modelagem
- Treinamento com AutoMLx
- Comparação entre modelos
- Seleção automática

### 5. Avaliação
- MAE, RMSE e R²

### 6. Exportação
- Modelo salvo via `pickle`
- Pronto para uso em produção

---

## 📊 Resultados

| Modelo                | MAE ↓ (kg) | RMSE ↓ (kg) | R² ↑    |
|----------------------|-----------|------------|--------|
| Regressão Linear     | 18.62     | 22.14      | 0.6031 |
| LightGBM (baseline)  | 68.89     | 73.09      | -3.3236 |
| AutoML (tuned)       | 18.62     | 22.14      | 0.6031 |

📌 O problema apresentou comportamento majoritariamente linear, sendo bem resolvido por modelos simples.

---

## 🏗️ Arquitetura

```
[Client] → [FastAPI] → [Model + Encoder] → [Prediction]
                               ↓
                            [LLM]
                               ↓
                            [n8n]
```

---

## 🤖 Integração com LLM

Após a predição, o sistema pode gerar uma análise estruturada do animal, por exemplo:

- Classificação de porte
- Avaliação de desenvolvimento
- Sugestões de manejo

---

## 🔄 Automação com n8n

O projeto pode ser integrado com n8n para:

- Receber dados via webhook
- Chamar a API automaticamente
- Aplicar regras de decisão
- Enviar notificações (WhatsApp, email, etc.)

---

## 📁 Estrutura do Projeto

```bash
.
├── API/          # Configuração da API
│
├── data/         # Dados de entrada
│
├── export/       # Módulo com funções para exportar modelos, artefatos e documentações
│
├── helpers/      # Funções auxiliares de parseamento e plotagem
│
├── LLM/          # Agente(s) de IA para análise de gado
│
├── models/       # Repositório de modelos
│
├── tests/       # Repositório de testes automatizados
│
├── utils/        # Utilitários de ML
│
├── main.ipynb    # Notebook principal
│
└── loader.py     # Classe para carregar e armazenar informações dos datasets de entrada
```

---

## ⚙️ Como rodar o projeto

### Versão Python
```bash
python=3.12.13
```

### Instalar dependências
```bash
pip install -r requirements.txt
```

### Rodar a API
```bash
uvicorn API.endpoint:app --reload
```


---

## 💡 Possíveis melhorias

- Deploy em cloud (AWS, GCP)
- Versionamento automatizado de modelos
- Monitoramento de drift
- Pipeline CI/CD

---

## 📌 Conclusão

O projeto demonstra a construção de uma solução completa de Machine Learning aplicada, indo desde a exploração de dados até a disponibilização via API e integração com sistemas automatizados.

Ele evidencia não apenas capacidade técnica, mas também **visão de produto e aplicação real de IA em negócios**.