FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HF_REPO="AdiSingh003/distilbert-goodreadnews-finetuned"

CMD ["sh", "-c", \
     "echo '=== Evaluating model from HuggingFace Hub: '${HF_REPO} && \
      python src/eval.py \
        --model_path ${HF_REPO} \
        --results_file hub_eval_results.json"]
