from pathlib import Path
import sys
from functools import lru_cache
import re
import os
import yaml
import pandas as pd
from joblib import load

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # "http://localhost:3000"
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_path = Path(__file__).resolve().parents[2]
sys.path.append(str(llm_path))
import LLM_API_calls
from Main_modules import load_model_data
from Clustering_classes import Clustering_classifier


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str | None = None
    role: str | None = "You are a helpful assistant."
    messages: list[Message] | None = None


class RagRequest(BaseModel):
    scenario: str


def _is_numeric_series(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def encode_query_row(
    row: dict,
    training_df: pd.DataFrame,
    x_encoder,
    cat_defaults,
):
    encoded = []

    for col in training_df.columns:
        val = row[col]
        series = training_df[col]

        if pd.api.types.is_numeric_dtype(series):
            try:
                encoded.append(float(val))
            except Exception:
                encoded.append(float(series.mean()))
        else:
            if x_encoder and col in x_encoder:
                try:
                    encoded.append(float(x_encoder[col].transform([str(val)])[0]))
                except Exception:
                    encoded.append(float(cat_defaults.get(col, 0)))
            else:
                encoded.append(float(cat_defaults.get(col, 0)))

    return encoded


def _build_defaults(df: pd.DataFrame, cols: list[str]) -> dict[str, object]:
    defaults: dict[str, object] = {}
    for col in cols:
        series = df[col]
        if _is_numeric_series(series):
            defaults[col] = float(series.mean())
        else:
            mode = series.mode(dropna=True)
            defaults[col] = mode.iloc[0] if not mode.empty else ""
    return defaults


def _parse_scenario(text: str, allowed_keys: list[str]) -> dict[str, str]:
    pairs = re.findall(r"([A-Za-z_]+)\s*:\s*([^\n,]+)", text)
    parsed = {}

    for k, v in pairs:
        k = k.strip()
        v = v.strip().strip("'\"")
        if k in allowed_keys:
            parsed[k] = v

    return parsed


def _coerce_value(raw: str, series: pd.Series) -> object:
    if _is_numeric_series(series):
        try:
            return float(raw)
        except ValueError:
            return float(series.mean())
    return raw


@lru_cache(maxsize=1)
def _cluster_resources():
    cfg_path = Path(__file__).resolve().parents[2] / "config.yml"
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    use_case, ml_task_type = next(iter(cfg["Use_Case"].items()))
    model_name = cfg["ML_models"][0]
    num_outputs, folder_name = next(iter(cfg["Output_variable_dict"].items()))

    base_ml_folder = Path(cfg["ML_files_folder"])
    if not base_ml_folder.is_absolute():
        base_ml_folder = (llm_path / base_ml_folder).resolve()
    ml_folder = base_ml_folder / use_case / folder_name

    training_data, meta_data_filename, test_data, model_file = load_model_data(
        model_name, str(ml_folder) + os.sep
    )

    num_outputs = int(num_outputs)
    x_labels = [
        col
        for col in training_data.columns
        if col not in ["LOS_CLASS", "LOS_CLASS_predicted"]
    ]

    y_labels = ["LOS_CLASS"]
    predicted_labels = ["LOS_CLASS_predicted"]

    x_encoder = None
    y_encoder = None
    if ml_task_type == "classification":
        enc_path = ml_folder / "label_encoders.pkl"
        if enc_path.exists():
            x_encoder = load(str(enc_path))
        y_path = ml_folder / "MLP_Y_scaler.save"
        if y_path.exists():
            y_encoder = load(str(y_path))

    if x_encoder is not None and y_encoder is None:
        y_encoder = {}

    clusterer = Clustering_classifier(
        "Accuracy", x_labels, y_labels, predicted_labels, cfg["Immutables"][use_case]
    )
    clusterer.Create_Clusters(
        training_data,
        x_labels,
        metric="Accuracy",
        X_cat_encoder=x_encoder,
        Y_cat_encoder=y_encoder,
        Clusterer=cfg.get("Clustering_algorithm", "Gaussian"),
        expert=cfg.get("Expert_algorithm", "OLS"),
        max_clusters=8,
    )

    defaults = _build_defaults(training_data, x_labels)
    cat_maps: dict[str, dict[str, int]] = {}
    cat_defaults: dict[str, int] = {}
    for col in x_labels:
        if not _is_numeric_series(training_data[col]):
            values = training_data[col].astype(str)
            categories = pd.Index(values.unique())
            mapping = {str(v): int(i) for i, v in enumerate(categories)}
            cat_maps[col] = mapping
            mode = values.mode(dropna=True)
            cat_defaults[col] = mapping.get(
                str(mode.iloc[0]) if not mode.empty else "", 0
            )

    return (
        clusterer,
        training_data,
        x_labels,
        x_encoder,
        y_encoder,
        defaults,
        cat_maps,
        cat_defaults,
    )


def _build_cluster_context(scenario: str) -> str:
    (
        clusterer,
        training_data,
        x_labels,
        x_encoder,
        y_encoder,
        defaults,
        cat_maps,
        cat_defaults,
    ) = _cluster_resources()

    parsed = _parse_scenario(scenario, x_labels)

    row: dict[str, object] = {}
    for col in x_labels:
        if col in parsed:
            row[col] = parsed[col]
        else:
            row[col] = defaults[col]

    query_list = encode_query_row(
        row,
        training_data[x_labels],
        x_encoder,
        cat_defaults,
    )

    for v, c in zip(query_list, x_labels):
        if not isinstance(v, (int, float)):
            raise ValueError(
                f"Invalid clustering input: feature '{c}' has non-numeric value '{v}'"
            )

    assigned_clusters, membership = clusterer._find_Similar_Clusters(query_list)

    context_parts = ["Clustering RAG context:\n", membership]

    missing = [c for c in x_labels if c not in parsed]
    if len(missing) / len(x_labels) > 0.3:
        context_parts.insert(
            0,
            "Note: Some demographic or clinical information was missing; "
            "cluster similarity is approximate.\n\n",
        )

    for cluster_id in assigned_clusters:
        description = clusterer.Cluster_text_descriptions.get(cluster_id, "")
        similar_points = clusterer._find_Similar_Points(
            query_list,
            cluster_id,
            X_cat_encoder=x_encoder,
            Y_cat_encoder=y_encoder,
            Delta_string=False,
        )
        context_parts.append(description)
        context_parts.append("\n\n")
        context_parts.append(similar_points)
        context_parts.append("\n\n")

    return "".join(context_parts).strip()


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        if req.messages:
            messages = [m.model_dump() for m in req.messages]
            if req.role and not any(m["role"] == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": req.role})
            reply = LLM_API_calls.ChatGPT_messages(messages)
        else:
            reply = LLM_API_calls.ChatGPT(role=req.role, content=req.message or "")
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag_context")
def rag_context(req: RagRequest):
    try:
        context = _build_cluster_context(req.scenario)
        return {"context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
