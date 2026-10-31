# LLM pipeline — ELMoE experiments

This directory contains a research pipeline for using large language models (LLMs) to interpret pretrained machine-learning models. It combines cluster descriptions and retrieved examples with task-specific prompts, then compares the LLM's answers with local explanation baselines.

Start with `config.yml` and `main_with_training.py`. The configuration filename in this directory is **`config.yml`**, not `config.yaml`.

The main tasks are:

- **Accuracy:** ask an LLM to reproduce a black-box model's prediction and measure agreement with that model.
- **Counterfactuals:** ask an LLM to propose changed inputs that produce a desired output, while preserving immutable features. Evaluate the proposed inputs using the original ML model.
- **Feature importance:** supporting code and prompt templates exist, but this task is not fully connected to the main runner's parsing and evaluation flow.

Configured use cases include Networking regression, air-quality (`AQ`) regression, and Healthcare classification. The checked-in configuration selects Healthcare counterfactuals, a pretrained Transformer, ChatGPT, and one output variable.

## How the code is distributed

| File or directory | Responsibility |
| --- | --- |
| `config.yml` | Experiment settings: use case, model names, task, artifact paths, sampling, clustering, retrieval, counterfactual goals, and saving. |
| `main_with_training.py` | Entry point and experiment orchestration. Loads and validates configuration, loops over experiment combinations, collects answers, evaluates results, saves artifacts, and plots regression results. |
| `Main_modules.py` | Shared pipeline operations: artifact loading, test selection, prompt assembly, LLM dispatch, baseline generation, and evaluation dispatch. |
| `Clustering_classes.py` | `Clustering_classifier` coordinates clustering, expert characteristics, query-to-cluster assignment, and retrieval. Includes Gaussian-mixture and fuzzy C-means implementations. |
| `Expert_classes.py` | OLS and SVR local experts and cluster-level feature-effect descriptions. |
| `Datapoint_search_algorithms.py` | Similar-example retrieval through KNN, BM25, dense embeddings, hybrid reciprocal-rank fusion, MMR, or cross-encoder reranking. |
| `Prompt_generator_module.py` | Builds the LLM role, dataset context, question, tasks, justification guidance, output format, constraints, and retrieved evidence. Also obtains ML predictions when constructing counterfactual questions. |
| `Dictionaries/` | Active prompt dictionaries: roles, dataset/feature descriptions, questions, tasks, output formats, constraints, and justification guidance. This is the directory imported by the current prompt generator. |
| `LLM_API_calls.py` | Hosted LLM calls and model-name routing, including OpenAI, Together, and a Hugging Face inference wrapper. Loads environment variables with `python-dotenv`. |
| `Trained_ML_model_functions.py` | Loads saved models/scalers and provides regression and classification prediction helpers. Registers custom Transformer layers during model loading. |
| `Cat_variable_functions.py` | Categorical encoding/decoding, numerical/categorical input preparation, and classification output conversion. |
| `FTransformer_model*.py` | Feature Transformer architectures and custom layers, including regression and categorical-input classification variants. |
| `Custom_transformer_layers_and_helper_functions.py` | Additional Transformer layer and input-shape helpers. |
| `XAI_benchmarks.py` | Locally implemented LIME-style prediction and DiCE-style counterfactual baselines, plus SHAP attribution helpers. The counterfactual baseline searches a configured feature-delta grid. |
| `Evaluation.py` | Parses and filters answers, runs the black-box model, and computes prediction-agreement and counterfactual metrics. |
| `Plotting_functions.py` | Matplotlib and Plotly comparison plots. |

### Supporting and alternative workflows

| File or directory | Purpose |
| --- | --- |
| `Final_ML_models.py`, `Final_ML_models Networking.py`, `Final_ML_models AQ.py`, `Final_ML_models Healthcare.py` | Separate scripts for training the underlying ML models and producing experiment artifacts. They contain machine-specific paths and refer to helper modules outside this directory. |
| `RAG_classes.py` | Alternative/older RAG implementations, including KNN, vector-database, hybrid, and autoencoder approaches. These classes are not instantiated by `main_with_training.py`. |
| `main.py`, `Module_info.txt` | Earlier pipeline example and accompanying notes. `main.py` imports helper names no longer present in `Main_modules.py`; use `main_with_training.py` for the current workflow. |
| `Loading_trained_ML_model.py` | Separate model-loading implementation; the active main workflow uses `Trained_ML_model_functions.py`. |
| `Loading_saved_results.py`, `redoing_sparsity_scores.py`, `exploring results/` | Saved-result inspection and metric reanalysis. Review their configured paths before use. |
| `Write_toDoc.py` | Helper for exporting results to Word. It is not called by the main runner. |
| `BigQuery sql scripts.py`, `diagnosis_csv_creator.py`, `diagnoses_combined.csv` | Healthcare data-extraction/preparation notes, a diagnosis CSV utility, and a data file. |
| `webapp/backend/app.py` | Separate FastAPI interface exposing `/api/chat` and `/api/rag_context`. It reuses pipeline components and includes Healthcare-specific assumptions. |
| `webapp/frontend/` | HTML/CSS browser interface for the backend. |
| Root-level prompt dictionary files, `Clustering_classes - Copy.py` | Parallel copies of some modules; the main runner uses `Clustering_classes.py` and the prompt generator imports `Dictionaries/`. |

## Main execution flow

```text
config.yml + downloaded model/data artifacts
    -> load_config() and validate_config()
    -> load training/test data, saved ML model paths, and encoders
    -> select test points
    -> build clusters and local expert characteristics
    -> retrieve cluster descriptions and similar examples for each query
    -> assemble prompt and call configured LLMs
    -> generate baseline answers
    -> evaluate both against the pretrained ML model / counterfactual target
    -> save reports, test data, prompts, configuration, and regression plots
```

1. `load_config()` appends the selected use case to `ML_files_folder` and resolves the output directory as `<saving_folder>/<Experiment>/<Use_case>`.
2. `run()` iterates through use cases, output counts, ML models, and metrics. It reads model-specific training and test CSVs and determines the saved model filename. Some non-Healthcare training logs are cleaned before clustering.
3. The final `num_outputs` columns of the test CSV define the output labels; preceding columns define the input labels. Predicted-output column names are constructed as `<output_label>_predicted`.
4. Test selection supports random sampling or distance-based sampling. For classification counterfactuals, a separate selector replaces the initial selection with points whose model predictions are outside the desired class values.
5. Accuracy clustering uses input features. Counterfactual clustering uses immutable features and predicted-output columns. The runner requests at most two clusters for Accuracy and eight for Counterfactuals.
6. Each query receives cluster membership information, cluster descriptions, and similar examples. OLS/SVR expert characteristics are computed, but the current retrieval method sets the expert-gradient text to an empty string, so those gradients are omitted from the prompt.
7. Prompts are sent to the configured LLM providers. Replies are stripped of Markdown code fences and parsed as Python dictionary literals using `ast.literal_eval`.
8. Baselines are generated locally. Accuracy uses a LIME-style surrogate; Counterfactuals uses a DiCE-style grid search. Both are evaluated through `Evaluation.py`.

The filename `main_with_training.py` does **not** mean the script trains the underlying black-box models or fine-tunes an LLM. It loads pretrained model artifacts and computes clustering/local-expert information for the interpretation experiment. Black-box model training is in the separate `Final_ML_models*.py` scripts.

## Source of the model and data files

**The Google Drive link recorded in [`../datasets.txt`](../datasets.txt) is the source of the files used by `main_with_training.py`:**

[Download the ELMoE framework files from Google Drive](https://drive.google.com/drive/folders/1U6m7ApOIRSinT1iXUzpGT02xDsHJt13M?usp=sharing)

Download the required artifacts and set `ML_files_folder` to their local base directory. The script loads local files; it does not download them from Google Drive automatically.

The loader expects this layout, where `<Model>` is the exact configured name, such as `Transformer`, `MLP`, `Ridge`, `DT`, or `RF`:

```text
data/                            # ML_files_folder
  Healthcare/                    # key from Use_Case
    One variable/                # value from Output_variable_dict
      <Model>_train_data.csv
      <Model>_test_data.csv
      <Model>_model.keras         # MLP / CNN_LSTM / CNN / Transformer
      <Model>_model.joblib        # other model names; alternative to .keras
      <Model>_X_scaler.save       # when required by the prediction helper
      <Model>_Y_scaler.save       # when required by the prediction helper
      label_encoders.pkl         # required by the classification runner
      MLP_Y_scaler.save           # classification target encoders, loaded by this exact name
      <Model>_metadata.txt        # loader constructs this path; runner does not read it
```

Create the analogous use-case/output folders for Networking or AQ as needed. Folder names must match the YAML values, including spaces.

Training CSVs need the input columns, ground-truth output columns, and corresponding `<output_label>_predicted` columns used by clustering and expert analysis. Keep the test CSV's output columns last and preserve feature order. Classification artifacts must use compatible input and target encoders; `label_encoders.pkl` should contain at least an empty dictionary if there are no categorical inputs.

## Environment and execution

Run commands from this `LLM pipeline` directory so that relative configuration and data paths resolve correctly. The code uses Python 3.10+ syntax; exact compatibility also depends on the pinned packages and saved model artifacts.

Example setup in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install together shap
```

`together` and `shap` are imported by the active pipeline but are absent from the checked-in `requirements.txt`. Optional BM25 retrieval additionally needs `rank_bm25`. The separate web application needs FastAPI and an ASGI server such as Uvicorn; these are also absent from that requirements file.

Add the credentials for the selected provider to a local `.env` file or the process environment:

```dotenv
OPENAI_API_KEY=your_openai_key
# TOGETHER_API_KEY=your_together_key
# HUGGINGFACE_TOKEN=your_hugging_face_token
```

`ChatGPT` currently maps to `gpt-4o` in `LLM_API_calls.py`. Together model aliases and their identifiers are defined in `Together_models()`. Provider model identifiers are source-code settings rather than YAML parameters. API calls transmit the assembled prompts, including retrieved examples, to the selected provider.

After installing dependencies, placing the artifacts, and reviewing the configuration and limitations below, run:

```powershell
.\.venv\Scripts\python.exe main_with_training.py
```

The entry point always reads `config.yml`; it has no command-line configuration-path argument. Model training scripts are not required when using the downloaded pretrained artifacts.

## Configuration reference

| Setting | Meaning and current behavior |
| --- | --- |
| `Use_Case` | Maps a use case to `regression` or `classification`. Use one entry per run because path resolution uses the first entry. |
| `ML_models` | Names used to find the saved models and CSVs. Current selection: `Transformer`. |
| `LLM_models` | Provider aliases handled by `LLM_API_calls.py`. Current selection: `ChatGPT`. Use one LLM per run until the multi-LLM bookkeeping issue below is fixed. |
| `Metrics` | Current selection: `Counterfactuals`. `Accuracy` is also wired into parsing/evaluation; standalone `Feature_importance` is incomplete. |
| `Output_variable_dict` | Maps output count to artifact subfolder, for example `1: "One variable"`. |
| `ML_files_folder` | Base artifact directory. Current value: `data/`. |
| `output_upgrade_CF` | Per-use-case output deltas for regression, or desired class labels for classification, in output-column order. Current Healthcare target: `Short`. |
| `Immutables` | Per-use-case input features that counterfactual proposals should preserve. |
| `Dice_grid` | Allowed numeric feature deltas for the counterfactual baseline. Every mutable input needs an entry. |
| `Test_subset` | Requested number of test examples; currently 50. |
| `Train_subset` | Currently 100. Passed to the distance sampler's size check, but does not limit the training rows supplied to clustering. |
| `Test_selection_strategy` | `bin_quantile_sampling` (current), `outlier_sampling`, `mixture_sampling`, or `random`. See the random-selection issue below. Classification counterfactual selection overrides this initial subset. |
| `Test_selection_standardize` | Standardizes test distances using training-set statistics. The runner also hardcodes an upper distance quantile cap of 0.80 for distance sampling. |
| `seed` | Used by the distance sampler; not a global reproducibility guarantee. Classification counterfactual selection uses a hardcoded seed of 42. |
| `Clustering_algorithm` | `Gaussian` (current) or `Fuzzy`. See the fuzzy-clustering compatibility issue below. |
| `Expert_algorithm` | `OLS` (current) or `SVR`; computes local expert characteristics. |
| `Similar_points_method` | `knn` (current), `bm25`, `dense`, `hybrid_rrf`, `mmr`, or `cross_encoder`. |
| `Similar_points_kwargs` | Retriever settings such as embedding model, candidate pool, MMR weight, and reranker model. |
| `LLM_outputs` | `One_output`: primary answer; `Two_output`: also feature importance; `More_output`: also local error. The main loop currently evaluates/stores the primary answers, not all parsed auxiliary fields. |
| `Save_Yes_or_No` | Quoted string `'Yes'` or `'No'` controlling result saving. |
| `saving_folder`, `Experiment` | Base output directory and experiment name. |
| `User_Train_enable` | Recorded in the result key; does not enable or disable expert fitting in this runner. |
| `RAG_Class_variables`, `RAG_db_path` | Legacy settings. The former is checked for non-emptiness, but neither selects or persists the current clustering-based retrieval system. |
| `RAG_debug` | Present in YAML, but the runner reads `Cluster_classifier_debug` instead. Add that exact key to enable its prompt debug output. |

## Evaluation and saved outputs

Accuracy evaluation compares LLM/baseline predictions with predictions from the original ML model. Regression reports include MAE, MSE, RMSE, and R-squared; classification reports include accuracy, precision, recall, F1, and confusion matrices. These measure agreement with the model, rather than directly measuring its accuracy against observed labels.

Counterfactual evaluation runs the original model on proposed inputs. Reports include validity, proximity, and sparsity. Sparsity counts changed mutable features; proximity is implemented as a clipped similarity score in `[0, 1]`, with higher values indicating closer inputs. For classification, the target is the configured desired class. For regression, evaluation currently constructs the target from the test CSV's output plus the configured delta; counterfactual prompt construction uses the original model prediction as its starting output.

With the checked-in configuration, saved outputs go to:

```text
LLM_RAG_results/
  Ablation_results/
    Healthcare/
      ChatGPT_results.pkl
      RAG_test_data.csv
      prompts.pkl
      config_used.json
```

- `results.pkl`, or `<first_LLM_name>_results.pkl` when the output path contains `Ablation_results`: experiment dictionaries containing reports, model/target outputs, LLM and baseline results, parsed answer tables, dropped-answer indices, test data, retrieval logs, and labels.
- `RAG_test_data.csv`: test data from the first saved result entry.
- `prompts.pkl`: one role-plus-content prompt per model/metric/cluster-class/output-count key. Repeated test points overwrite that key, so this is not a complete prompt history. Per-point question/retrieval/answer logs are stored in each result's `RAG_outputs`.
- `config_used.json`: configuration snapshot with resolved paths converted to strings.
- `plots/<model>/<N>_output_regressor/`: regression comparison plots in Matplotlib PNG and Plotly HTML subfolders. Classification plotting is currently a no-op.

Result keys contain `(use_case, task_type, metric, model_name, num_outputs, llm_name, training_enabled, User_Train_enable)`. The cluster object's `training_enabled` flag remains `False` in the current implementation even though expert characteristics are computed.

## Current implementation limitations

These notes describe the checked-in code, not fixes applied by this README:

- **External artifacts are required.** The configured `data/` folder is not included in this directory snapshot. Training scripts also depend on external data, machine-specific paths, and helpers such as `DT_model_files` and `Transformer_model_files`.
- **Baseline sample size is fixed at 1,000.** The runner passes `random_sample_length=1000`, and the baseline samples without replacement. Smaller training CSVs will fail unless this is adjusted.
- **Random selection leaves `dist_test_data` undefined.** The result-recording block still reads it. Use a distance-based strategy until the random branch initializes that value.
- **Multiple LLMs are not handled correctly in evaluation/storage.** The evaluation arguments are changed to baseline predictions without being reset for the next LLM, and the result-storage block is outside the LLM loop. Only the final LLM entry is stored.
- **Multiple use cases share paths resolved from the first case.** Configure one use case per run until path construction is moved into the corresponding loop.
- **Fuzzy clustering has an interface mismatch.** The coordinator passes `X_cat_encoder` to `Describe_clusters()`, but the fuzzy implementation does not accept that argument. Gaussian is the current default.
- **Standalone feature-importance runs are incomplete.** Answer splitting and `Evaluation_all_answers()` only implement the Accuracy and Counterfactuals paths. Auxiliary feature-importance/local-error outputs are parsed but not fully evaluated or persisted.
- **Counterfactual distance metadata can be stale.** Classification counterfactual selection replaces the test rows after distance sampling without updating `dist_test_data`. The overridden test rows also remain in use for later metrics in the same loop.
- **Regression counterfactual targets differ between prompting and evaluation.** The prompt requests a change relative to the model prediction, while evaluation adds the delta to the observed test output. Account for this distinction when interpreting validity results.
- **Counterfactual filtering needs alignment review.** Evaluation constructs aligned test rows after dropping invalid answers, but passes the full original input dataframe into `cf_metrics()`. Dropped answers can therefore cause inconsistent comparisons or shape problems.
- **Configuration validation is partial.** The counterfactual-length assertion checks the number of entries in the use-case mapping rather than each selected case's target list. Ensure each target list covers the requested output count.
- **LLM parsing assumes a valid dictionary literal.** Malformed answers can fail before evaluation's row-filtering stage; the API/parsing path has no retry mechanism.
- **Disabling saving does not fully suppress filesystem output.** Validation still creates the result directory. Regression plotting passes `str(None)` as the filename when saving is disabled, which can produce unintended files.

This README was prepared by inspecting the source and configuration. No end-to-end experiment, dependency installation, artifact download, or hosted LLM call was performed as part of documenting the directory.
