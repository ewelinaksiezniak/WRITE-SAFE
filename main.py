import re
from collections import defaultdict

import numpy as np
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
from collections import Counter, defaultdict

import torch
# from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset

from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


author_bigrams = defaultdict(Counter)
bert_cache = {}


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)



DATA_PATH = "proto.parquet"

nlp = spacy.load("en_core_web_sm")


def compute_features(text: str) -> dict:

    if not isinstance(text, str):
        text = str(text)

    doc = nlp(text)

    words = [t.text for t in doc if t.is_alpha]

    if len(words) == 0:
        return {
            "func_words_ratio": 0.0,
            "avg_word_length": 0.0,
            "avg_sentence_len": 0.0,
            "punct_ratio": 0.0,
            "noun_ratio": 0.0,
            "verb_ratio": 0.0,
            "adj_ratio": 0.0,
            "caps_ratio": 0.0,
            "exclam_ratio": 0.0,
            "qmark_ratio": 0.0,
        }


    func_count = sum(1 for w in words if w.lower() in STOP_WORDS)
    func_words_ratio = func_count / len(words)


    avg_word_length = float(np.mean([len(w) for w in words]))


    sentence_lengths = [
        len([t for t in sent if t.is_alpha]) for sent in doc.sents
    ]
    avg_sentence_len = (
        float(np.mean(sentence_lengths)) if sentence_lengths else 0.0
    )


    punct_pattern = r"[!?.,;:…]"
    punct_count = len(re.findall(punct_pattern, text))
    punct_ratio = punct_count / len(text) if len(text) > 0 else 0.0


    tokens_alpha = [t for t in doc if t.is_alpha]
    total_tokens = len(tokens_alpha)

    if total_tokens == 0:
        noun_ratio = verb_ratio = adj_ratio = 0.0
    else:
        nouns = sum(1 for t in tokens_alpha if t.pos_ == "NOUN")
        verbs = sum(1 for t in tokens_alpha if t.pos_ == "VERB")
        adjs = sum(1 for t in tokens_alpha if t.pos_ == "ADJ")

        noun_ratio = nouns / total_tokens
        verb_ratio = verbs / total_tokens
        adj_ratio = adjs / total_tokens


    caps_count = sum(1 for w in words if w.isupper() and len(w) > 1)
    caps_ratio = caps_count / len(words)


    exclam_count = text.count("!")
    qmark_count = text.count("?")

    exclam_ratio = exclam_count / len(text) if len(text) > 0 else 0.0
    qmark_ratio = qmark_count / len(text) if len(text) > 0 else 0.0

    return {
        "func_words_ratio": func_words_ratio,
        "avg_word_length": avg_word_length,
        "avg_sentence_len": avg_sentence_len,
        "punct_ratio": punct_ratio,
        "noun_ratio": noun_ratio,
        "verb_ratio": verb_ratio,
        "adj_ratio": adj_ratio,
        "caps_ratio": caps_ratio,
        "exclam_ratio": exclam_ratio,
        "qmark_ratio": qmark_ratio,
    }


def build_features_df(path: str) -> pd.DataFrame:
    df_raw = pd.read_parquet(path)

    rows = []
    for _, row in df_raw.iterrows():
        text = row["content"]
        author = row["author"]
        date_time = row.get("date_time", None)


        feats = compute_features(text)
        feats["author"] = author
        feats["date_time"] = date_time
        rows.append(feats)


        tokens = re.findall(r"\b\w+\b", str(text).lower())
        bigrams = zip(tokens, tokens[1:])

        author_bigrams[author].update(" ".join(bg) for bg in bigrams)

    df_features = pd.DataFrame(rows)



    df_features["date_time"] = pd.to_datetime(
        df_features["date_time"], errors="coerce"
    )
    df_features = df_features.sort_values(["author", "date_time"])


    df_features["post_index"] = df_features.groupby("author").cumcount()

    return df_features



df_features = build_features_df(DATA_PATH)

df_texts = pd.read_parquet(DATA_PATH)


METRIC_LABELS = {
    "func_words_ratio": "Function words ratio",
    "avg_word_length": "Average word length",
    "avg_sentence_len": "Average sentence length",
    "punct_ratio": "Punctuation ratio",
    "noun_ratio": "Nouns ratio",
    "verb_ratio": "Verbs ratio",
    "adj_ratio": "Adjectives ratio",
    "caps_ratio": "Caps lock ratio",
    "exclam_ratio": "Exclamations – ratio",
    "qmark_ratio": "Question marks – ratio",
}


GLOBAL_STATS = {}
for metric in METRIC_LABELS.keys():
    series = df_features[metric].dropna()
    GLOBAL_STATS[metric] = {
        "mean": float(series.mean()),
        "q1":   float(series.quantile(0.25)),
        "q3":   float(series.quantile(0.75)),
        "std":  float(series.std(ddof=0)) or 1.0,
    }




from torch.utils.data import Dataset as TorchDataset
import torch

def train_user_vs_background(selected_author: str):

    MODEL_PATH = "./authorship_distilbert"

    # 1) tokenizer + model z surrogate
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=2,  # binary: 0 = background, 1 = user
        id2label={0: "background", 1: "user"},
        label2id={"background": 0, "user": 1},
        ignore_mismatched_sizes=True,
    )


    if hasattr(model, "distilbert"):
        encoder = model.distilbert
    elif hasattr(model, "bert"):
        encoder = model.bert
    else:
        encoder = None

    if encoder is not None:
        for param in encoder.parameters():
            param.requires_grad = False

    print(f"\n[INFO] Trenuję user vs background dla autora: {selected_author}")


    user_df = df_texts[df_texts["author"] == selected_author].copy()
    bg_df = df_texts[df_texts["author"] != selected_author].copy()

    print("Liczba tweetów usera:", len(user_df))
    print("Liczba tweetów background:", len(bg_df))


    N_USER = min(200, len(user_df))
    N_BG = min(200, len(bg_df))

    if N_USER < 20 or N_BG < 20:
        print("[WARN] Za mało przykładów user/background do sensownego treningu.")

    user_df = user_df.sample(N_USER, random_state=SEED)
    bg_df = bg_df.sample(N_BG, random_state=SEED)

    user_df["label"] = 1
    bg_df["label"] = 0

    full_df = (
        pd.concat([user_df, bg_df])
        .sample(frac=1.0, random_state=SEED)
        .reset_index(drop=True)
    )

    # train / test split
    train_frac = 0.8
    train_size = int(len(full_df) * train_frac)
    train_df = full_df.iloc[:train_size]
    test_df = full_df.iloc[train_size:]

    print("Train size:", len(train_df), "Test size:", len(test_df))


    class TextDataset(TorchDataset):
        def __init__(self, df, tokenizer):
            self.texts = df["content"].tolist()
            self.labels = df["label"].tolist()
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            enc = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=128,
            )
            item = {
                "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(label, dtype=torch.long),
            }
            return item

    train_ds = TextDataset(train_df, tokenizer)
    test_ds = TextDataset(test_df, tokenizer)


    training_args = TrainingArguments(
        output_dir="./user_vs_background",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        evaluation_strategy="epoch",
        logging_steps=20,
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )

    trainer.train()


    pred_output = trainer.predict(test_ds)
    y_true = pred_output.label_ids
    y_pred = pred_output.predictions.argmax(axis=-1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=["background", "user"],
        digits=3,
    )

    return report, tokenizer, model


author_options = sorted(df_features["author"].unique())
metric_options = [
    {"label": label, "value": key} for key, label in METRIC_LABELS.items()
]

def integrated_gradients(
    text: str,
    tokenizer,
    model,
    target_label_id: int = 1,
    max_length: int = 128,
    n_steps: int = 20,
):

    model.eval()


    device = next(model.parameters()).device


    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)


    emb_layer = model.get_input_embeddings()


    embeddings = emb_layer(input_ids)


    if tokenizer.pad_token_id is not None:
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline = emb_layer(baseline_ids)
    else:
        baseline = torch.zeros_like(embeddings)

    diff = embeddings - baseline


    total_grads = torch.zeros_like(embeddings)

    for step in range(1, n_steps + 1):
        alpha = float(step) / n_steps

        interpolated = baseline + alpha * diff
        interpolated = interpolated.detach().requires_grad_(True)

        outputs = model(
            inputs_embeds=interpolated,
            attention_mask=attention_mask,
        )
        logit = outputs.logits[0, target_label_id]

        model.zero_grad()
        grads = torch.autograd.grad(logit, interpolated)[0]
        total_grads += grads


    avg_grads = total_grads / n_steps


    ig = diff * avg_grads


    token_importance = ig.sum(dim=-1).squeeze(0)


    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())


    attn = attention_mask[0]
    results = []
    for tok, imp, mask_val in zip(tokens, token_importance.tolist(), attn.tolist()):
        if mask_val == 0:
            continue

        if tok in tokenizer.all_special_tokens:
            continue
        results.append((tok, imp))


    results.sort(key=lambda x: x[1], reverse=True)

    return results


app = Dash(__name__, serve_locally=True)

app.layout = html.Div(
    style={"fontFamily": "Arial", "maxWidth": "1100px", "margin": "0 auto"},
    children=[
        html.H2("Stylometric analysis"),
        html.Div(
            style={"display": "flex", "gap": "20px", "marginBottom": "20px"},
            children=[
                html.Div(
                    children=[
                        html.Label("Autor"),
                        dcc.Dropdown(
                            id="author-dropdown",          # <<< TO ID
                            options=[{"label": a, "value": a} for a in author_options],
                            value=author_options[0] if author_options else None,
                            clearable=False,
                        ),
                    ],
                    style={"flex": "1"},
                ),
                html.Div(
                    children=[
                        html.Label("Stylometric metric"),
                        dcc.Dropdown(
                            id="metric-dropdown",
                            options=metric_options,
                            value="avg_word_length",
                            clearable=False,
                        ),
                    ],
                    style={"flex": "1"},
                ),
            ],
        ),
        html.Div(
            style={"display": "flex", "gap": "30px"},
            children=[
                html.Div(
                    dcc.Graph(id="metric-graph"),
                    style={"flex": "2"},
                ),
                html.Div(
                    id="bigrams-div",
                    style={"flex": "1"},
                ),
            ],
        ),
        html.Div(
            id="summary-div",
            style={"marginTop": "20px", "fontSize": "16px"},
        ),

        html.Hr(),
        html.Div(
            children=[
                html.H4("Compare a new text with an author's general style"),
                html.P(
                    "Paste text into the field below"
                    "and we will calculate how closely its writing style resembles that of the author selected above."
                ),
                dcc.Textarea(
                    id="custom-text",
                    style={
                        "width": "100%",
                        "height": "150px",
                        "fontFamily": "monospace",
                    },
                    placeholder="Paste text for an analysis",
                ),
                html.Div(
                    style={"marginTop": "10px", "display": "flex", "gap": "10px"},
                    children=[
                        html.Button(
                            "Analyze text",
                            id="analyze-button",
                            n_clicks=0,
                        ),
                        html.Button(
                            "Run BERT training + inference",
                            id="train-bert-button",
                            n_clicks=0,
                        ),
                    ],
                ),
                html.Div(
                    id="bert-output",
                    style={
                        "marginTop": "10px",
                        "whiteSpace": "pre-wrap",
                        "fontFamily": "monospace",
                        "fontSize": "14px",
                    },
                ),
                dcc.Graph(
                    id="similarity-graph",
                    style={"marginTop": "20px", "height": "300px"},
                ),

            ],
            style={"marginTop": "30px"},
        ),
    ],
)



@app.callback(
    Output("metric-graph", "figure"),
    Output("bigrams-div", "children"),
    Output("summary-div", "children"),
    Input("author-dropdown", "value"),
    Input("metric-dropdown", "value"),
)

def update_graph_and_bigrams(selected_author, selected_metric):
    if selected_author is None or selected_metric is None:
        return px.line(), html.Div("No data."), html.Div()

    sub = df_features[df_features["author"] == selected_author].copy()

    if sub.empty:
        return (
            px.line(title="No data for the selected author"),
            html.Div("No data for the selected author"),
            html.Div("No data for the selected author"),
        )

    label = METRIC_LABELS[selected_metric]

    fig = px.line(
        sub,
        x="post_index",
        y=selected_metric,
        markers=True,
        title=f"{label} w kolejnych tweetach autora: {selected_author}",
    )


    stats = GLOBAL_STATS[selected_metric]
    mean_global = stats["mean"]
    q1 = stats["q1"]
    q3 = stats["q3"]

    fig.add_hrect(
        y0=q1,
        y1=q3,
        fillcolor="lightgray",
        opacity=0.2,
        line_width=0,
        annotation_text="Q1–Q3 globalnie",
        annotation_position="top left",
    )

    fig.add_hline(
        y=mean_global,
        line_dash="dash",
        line_color="black",
        annotation_text="Średnia globalna",
        annotation_position="bottom left",
    )

    fig.update_layout(
        xaxis_title="Order of the author's posts",
        yaxis_title=label,
        template="plotly_white",
        legend_title_text="",
    )

    counter = author_bigrams.get(selected_author, Counter())
    top10 = counter.most_common(10)

    if not top10:
        bigrams_block = html.Div("No bigrams for the selected author.")
    else:
        bigrams_block = html.Div(
            children=[
                html.H4(f"Top 10 bigrams– {selected_author}"),
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Bigram"),
                                    html.Th("Count"),
                                ]
                            )
                        ),
                        html.Tbody(
                            [
                                html.Tr([html.Td(bg), html.Td(count)])
                                for bg, count in top10
                            ]
                        ),
                    ],
                    style={"width": "100%", "borderCollapse": "collapse"},
                ),
            ],
            style={"fontSize": "14px"},
        )


    above = []
    below = []

    for metric_key, metric_label in METRIC_LABELS.items():
        author_values = sub[metric_key].dropna()
        if author_values.empty:
            continue

        author_mean = float(author_values.mean())
        stats_all = GLOBAL_STATS[metric_key]
        mean_global = stats_all["mean"]
        q1 = stats_all["q1"]
        q3 = stats_all["q3"]

        if author_mean > q3:
            above.append((metric_label, author_mean, mean_global))
        elif author_mean < q1:
            below.append((metric_label, author_mean, mean_global))


    children = [html.Strong(f"The style of {selected_author} is characterized by:")]

    if above:
        children.append(html.Br())
        children.append(html.Span("above-average:"))
        children.append(
            html.Ul([
                html.Li(
                    f"{label} "
                    f"(average for an author: {a_mean:.2f}, typical style: {g_mean:.2f})"
                )
                for (label, a_mean, g_mean) in above
            ])
        )

    if below:
        children.append(html.Br())
        children.append(html.Span("below average:"))
        children.append(
            html.Ul([
                html.Li(
                    f"{label} "
                    f"(average for an author: {a_mean:.2f}, typical style: {g_mean:.2f})"
                )
                for (label, a_mean, g_mean) in below
            ])
        )

    if not above and not below:
        children.append(html.Span(" with all metric values within the typical range"))

    summary_block = html.Div(children)

    return fig, bigrams_block, summary_block

@app.callback(
    Output("similarity-graph", "figure"),
    Input("analyze-button", "n_clicks"),
    State("custom-text", "value"),
    State("author-dropdown", "value"),
)
def analyze_custom_text(n_clicks, text, selected_author):
    import numpy as np
    import plotly.express as px


    if not n_clicks or not text or not selected_author:
        fig = px.bar(title="Paste text and click analyze”.")
        fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False})
        return fig


    text_feats = compute_features(text)


    sub = df_features[df_features["author"] == selected_author]
    if sub.empty:
        fig = px.bar(title="No data for selected author.")
        fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False})
        return fig

    metrics = []
    similarities = []

    for metric_key, metric_label in METRIC_LABELS.items():
        if metric_key not in text_feats:
            continue

        author_values = sub[metric_key].dropna()
        if author_values.empty:
            continue

        author_mean = float(author_values.mean())
        text_value = float(text_feats[metric_key])
        stats = GLOBAL_STATS[metric_key]
        mean_global = stats["mean"]
        std_global = stats["std"] or 1.0


        author_z = (author_mean - mean_global) / std_global
        text_z = (text_value - mean_global) / std_global


        dist = abs(author_z - text_z)


        similarity = 1.0 / (1.0 + dist)

        metrics.append(metric_label)
        similarities.append(similarity)

    if not metrics:
        fig = px.bar(title="No metrics available.")
        fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False})
        return fig

    overall_sim = float(np.mean(similarities)) * 100.0

    fig = px.bar(
        x=metrics,
        y=similarities,
        labels={"x": "Metric", "y": "Similarity (0–1)"},
        title=(
            f"Style similarity of the new text to the author {selected_author} "
            f"(average ~{overall_sim:.0f}%)"
        ),
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_range=[0, 1],
        template="plotly_white",
    )

    return fig


@app.callback(
    Output("bert-output", "children"),
    Input("train-bert-button", "n_clicks"),
    State("custom-text", "value"),
    State("author-dropdown", "value"),
    prevent_initial_call=True,
)
def train_and_infer_bert(n_clicks, text, selected_author):
    if not n_clicks:
        return ""

    if not selected_author:
        return "Select the author."

    if not text or not text.strip():
        return "Paste text."


    try:
        report, tokenizer, model = train_user_vs_background(selected_author)
    except Exception as e:
        return f"Error during training: {e}"


    bert_cache[selected_author] = (tokenizer, model)

    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        outputs = model(**{k: v for k, v in inputs.items()})
        probs = torch.softmax(outputs.logits, dim=-1).numpy()[0]
        prob_bg, prob_user = float(probs[0]), float(probs[1])


    try:
        ig_results = integrated_gradients(
            text,
            tokenizer,
            model,
            target_label_id=1,
            max_length=128,
            n_steps=20,
        )

        top_k = 10
        top_tokens = ig_results[:top_k]
        ig_list = html.Ul([
            html.Li(f"{tok}: {imp:.4f}")
            for tok, imp in top_tokens
        ])
    except Exception as e:
        ig_list = html.Div(f"Error while computing Integrated Gradients: {e}")

    return [
        html.Strong(f"Training and inference results for the author: {selected_author}"),
        html.Br(),
        html.Span(
            f"Probability that this text was written by the author {selected_author}: "
            f"{prob_user * 100:.1f}%"
        ),
        html.Br(),
        html.Span(
            f"Probability that this was written by the background (another author): "
            f"{prob_bg * 100:.1f}%"
        ),
        html.Br(),
        html.Br(),
        html.Span("Top Integrated-Gradients tokens (most revealing):"),
        ig_list,
        html.Br(),
        html.Span("Evaluation report (test set)"),
        html.Pre(report),
    ]

if __name__ == "__main__":
    print(df_features.head())

    app.run(
        debug=False,
        host="127.0.0.1",
        port=8050,
    )






