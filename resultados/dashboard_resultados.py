import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

WORKBOOK_PATH = Path(__file__).parent / "resultados_gerais.xlsx"
SHEET_NAME = "in"

REQUIRED_COLUMNS = [
    "Modelo",
    "Hardware",
    "Tamanho",
    "Idioma",
    "Intervalo_s",
    "Tempo_Processamento_s",
    "RTF",
    "WER",
    "Pico_RAM_MB",
    "Pico_VRAM_MB",
]

METRICS = [
    ("Tempo_Processamento_s", "Tempo de processamento (s)", True),
    ("RTF", "RTF (menor = melhor)", True),
    ("WER", "WER (menor = melhor)", True),
    ("Pico_RAM_MB", "Pico de RAM (MB)", True),
    ("Pico_VRAM_MB", "Pico de VRAM (MB)", True),
]

SIZE_ORDER = ["tiny", "base", "small", "medium"]
MODEL_ORDER = ["OpenAI-Whisper", "Faster-Whisper", "WhisperX"]


def _normalize_text_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _ordered_sizes(values: list[str]) -> list[str]:
    preferred = [size for size in SIZE_ORDER if size in values]
    remaining = sorted([size for size in values if size not in SIZE_ORDER])
    return preferred + remaining


def _ordered_models(values: list[str]) -> list[str]:
    preferred = [model for model in MODEL_ORDER if model in values]
    remaining = sorted([model for model in values if model not in MODEL_ORDER])
    return preferred + remaining


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Planilha não encontrada: {path}")

    df = pd.read_excel(path, sheet_name=SHEET_NAME)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Colunas obrigatórias ausentes na planilha: " + ", ".join(missing)
        )

    df = df[REQUIRED_COLUMNS].copy()
    df["Modelo"] = _normalize_text_col(df["Modelo"])
    df["Hardware"] = _normalize_text_col(df["Hardware"])
    df["Tamanho"] = _normalize_text_col(df["Tamanho"]).str.lower()
    df["Idioma"] = _normalize_text_col(df["Idioma"]).str.upper()

    for col in ["Intervalo_s", "Tempo_Processamento_s", "RTF", "WER", "Pico_RAM_MB", "Pico_VRAM_MB"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_filtered_df(df: pd.DataFrame) -> pd.DataFrame:
    english_df = df[df["Idioma"] == "EN"].copy()

    with st.sidebar:
        st.header("Filtros")

        hardware_options = sorted(english_df["Hardware"].dropna().unique().tolist())
        size_options = _ordered_sizes(english_df["Tamanho"].dropna().unique().tolist())
        interval_options = sorted(english_df["Intervalo_s"].dropna().astype(int).unique().tolist())
        model_options = _ordered_models(english_df["Modelo"].dropna().unique().tolist())

        selected_hardware = st.multiselect(
            "Hardware",
            options=hardware_options,
            default=hardware_options,
        )
        selected_sizes = st.multiselect(
            "Tamanho",
            options=size_options,
            default=size_options,
        )
        selected_intervals = st.multiselect(
            "Intervalo (s)",
            options=interval_options,
            default=interval_options,
        )
        selected_models = st.multiselect(
            "Modelo",
            options=model_options,
            default=model_options,
        )

    filtered = english_df[
        english_df["Hardware"].isin(selected_hardware)
        & english_df["Tamanho"].isin(selected_sizes)
        & english_df["Intervalo_s"].isin(selected_intervals)
        & english_df["Modelo"].isin(selected_models)
    ].copy()

    return filtered


def summarize_rankings(filtered: pd.DataFrame) -> pd.DataFrame:
    agg = (
        filtered.groupby(["Modelo", "Hardware", "Tamanho"], dropna=False)[
            ["Tempo_Processamento_s", "RTF", "WER", "Pico_RAM_MB", "Pico_VRAM_MB"]
        ]
        .mean()
        .reset_index()
    )

    # Score composto: média dos ranks por métrica (quanto menor, melhor)
    rank_cols = []
    for metric, _, _ in METRICS:
        rank_col = f"rank_{metric}"
        agg[rank_col] = agg[metric].rank(method="min", ascending=True)
        rank_cols.append(rank_col)

    agg["Score_Geral"] = agg[rank_cols].mean(axis=1)
    agg = agg.sort_values("Score_Geral", ascending=True)
    return agg


def _minmax_norm(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series([0.0] * len(series), index=series.index)
    min_v = valid.min()
    max_v = valid.max()
    if max_v == min_v:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - min_v) / (max_v - min_v)


def recommend_realtime_setup(filtered: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = (
        filtered.groupby(["Modelo", "Hardware", "Tamanho"], dropna=False)[
            ["Tempo_Processamento_s", "RTF", "WER", "Pico_RAM_MB", "Pico_VRAM_MB"]
        ]
        .mean()
        .reset_index()
    )

    # Menor é melhor para todas as métricas.
    weights = {
        "RTF": 0.45,
        "WER": 0.30,
        "Tempo_Processamento_s": 0.15,
        "Pico_RAM_MB": 0.07,
        "Pico_VRAM_MB": 0.03,
    }

    normalized = base.copy()
    for metric in weights:
        normalized[f"norm_{metric}"] = _minmax_norm(normalized[metric]).fillna(1.0)

    normalized["Score_Streaming"] = 0.0
    for metric, weight in weights.items():
        normalized["Score_Streaming"] += normalized[f"norm_{metric}"] * weight

    model_rank = {model: idx for idx, model in enumerate(_ordered_models(normalized["Modelo"].dropna().unique().tolist()))}
    size_rank = {size: idx for idx, size in enumerate(_ordered_sizes(normalized["Tamanho"].dropna().unique().tolist()))}
    normalized["_model_rank"] = normalized["Modelo"].map(model_rank).fillna(999)
    normalized["_size_rank"] = normalized["Tamanho"].map(size_rank).fillna(999)

    ranked = normalized.sort_values(
        ["Score_Streaming", "RTF", "WER", "_model_rank", "_size_rank"],
        ascending=[True, True, True, True, True],
    ).drop(columns=["_model_rank", "_size_rank"])

    realtime_viable = ranked[ranked["RTF"] <= 1.0].copy()
    return ranked, realtime_viable


def metric_cards(filtered: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros (EN)", int(len(filtered)))
    col2.metric("Modelos", int(filtered["Modelo"].nunique()))
    col3.metric("Hardwares", int(filtered["Hardware"].nunique()))
    col4.metric("Tamanhos", int(filtered["Tamanho"].nunique()))


def plot_metric_overview(filtered: pd.DataFrame, metric: str, label: str) -> None:
    if filtered.empty:
        return

    chart_df = (
        filtered.groupby(["Modelo", "Hardware", "Tamanho"], dropna=False)[metric]
        .mean()
        .reset_index()
    )
    size_order = _ordered_sizes(chart_df["Tamanho"].dropna().unique().tolist())
    model_order = _ordered_models(chart_df["Modelo"].dropna().unique().tolist())

    fig = px.bar(
        chart_df,
        x="Modelo",
        y=metric,
        color="Hardware",
        facet_col="Tamanho",
        barmode="group",
        title=f"{label} por modelo, hardware e tamanho (média)",
        labels={metric: label},
        category_orders={"Tamanho": size_order, "Modelo": model_order},
    )
    fig.update_layout(height=430, margin=dict(t=70, l=20, r=20, b=20))
    st.plotly_chart(fig, width="stretch")


def plot_interval_trend(filtered: pd.DataFrame) -> None:
    if filtered.empty:
        return

    trend_df = (
        filtered.groupby(["Intervalo_s", "Modelo", "Hardware"], dropna=False)["RTF"]
        .mean()
        .reset_index()
        .sort_values("Intervalo_s")
    )
    model_order = _ordered_models(trend_df["Modelo"].dropna().unique().tolist())

    fig = px.line(
        trend_df,
        x="Intervalo_s",
        y="RTF",
        color="Modelo",
        line_dash="Hardware",
        markers=True,
        title="Tendência de RTF por intervalo (menor = melhor)",
        labels={"Intervalo_s": "Intervalo (s)", "RTF": "RTF"},
        category_orders={"Modelo": model_order},
    )
    fig.update_layout(height=420, margin=dict(t=60, l=20, r=20, b=20))
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    st.set_page_config(
        page_title="Dashboard de Resultados - Whisper",
        page_icon="📊",
        layout="wide",
    )

    st.title("Benchmarks de Modelos de ASR")
    st.caption(
        "Fonte: resultados_gerais.xlsx | idioma fixado em EN | colunas de chunk ignoradas"
    )

    try:
        df = load_data(WORKBOOK_PATH)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    filtered = build_filtered_df(df)

    if filtered.empty:
        st.warning("Nenhum dado encontrado com os filtros atuais.")
        st.stop()

    metric_cards(filtered)

    st.subheader("Comparativo por métrica")
    selected_metric = st.selectbox(
        "Métrica principal",
        options=[m[0] for m in METRICS],
        format_func=lambda x: dict((m[0], m[1]) for m in METRICS)[x],
        index=1,
    )
    metric_label = dict((m[0], m[1]) for m in METRICS)[selected_metric]
    plot_metric_overview(filtered, selected_metric, metric_label)

    st.subheader("Efeito do intervalo")
    plot_interval_trend(filtered)

    st.subheader("Ranking agregado")
    st.caption("Score_Geral = média dos ranks em Tempo, RTF, WER, RAM e VRAM (menor é melhor)")
    ranking = summarize_rankings(filtered)
    st.dataframe(
        ranking[[
            "Modelo",
            "Hardware",
            "Tamanho",
            "Tempo_Processamento_s",
            "RTF",
            "WER",
            "Pico_RAM_MB",
            "Pico_VRAM_MB",
            "Score_Geral",
        ]],
        width="stretch",
        hide_index=True,
    )

    st.subheader("Dados filtrados")
    model_rank = {model: idx for idx, model in enumerate(_ordered_models(filtered["Modelo"].dropna().unique().tolist()))}
    size_rank = {size: idx for idx, size in enumerate(_ordered_sizes(filtered["Tamanho"].dropna().unique().tolist()))}
    filtered_sorted = (
        filtered.assign(
            _model_rank=filtered["Modelo"].map(model_rank).fillna(999),
            _size_rank=filtered["Tamanho"].map(size_rank).fillna(999),
        )
        .sort_values(["_model_rank", "Hardware", "_size_rank", "Intervalo_s"])
        .drop(columns=["_model_rank", "_size_rank"])
    )
    st.dataframe(
        filtered_sorted,
        width="stretch",
        hide_index=True,
    )

    st.subheader("Modelo que se destaca para Produção em Tempo Real")
    st.caption(
        "Cenário alvo: transcrição em streaming durante reuniões, com tradução e síntese de voz rodando ao mesmo tempo. "
        "A recomendação prioriza baixa latência (RTF), seguida de qualidade (WER)."
    )

    ranked_streaming, realtime_viable = recommend_realtime_setup(filtered)
    preferred = realtime_viable.head(1)
    if preferred.empty:
        preferred = ranked_streaming.head(1)
        st.warning("Nenhuma configuração com RTF <= 1 nos filtros atuais; exibindo melhor opção relativa.")

    chosen = preferred.iloc[0]
    st.success(
        f"Atual melhor modelo: {chosen['Modelo']} | {chosen['Hardware']} | {chosen['Tamanho']} "
        f"(RTF={chosen['RTF']:.3f}, WER={chosen['WER']:.4f}, Score={chosen['Score_Streaming']:.3f})"
    )

    st.markdown(
        "\n".join(
            [
                "Critério usado no score de streaming:",
                "- 45% RTF (latência)",
                "- 30% WER (qualidade)",
                "- 15% Tempo de processamento",
                "- 7% RAM",
                "- 3% VRAM",
            ]
        )
    )

    st.write("As 5 melhores configurações para streaming nos filtros atuais")
    st.dataframe(
        ranked_streaming[[
            "Modelo",
            "Hardware",
            "Tamanho",
            "RTF",
            "WER",
            "Tempo_Processamento_s",
            "Pico_RAM_MB",
            "Pico_VRAM_MB",
            "Score_Streaming",
        ]].head(5),
        width="stretch",
        hide_index=True,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help-run":
        print("Use: streamlit run dashboard_resultados.py")
        raise SystemExit(0)
    main()
