"""Streamlit rendering helpers.

Kept separate from ``app.py`` so the view logic stays testable and the app file
reads as a script. Every function here takes plain dicts - the UI never touches
the graph or the model directly.
"""

from __future__ import annotations

import json
from typing import Any

import plotly.graph_objects as go
import streamlit as st

SOURCE_BADGE = {
    "vector_store": ("Vector store", "#2563eb", "Answered from the local ChromaDB corpus"),
    "web_search": ("Web search", "#d97706", "City not in the corpus - routed to search"),
    "memory": ("Memory", "#059669", "Reused from the checkpointed conversation state"),
    "model_prior": ("Model prior", "#6b7280", "No retrieval succeeded; model knowledge only"),
}


def source_badge(source: str) -> None:
    label, colour, explanation = SOURCE_BADGE.get(source, SOURCE_BADGE["model_prior"])
    st.markdown(
        f"<span style='background:{colour};color:#fff;padding:2px 10px;border-radius:999px;"
        f"font-size:0.75rem;font-weight:600;letter-spacing:.02em'>{label}</span>"
        f"<span style='opacity:.65;font-size:0.78rem;margin-left:.6rem'>{explanation}</span>",
        unsafe_allow_html=True,
    )


def weather_chart(points: list[dict[str, Any]], city: str) -> go.Figure:
    """Temperature range plus precipitation probability on a secondary axis."""
    dates = [p["date"] for p in points]
    labels = [f"{p.get('weekday', '')} {p['date'][5:]}" for p in points]
    highs = [p["temp_max_c"] for p in points]
    lows = [p["temp_min_c"] for p in points]
    precip = [p.get("precipitation_chance", 0) for p in points]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dates,
            y=precip,
            name="Rain chance",
            yaxis="y2",
            marker_color="rgba(96,165,250,0.30)",
            hovertemplate="%{y}% chance<extra></extra>",
        )
    )
    # Shaded band between the daily low and high.
    fig.add_trace(
        go.Scatter(
            x=dates + dates[::-1],
            y=highs + lows[::-1],
            fill="toself",
            fillcolor="rgba(239,68,68,0.10)",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=highs,
            name="High",
            mode="lines+markers",
            line={"color": "#ef4444", "width": 3},
            hovertemplate="High %{y:.1f}°C<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=lows,
            name="Low",
            mode="lines+markers",
            line={"color": "#3b82f6", "width": 3, "dash": "dot"},
            hovertemplate="Low %{y:.1f}°C<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{city}: {len(points)}-day outlook",
        height=380,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
        xaxis={"tickmode": "array", "tickvals": dates, "ticktext": labels},
        yaxis={"title": "°C"},
        yaxis2={
            "title": "Rain %",
            "overlaying": "y",
            "side": "right",
            "range": [0, 100],
            "showgrid": False,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def timeline_chart(traces: list[dict[str, Any]]) -> go.Figure:
    """Gantt-style node timeline - this is where parallelism becomes visible."""
    ordered = sorted(traces, key=lambda t: t.get("started_at", 0))
    origin = min((t.get("started_at", 0) for t in ordered), default=0)
    colours = {
        "planner": "#8b5cf6",
        "retrieve_knowledge": "#2563eb",
        "web_search": "#d97706",
        "tool_planner": "#0891b2",
        "execute_tool": "#059669",
        "composer": "#db2777",
        "clarify": "#6b7280",
    }

    fig = go.Figure()
    for index, trace in enumerate(ordered):
        start = (trace.get("started_at", 0) - origin) * 1000
        width = max(float(trace.get("duration_ms", 0)), 1.0)
        label = trace.get("label", trace.get("node", ""))
        fig.add_trace(
            go.Bar(
                x=[width],
                y=[f"{index}. {label}"],
                base=[start],
                orientation="h",
                marker_color=colours.get(trace.get("node", ""), "#6b7280"),
                hovertemplate=(
                    f"<b>{label}</b><br>start +%{{base:.0f}} ms<br>"
                    f"duration {width:.0f} ms<br>{trace.get('detail', '')[:120]}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    fig.update_layout(
        height=max(180, 42 * len(ordered)),
        margin={"l": 10, "r": 10, "t": 10, "b": 30},
        xaxis={"title": "milliseconds since turn start"},
        yaxis={"autorange": "reversed"},
        bargap=0.35,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _show_image(url: str, caption: str | None) -> None:
    """``st.image`` renamed its width argument across Streamlit versions.

    1.41+ uses ``use_container_width``; earlier releases use
    ``use_column_width``. Try newest first so the code works on both without
    pinning Streamlit.
    """
    for kwargs in (
        {"use_container_width": True},
        {"use_column_width": True},
        {},
    ):
        try:
            st.image(url, caption=caption, **kwargs)
            return
        except TypeError:
            continue
        except Exception:  # noqa: BLE001 - a dead URL must not break the page
            break
    st.caption("Image unavailable")


def gallery(images: list[dict[str, Any]], columns: int = 4) -> None:
    if not images:
        return
    for row_start in range(0, len(images), columns):
        row = images[row_start : row_start + columns]
        for column, image in zip(st.columns(len(row)), row):
            with column:
                _show_image(image.get("url", ""), image.get("caption", "") or None)


def render_report(report: dict[str, Any], *, key: str) -> None:
    """The main answer card: text, chart, gallery."""
    city = report.get("city") or "Unknown destination"
    country = report.get("country") or ""

    st.subheader(f"{city}{f', {country}' if country else ''}")
    source_badge(report.get("knowledge_source", "model_prior"))

    for warning in report.get("warnings", []):
        st.warning(warning, icon=":material/warning:")

    st.markdown(report.get("city_summary", ""))

    highlights = report.get("highlights") or []
    if highlights:
        st.markdown(
            " ".join(
                f"<span style='display:inline-block;background:rgba(127,127,127,.14);"
                f"padding:3px 10px;border-radius:999px;margin:3px 4px 3px 0;font-size:.82rem'>"
                f"{h}</span>"
                for h in highlights
            ),
            unsafe_allow_html=True,
        )
    if report.get("best_time_to_visit"):
        st.caption(f"Best time to visit: {report['best_time_to_visit']}")

    forecast = report.get("weather_forecast") or []
    if forecast:
        st.plotly_chart(
            weather_chart(forecast, city), use_container_width=True, key=f"{key}-weather"
        )
    else:
        st.info("No forecast data available for this turn.", icon=":material/cloud_off:")

    images = report.get("images") or [{"url": u} for u in report.get("image_urls", [])]
    if images:
        gallery(images)
    else:
        st.caption("No images available for this turn.")

    if report.get("sources"):
        with st.expander("Sources"):
            for url in report["sources"]:
                st.markdown(f"- {url}")


def render_trace(turn_data: dict[str, Any], *, key: str) -> None:
    """Diagnostics: the parts that prove *how* the answer was produced."""
    traces = turn_data.get("traces", [])
    results = turn_data.get("tool_results", [])
    stats = turn_data.get("stats", {})

    with st.expander("How this answer was produced", expanded=False):
        route_col, plan_col, speed_col = st.columns(3)
        route_col.metric("Knowledge route", turn_data.get("route", "-"))
        plan_col.metric("Tool calls", len(turn_data.get("tool_calls", [])))
        speed_col.metric(
            "Fan-out speedup",
            f"{stats.get('speedup', 1.0)}x",
            delta=f"-{stats.get('saved_ms', 0):.0f} ms",
            help=(
                "Summed tool time versus wall-clock time for the parallel step. "
                "Above 1.0 means the tools genuinely overlapped."
            ),
        )

        plan = turn_data.get("plan", {})
        if plan.get("rationale"):
            st.caption(f"Planner: {plan['rationale']}")

        if traces:
            st.plotly_chart(
                timeline_chart(traces), use_container_width=True, key=f"{key}-timeline"
            )

        # Streamlit forbids nesting expanders, so the detail views are tabs.
        tabs = st.tabs(
            ["Tool executions", "Raw tool_calls payload", "Structured output (Pydantic)", "Node log"]
        )
        with tabs[0]:
            if not results:
                st.caption("No tools ran this turn.")
            for result in results:
                with st.container(border=True):
                    status = (
                        f":green[ran in {result.get('duration_ms', 0):.0f} ms]"
                        if result.get("ok")
                        else ":red[FAILED]"
                    )
                    st.markdown(f"**{result.get('name')}** · {status}")
                    st.caption("arguments the model supplied")
                    st.code(json.dumps(result.get("args", {}), indent=2), language="json")
                    if result.get("ok"):
                        st.json(result.get("result", {}), expanded=False)
                    else:
                        st.error(result.get("error", "unknown error"))
        with tabs[1]:
            st.caption(
                "Exactly what the model emitted. Parsed and executed by our own "
                "node - no ToolNode, no bind_tools."
            )
            st.code(turn_data.get("raw_tool_payload", "(none)"), language="json")
        with tabs[2]:
            st.code(
                json.dumps(turn_data.get("report", {}), indent=2, default=str)[:20000],
                language="json",
            )
        with tabs[3]:
            for trace in traces:
                st.markdown(
                    f"**{trace.get('node')}** · {trace.get('duration_ms', 0):.0f} ms  \n"
                    f"<span style='opacity:.7;font-size:.82rem'>{trace.get('detail', '')}</span>",
                    unsafe_allow_html=True,
                )
