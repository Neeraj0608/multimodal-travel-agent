"""Write graph.png (and graph.mmd) from the compiled LangGraph topology.

Three strategies, tried in order, so this works on a machine with no Graphviz
binaries and no internet:

1. LangGraph's Mermaid renderer via mermaid.ink   (needs network)
2. pygraphviz                                      (needs the Graphviz binaries)
3. a local Pillow renderer                         (always available)

Usage:  python scripts/export_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from travel_agent import bootstrap  # noqa: E402

bootstrap()

from travel_agent.graph.builder import build_graph  # noqa: E402

OUT_PNG = ROOT / "graph.png"
OUT_MMD = ROOT / "assets" / "graph.mmd"


def _mermaid_source(graph) -> str:
    return graph.get_graph().draw_mermaid()


def try_mermaid_ink(graph) -> bytes | None:
    try:
        return graph.get_graph().draw_mermaid_png()
    except Exception as exc:  # noqa: BLE001
        print(f"  mermaid.ink renderer unavailable: {type(exc).__name__}: {str(exc)[:120]}")
        return None


def try_pygraphviz(graph) -> bytes | None:
    try:
        return graph.get_graph().draw_png()
    except Exception as exc:  # noqa: BLE001
        print(f"  pygraphviz renderer unavailable: {type(exc).__name__}: {str(exc)[:120]}")
        return None


def draw_locally() -> bytes:
    """Hand-rolled diagram of the topology using Pillow only."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1180, 940
    background = (255, 255, 255)
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)

    def font(size: int, bold: bool = False):
        for name in (
            "seguisb.ttf" if bold else "segoeui.ttf",
            "arialbd.ttf" if bold else "arial.ttf",
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        ):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    title_font, node_font, edge_font, small_font = font(26, True), font(17, True), font(14), font(13)

    # (key, label, subtitle, x, y, w, h, fill, outline)
    nodes = {
        "start": ("START", "", 500, 40, 180, 46, (241, 245, 249), (148, 163, 184)),
        "planner": ("planner", "resolve city + what this turn needs", 420, 132, 340, 66, (237, 233, 254), (139, 92, 246)),
        "clarify": ("clarify", "no city resolved", 60, 250, 250, 60, (243, 244, 246), (107, 114, 128)),
        "retrieve": ("retrieve_knowledge", "ChromaDB similarity + city filter", 400, 258, 380, 66, (219, 234, 254), (37, 99, 235)),
        "web": ("web_search", "fallback for uncovered cities", 830, 372, 300, 66, (254, 243, 199), (217, 119, 6)),
        "toolplan": ("tool_planner", "model emits raw tool_calls", 400, 462, 380, 66, (207, 250, 254), (8, 145, 178)),
        "exec1": ("execute_tool", "get_weather_forecast", 250, 606, 290, 62, (209, 250, 229), (5, 150, 105)),
        "exec2": ("execute_tool", "search_city_images", 610, 606, 290, 62, (209, 250, 229), (5, 150, 105)),
        "composer": ("composer", "validated TravelReport (Pydantic)", 400, 742, 380, 66, (252, 231, 243), (219, 39, 119)),
        "end": ("END", "", 500, 856, 180, 46, (241, 245, 249), (148, 163, 184)),
    }

    def centre(key: str) -> tuple[int, int]:
        _, _, x, y, w, h, _, _ = nodes[key]
        return x + w // 2, y + h // 2

    def bottom(key: str) -> tuple[int, int]:
        _, _, x, y, w, h, _, _ = nodes[key]
        return x + w // 2, y + h

    def top(key: str) -> tuple[int, int]:
        _, _, x, y, w, _, _, _ = nodes[key]
        return x + w // 2, y

    def arrow(p1: tuple[int, int], p2: tuple[int, int], label: str = "", dashed: bool = False,
              colour: tuple[int, int, int] = (100, 116, 139)) -> None:
        if dashed:
            total = max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]), 1)
            steps = max(int(total / 9), 1)
            for i in range(steps):
                if i % 2:
                    continue
                a = (p1[0] + (p2[0] - p1[0]) * i / steps, p1[1] + (p2[1] - p1[1]) * i / steps)
                b = (p1[0] + (p2[0] - p1[0]) * (i + 1) / steps, p1[1] + (p2[1] - p1[1]) * (i + 1) / steps)
                draw.line([a, b], fill=colour, width=2)
        else:
            draw.line([p1, p2], fill=colour, width=2)

        # arrowhead
        import math

        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        for sign in (1, -1):
            tip = (
                p2[0] - 11 * math.cos(angle + sign * 0.42),
                p2[1] - 11 * math.sin(angle + sign * 0.42),
            )
            draw.line([p2, tip], fill=colour, width=2)

        if label:
            mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            box = draw.textbbox((0, 0), label, font=edge_font)
            tw, th = box[2] - box[0], box[3] - box[1]
            draw.rectangle(
                [mid[0] - tw / 2 - 5, mid[1] - th / 2 - 3, mid[0] + tw / 2 + 5, mid[1] + th / 2 + 3],
                fill=background,
            )
            draw.text((mid[0] - tw / 2, mid[1] - th / 2), label, fill=(71, 85, 105), font=edge_font)

    draw.text((40, 26), "Multi-Modal Travel Assistant - LangGraph topology", fill=(15, 23, 42), font=title_font)

    # edges first so nodes paint over the endpoints
    arrow(bottom("start"), top("planner"))
    arrow((nodes["planner"][2] + 20, nodes["planner"][5] + nodes["planner"][3]),
          (nodes["clarify"][2] + nodes["clarify"][4], nodes["clarify"][3] + 20),
          "no city", dashed=True, colour=(107, 114, 128))
    arrow(bottom("planner"), top("retrieve"), "needs summary")
    arrow((nodes["planner"][2] + nodes["planner"][4], nodes["planner"][3] + 50),
          (nodes["toolplan"][2] + nodes["toolplan"][4] + 10, nodes["toolplan"][3] + 10),
          "cached summary (memory)", dashed=True, colour=(5, 150, 105))
    arrow((nodes["retrieve"][2] + nodes["retrieve"][4], nodes["retrieve"][3] + 40),
          (nodes["web"][2], nodes["web"][3] + 20), "miss", colour=(217, 119, 6))
    arrow(bottom("retrieve"), top("toolplan"), "hit", colour=(37, 99, 235))
    arrow((nodes["web"][2] + 60, nodes["web"][3] + nodes["web"][5]),
          (nodes["toolplan"][2] + nodes["toolplan"][4] - 40, nodes["toolplan"][3]), colour=(217, 119, 6))
    arrow((nodes["toolplan"][2] + 90, nodes["toolplan"][3] + nodes["toolplan"][5]), top("exec1"), "Send", colour=(5, 150, 105))
    arrow((nodes["toolplan"][2] + 290, nodes["toolplan"][3] + nodes["toolplan"][5]), top("exec2"), "Send", colour=(5, 150, 105))
    arrow(bottom("exec1"), (nodes["composer"][2] + 90, nodes["composer"][3]))
    arrow(bottom("exec2"), (nodes["composer"][2] + 290, nodes["composer"][3]))
    arrow(bottom("composer"), top("end"))
    arrow((nodes["clarify"][2] + nodes["clarify"][4] // 2, nodes["clarify"][3] + nodes["clarify"][5]),
          (nodes["end"][2] - 10, nodes["end"][3] + 20), dashed=True, colour=(107, 114, 128))

    for label, subtitle, x, y, w, h, fill, outline in nodes.values():
        draw.rounded_rectangle([x, y, x + w, y + h], radius=12, fill=fill, outline=outline, width=2)
        text_y = y + (h - (34 if subtitle else 18)) // 2
        draw.text((x + 16, text_y), label, fill=(15, 23, 42), font=node_font)
        if subtitle:
            draw.text((x + 16, text_y + 22), subtitle, fill=(71, 85, 105), font=small_font)

    legend = [
        ((5, 150, 105), "parallel fan-out: one Send per tool call"),
        ((37, 99, 235), "vector-store hit"),
        ((217, 119, 6), "web-search fallback (conditional edge)"),
    ]
    for index, (colour, text) in enumerate(legend):
        y = 690 + index * 24
        draw.rectangle([60, y, 78, y + 14], fill=colour)
        draw.text((88, y - 2), text, fill=(51, 65, 85), font=small_font)

    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> int:
    graph = build_graph()

    OUT_MMD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MMD.write_text(_mermaid_source(graph), encoding="utf-8")
    print(f"wrote {OUT_MMD.relative_to(ROOT)}")

    png = try_mermaid_ink(graph) or try_pygraphviz(graph)
    source = "langgraph"
    if png is None:
        png = draw_locally()
        source = "local Pillow renderer"

    OUT_PNG.write_bytes(png)
    print(f"wrote {OUT_PNG.relative_to(ROOT)} ({len(png):,} bytes) via {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
