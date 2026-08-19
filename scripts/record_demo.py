"""Record the README demo GIF by driving the running app in headless Chrome.

Start the app first, then:

    streamlit run app.py
    python scripts/record_demo.py

Writes assets/demo.gif. The sequence is chosen to show the three things that
are hard to convey in text: the vector-store vs web-search routing, the
parallel tool timeline, and a follow-up answered from memory.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "demo.gif"
URL = "http://localhost:8502"

WIDTH, HEIGHT = 1340, 1180
SCALE = 0.70  # downscale on save to keep the GIF small enough for a README


class Recorder:
    def __init__(self) -> None:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument(f"--window-size={WIDTH},{HEIGHT}")
        options.add_argument("--hide-scrollbars")
        options.add_argument("--force-device-scale-factor=1")
        options.add_argument("--disable-gpu")
        self.driver = webdriver.Chrome(options=options)
        self.frames: list[tuple[Image.Image, int]] = []

    # ------------------------------------------------------------- capture
    def shoot(self, hold_ms: int = 1400, times: int = 1) -> None:
        """Append the current view, held for hold_ms."""
        png = self.driver.get_screenshot_as_png()
        image = Image.open(io.BytesIO(png)).convert("RGB")
        for _ in range(times):
            self.frames.append((image, hold_ms))
        print(f"  frame {len(self.frames):>2}  {hold_ms}ms")

    def js(self, script: str, *args):
        return self.driver.execute_script(script, *args)

    def text(self) -> str:
        return self.js("return document.body.innerText") or ""

    def wait_for(self, needle: str, timeout: float = 60.0, absent: bool = False) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            present = needle in self.text()
            if present != absent:
                return True
            time.sleep(0.25)
        print(f"  ! timed out waiting for {needle!r} (absent={absent})")
        return False

    def settle(self, seconds: float = 0.6) -> None:
        time.sleep(seconds)

    # -------------------------------------------------------------- actions
    def ask(self, prompt: str) -> None:
        """Type into the chat box and submit, capturing the typed state."""
        self.js(
            """
            const ta = document.querySelector('textarea');
            const setter = Object.getOwnPropertyDescriptor(
                window.HTMLTextAreaElement.prototype, 'value').set;
            setter.call(ta, arguments[0]);
            ta.dispatchEvent(new Event('input', {bubbles: true}));
            """,
            prompt,
        )
        self.settle(0.4)
        self.shoot(900)
        self.js(
            "document.querySelector('button[data-testid=\"stChatInputSubmitButton\"]').click()"
        )

    def scroll_to(self, selector: str, index: int = 0, offset: int = -90) -> None:
        self.js(
            """
            const els = document.querySelectorAll(arguments[0]);
            const el = els[arguments[1]];
            if (el) {
                const y = el.getBoundingClientRect().top + window.scrollY + arguments[2];
                window.scrollTo({top: y, behavior: 'instant'});
            }
            """,
            selector,
            index,
            offset,
        )
        self.settle(0.5)

    def scroll_to_text(self, needle: str, offset: int = -80) -> bool:
        """Put the smallest element containing ``needle`` near the top.

        More robust than CSS selectors here: the interesting things in the UI
        are identified by their labels, not by stable test ids.
        """
        return bool(
            self.js(
                """
                const needle = arguments[0];
                const offset = arguments[1];
                const hits = [...document.querySelectorAll('div, p, h1, h2, h3, span')]
                    .filter(e => e.innerText && e.innerText.includes(needle));
                if (!hits.length) return false;
                const el = hits[hits.length - 1];

                // Streamlit scrolls an inner container, not the window, so
                // scrollIntoView plus a nudge on whichever ancestor actually
                // scrolls is the only reliable way to position a frame.
                el.scrollIntoView({block: 'start', behavior: 'instant'});
                let node = el.parentElement;
                while (node && node.scrollHeight <= node.clientHeight + 1) {
                    node = node.parentElement;
                }
                const scroller = node || document.scrollingElement;
                scroller.scrollTop += offset;
                return true;
                """,
                needle,
                offset,
            )
        )

    def wait_images(self, timeout: float = 30.0) -> bool:
        """Block until the gallery has finished loading.

        Scrolling before the images land is pointless: the page grows underneath
        and Streamlit re-anchors to the bottom, undoing the scroll.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.js(
                "const i = [...document.images];"
                "return i.length > 0 && i.every(x => x.complete && x.naturalWidth > 0);"
            ):
                return True
            time.sleep(0.3)
        return False

    def frame_answer(self, badge_text: str) -> None:
        """Frame the newest answer on its source badge, city heading above it."""
        self.wait_images()
        self.settle(0.8)
        self.scroll_to_text(badge_text, offset=-170)
        self.settle(0.4)
        self.scroll_to_text(badge_text, offset=-170)  # settle any late reflow
        self.settle(0.3)

    def click_text(self, needle: str) -> bool:
        return bool(
            self.js(
                """
                const t = arguments[0];
                const el = [...document.querySelectorAll('summary, button, div[role=button]')]
                    .find(e => e.innerText && e.innerText.includes(t));
                if (el) { el.click(); return true; }
                return false;
                """,
                needle,
            )
        )

    def toggle(self, label: str) -> bool:
        return bool(
            self.js(
                """
                const label = [...document.querySelectorAll('label')]
                    .find(l => l.innerText.trim().startsWith(arguments[0]));
                if (!label) return false;
                const box = label.querySelector('input[type=checkbox]');
                if (!box) return false;
                box.click();
                return true;
                """,
                label,
            )
        )

    # ---------------------------------------------------------------- save
    def save(self, path: Path) -> None:
        if not self.frames:
            raise RuntimeError("no frames captured")
        size = (int(WIDTH * SCALE), int(self.frames[0][0].height * SCALE))
        images = [f.resize(size, Image.LANCZOS) for f, _ in self.frames]
        durations = [d for _, d in self.frames]

        palette = images[0].convert("P", palette=Image.ADAPTIVE, colors=128)
        converted = [im.quantize(palette=palette, dither=Image.FLOYDSTEINBERG) for im in images]

        path.parent.mkdir(parents=True, exist_ok=True)
        converted[0].save(
            path,
            save_all=True,
            append_images=converted[1:],
            duration=durations,
            loop=0,
            optimize=True,
            disposal=2,
        )
        print(f"\nwrote {path.relative_to(ROOT)}  "
              f"{path.stat().st_size / 1_000_000:.2f} MB  {len(converted)} frames  {size[0]}x{size[1]}")

    def quit(self) -> None:
        self.driver.quit()


def main() -> int:
    rec = Recorder()
    try:
        print("loading app...")
        rec.driver.get(URL)
        if not rec.wait_for("Knowledge base:", timeout=90):
            print("app did not finish starting; is `streamlit run app.py` up?")
            return 1
        rec.settle(1.0)

        # 1. The landing state: provider, knowledge base, controls.
        print("scene 1: landing")
        rec.shoot(1800)

        # 2. A city that lives in the vector store.
        print("scene 2: vector-store answer")
        rec.ask("Tell me about Tokyo")
        rec.settle(0.8)
        rec.shoot(800)  # spinner
        rec.wait_for("Answered from the local ChromaDB corpus", timeout=90)
        rec.settle(1.4)
        rec.frame_answer("Answered from the local ChromaDB corpus")
        rec.shoot(2600)  # city, source badge, summary

        rec.scroll_to_text("Best time to visit", offset=-120)
        rec.shoot(2400)  # forecast chart

        rec.scroll_to_text("How this answer was produced", offset=-360)
        rec.shoot(2000)  # gallery

        # 3. The trace: routing reason, tool calls, measured parallelism.
        print("scene 3: trace panel")
        if rec.click_text("How this answer was produced"):
            rec.settle(1.4)
            rec.scroll_to_text("Fan-out speedup", offset=-120)
            rec.shoot(2600)
            rec.scroll_to_text("milliseconds since turn start", offset=-320)
            rec.shoot(2600)
            rec.click_text("How this answer was produced")
            rec.settle(0.6)

        # 4. Follow-up: city remembered, only the weather tool re-runs.
        print("scene 4: memory follow-up")
        rec.js("window.scrollTo({top: document.body.scrollHeight, behavior: 'instant'})")
        rec.settle(0.4)
        rec.ask("What about next week?")
        rec.settle(0.8)
        rec.shoot(800)
        rec.wait_for("Reused from the checkpointed conversation state", timeout=90)
        rec.settle(1.4)
        rec.frame_answer("Reused from the checkpointed conversation state")
        rec.shoot(3000)  # Memory badge, same city, shifted forecast

        # 5. A city the corpus does not cover: the other side of the switch.
        print("scene 5: web-search fallback")
        rec.js("window.scrollTo({top: document.body.scrollHeight, behavior: 'instant'})")
        rec.settle(0.4)
        rec.ask("Tell me about Kyoto")
        rec.settle(0.8)
        rec.wait_for("City not in the corpus", timeout=90)
        rec.settle(1.4)
        rec.frame_answer("City not in the corpus")
        rec.shoot(3000)  # Web search badge

        rec.scroll_to_text("Sources", offset=-420)
        rec.shoot(2600)  # Kyoto gallery

        rec.save(OUT)
        return 0
    finally:
        rec.quit()


if __name__ == "__main__":
    raise SystemExit(main())
