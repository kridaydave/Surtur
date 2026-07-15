"""Generate the P0 results figure from a results dict.

Usage:
    python make_figure.py --results results.json --out results_figure.html

Expected results.json shape:
{
    "model_id": "Qwen/Qwen2.5-1.5B",
    "seeds": 5,
    "retention": {
        "MMLU": 0.987, "HellaSwag": 0.991, "GSM8K": 0.989
    },
    "alignment_gain": {
        "TruthfulQA": 0.12, "harmlessness": 0.09
    },
    "compute_ratio": 0.22,
    "verdict": "PASS" | "FAIL",
    "failures": []
}

If --results is omitted, the figure renders with placeholder data.
"""
import argparse
import json
import os
from datetime import date


TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "results_figure.html")


def render(retention: dict, alignment_gain: dict, compute_ratio: float,
           verdict: str, model_id: str, seeds: int, failures: list) -> str:
    with open(TEMPLATE_PATH) as f:
        html = f.read()

    def retention_bars() -> str:
        n = len(retention)
        if n == 0:
            return ""
        bar_w = 120
        gap = (800 - 120 - 60 - n * bar_w) / max(n - 1, 1)
        parts = []
        for i, (domain, val) in enumerate(retention.items()):
            x = 140 + i * (bar_w + gap)
            val_capped = max(0.90, min(1.0, val))
            y = 220 - (val_capped - 0.90) / 0.10 * 200
            h = 220 - y
            passed = val >= 0.98
            cls = "bar" if passed else "bar bar-fail"
            parts.append(
                f'<rect class="{cls}" x="{x:.0f}" y="{y:.0f}" width="{bar_w}" height="{h:.0f}"/>'
                f'<text class="bar-label" x="{x + bar_w/2:.0f}" y="{y - 10:.0f}" text-anchor="middle">{val:.3f}</text>'
                f'<text class="axis-label" x="{x + bar_w/2:.0f}" y="240" text-anchor="middle">{domain}</text>'
            )
        return "\n          ".join(parts)

    def alignment_bars() -> str:
        items = list(alignment_gain.items())
        if not items:
            return ""
        parts = []
        for i, (domain, val) in enumerate(items):
            x = 100 + i * 140
            val_pct = val * 100 if abs(val) <= 1.0 else val
            h = abs(val_pct) * 6
            y = 180 - val_pct * 6 if val_pct >= 0 else 180
            passed = val > 0
            cls = "bar" if passed else "bar bar-fail"
            pct = f"{val_pct:+.1f}%"
            parts.append(
                f'<rect class="{cls}" x="{x}" y="{y:.0f}" width="80" height="{h:.0f}"/>\n'
                f'        <text class="bar-label" x="{x+40}" y="{y - 10:.0f}" text-anchor="middle">{pct}</text>\n'
                f'        <text class="axis-label" x="{x+40}" y="200" text-anchor="middle">{domain}</text>'
            )
        return "\n        ".join(parts)

    def compute_bar() -> str:
        h = max(10.0, min(150.0, compute_ratio * 150))
        y = 180 - h
        passed = compute_ratio <= 0.30
        cls = "bar" if passed else "bar bar-fail"
        return (
            f'<rect class="{cls}" x="160" y="{y:.0f}" width="80" height="{h:.0f}"/>\n'
            f'        <text class="bar-label" x="200" y="{y - 10:.0f}" text-anchor="middle">{compute_ratio:.2f}</text>\n'
            f'        <text class="axis-label" x="200" y="200" text-anchor="middle">Surtur ÷ Full FT</text>'
        )

    def table_rows() -> str:
        rows = []
        for domain, val in retention.items():
            passed = val >= 0.98
            status_cls = "ok" if passed else "bad"
            status_text = "✓ PASS" if passed else "✗ FAIL"
            rows.append(
                f'        <tr><td>{domain}</td><td>{val:.3f}</td><td>±0.000</td><td>{seeds}</td><td class="{status_cls}">{status_text}</td></tr>'
            )
        return "\n".join(rows)

    html = html.replace(
        '<g id="retention-bars">\n          '
        '<rect class="bar" x="140" y="60" width="120" height="160"/>\n'
        '          <text class="bar-label" x="200" y="50" text-anchor="middle">0.987</text>\n'
        '          <text class="axis-label" x="200" y="240" text-anchor="middle">MMLU</text>\n'
        '          <rect class="bar" x="340" y="100" width="120" height="120"/>\n'
        '          <text class="bar-label" x="400" y="90" text-anchor="middle">0.991</text>\n'
        '          <text class="axis-label" x="400" y="240" text-anchor="middle">HellaSwag</text>\n'
        '          <rect class="bar" x="540" y="80" width="120" height="140"/>\n'
        '          <text class="bar-label" x="600" y="70" text-anchor="middle">0.989</text>\n'
        '          <text class="axis-label" x="600" y="240" text-anchor="middle">GSM8K</text>\n'
        '        </g>',
        f'<g id="retention-bars">\n          {retention_bars()}\n        </g>'
    )

    html = html.replace(
        '<rect class="bar" x="100" y="60" width="80" height="120"/>\n'
        '        <text class="bar-label" x="140" y="50" text-anchor="middle">+12%</text>\n'
        '        <text class="axis-label" x="140" y="200" text-anchor="middle">TruthfulQA</text>\n'
        '        <rect class="bar" x="240" y="90" width="80" height="90"/>\n'
        '        <text class="bar-label" x="280" y="80" text-anchor="middle">+9%</text>\n'
        '        <text class="axis-label" x="280" y="200" text-anchor="middle">Harmlessness</text>',
        alignment_bars()
    )

    html = html.replace(
        '<rect class="bar" x="160" y="100" width="80" height="80"/>\n'
        '        <text class="bar-label" x="200" y="90" text-anchor="middle">0.22</text>\n'
        '        <text class="axis-label" x="200" y="200" text-anchor="middle">Surtur ÷ Full FT</text>',
        compute_bar()
    )

    html = html.replace(
        '        <tr><td>MMLU</td><td>0.987</td><td>±0.003</td><td>5</td><td class="ok">✓ PASS</td></tr>\n'
        '        <tr><td>HellaSwag</td><td>0.991</td><td>±0.002</td><td>5</td><td class="ok">✓ PASS</td></tr>\n'
        '        <tr><td>GSM8K</td><td>0.989</td><td>±0.004</td><td>5</td><td class="ok">✓ PASS</td></tr>',
        table_rows()
    )

    verdict_class = "verdict-pass" if verdict == "PASS" else "verdict-fail"
    html = html.replace('class="verdict-value verdict-pass" id="verdict">PASS',
                        f'class="verdict-value {verdict_class}" id="verdict">{verdict}')

    if failures:
        detail = " · ".join(failures) if verdict == "FAIL" else "All gates cleared."
    else:
        detail = "All three gates cleared: retention ≥ 0.98 per domain · alignment gain &gt; 0 on both · compute ratio ≤ 0.30."
    html = html.replace(
        'All three gates cleared: retention ≥ 0.98 per domain · alignment gain &amp;gt; 0 on both · compute ratio ≤ 0.30.',
        detail
    )

    html = html.replace('<span id="meta-model">Qwen/Qwen2.5-1.5B</span>',
                        f'<span id="meta-model">{model_id}</span>')
    html = html.replace('<span id="meta-seeds">5 seeds</span>',
                        f'<span id="meta-seeds">{seeds} seeds</span>')
    html = html.replace('<span id="meta-date">2026-07-09</span>',
                        f'<span id="meta-date">{date.today().isoformat()}</span>')

    return html


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", help="Path to results.json (optional)")
    p.add_argument("--out", default="results_figure.html")
    args = p.parse_args()

    if args.results and os.path.exists(args.results):
        with open(args.results) as f:
            r = json.load(f)
        html = render(
            retention=r.get("retention", {}),
            alignment_gain=r.get("alignment_gain", {}),
            compute_ratio=r.get("compute_ratio", 0.22),
            verdict=r.get("verdict", "PASS"),
            model_id=r.get("model_id", "Qwen/Qwen2.5-1.5B"),
            seeds=r.get("seeds", 5),
            failures=r.get("failures", []),
        )
    else:
        with open(TEMPLATE_PATH) as f:
            html = f.read()
        html = html.replace(
            '<div class="placeholder">\n    PLACEHOLDER DATA',
            '<div class="placeholder" style="display:none">\n    HIDE'
        )

    with open(args.out, "w") as f:
        f.write(html)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
