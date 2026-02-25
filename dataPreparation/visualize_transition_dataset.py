#!/usr/bin/env python3
"""
Create an interactive HTML viewer for transition-dataset JSONL files.

The viewer supports search, filtering, pagination, and readable rendering
of code context, action, current state, and next state.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple


def load_jsonl(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    examples: List[Dict[str, Any]] = []
    errors: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: {exc}")
                continue

            record["__row_id"] = len(examples)
            examples.append(record)

    return examples, errors


def build_html(examples: List[Dict[str, Any]], input_path: str, title: str) -> str:
    task_ids = sorted({str(ex.get("task_id", "")) for ex in examples if ex.get("task_id")})
    entry_points = sorted(
        {str(ex.get("entry_point", "")) for ex in examples if ex.get("entry_point")}
    )
    next_event_types = sorted(
        {str(ex.get("next_event_type", "")) for ex in examples if ex.get("next_event_type")}
    )

    payload = {
        "examples": examples,
        "meta": {
            "input_path": input_path,
            "count": len(examples),
            "task_ids": task_ids,
            "entry_points": entry_points,
            "next_event_types": next_event_types,
        },
    }

    data_json = json.dumps(payload, ensure_ascii=False)
    page_title = json.dumps(title)

    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>{title}</title>
<style>
:root {{
  --bg: #f4f6f8;
  --panel: #ffffff;
  --ink: #17212b;
  --muted: #51616f;
  --line: #d9e0e6;
  --accent: #0b7a75;
  --accent-ink: #ffffff;
  --code-bg: #0f1720;
  --code-ink: #e7edf3;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  color: var(--ink);
  background: linear-gradient(160deg, #eaf1f7, var(--bg));
}}
header {{
  position: sticky;
  top: 0;
  z-index: 10;
  background: rgba(255,255,255,0.94);
  backdrop-filter: blur(6px);
  border-bottom: 1px solid var(--line);
  padding: 14px 18px;
}}
.header-row {{
  display: flex;
  gap: 14px;
  align-items: center;
  flex-wrap: wrap;
}}
h1 {{
  margin: 0;
  font-size: 1.1rem;
  letter-spacing: 0.01em;
}}
.meta {{ color: var(--muted); font-size: 0.9rem; }}
.controls {{
  display: grid;
  grid-template-columns: repeat(6, minmax(140px, 1fr));
  gap: 10px;
  margin-top: 10px;
}}
label {{
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 0.8rem;
  color: var(--muted);
}}
input, select, button {{
  font: inherit;
  padding: 8px 10px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #fff;
}}
button {{
  cursor: pointer;
  border: none;
  background: var(--accent);
  color: var(--accent-ink);
}}
button.secondary {{
  background: #dde5eb;
  color: var(--ink);
}}
#status {{
  margin-top: 8px;
  font-size: 0.9rem;
  color: var(--muted);
}}
main {{ padding: 16px; }}
.card {{
  border: 1px solid var(--line);
  border-radius: 14px;
  background: var(--panel);
  padding: 12px;
  margin-bottom: 12px;
  box-shadow: 0 4px 16px rgba(15, 23, 32, 0.05);
}}
.card-top {{
  display: flex;
  justify-content: space-between;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 8px;
}}
.badges {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.badge {{
  border-radius: 999px;
  background: #eef4f9;
  color: #234;
  font-size: 0.75rem;
  padding: 4px 8px;
}}
details {{
  margin-top: 8px;
  border: 1px solid #e6edf2;
  border-radius: 8px;
  overflow: hidden;
}}
summary {{
  cursor: pointer;
  list-style: none;
  padding: 8px 10px;
  background: #f6f9fb;
  border-bottom: 1px solid #e6edf2;
  font-weight: 600;
}}
pre {{
  margin: 0;
  padding: 10px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: "JetBrains Mono", "Consolas", monospace;
  font-size: 0.84rem;
  line-height: 1.45;
  background: var(--code-bg);
  color: var(--code-ink);
}}
.footer-controls {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin: 16px 0;
  flex-wrap: wrap;
}}
@media (max-width: 1100px) {{
  .controls {{ grid-template-columns: repeat(2, minmax(140px, 1fr)); }}
}}
@media (max-width: 700px) {{
  .controls {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<header>
  <div class=\"header-row\">
    <h1 id=\"title\"></h1>
    <div class=\"meta\" id=\"meta\"></div>
  </div>
  <div class=\"controls\">
    <label>
      Search text
      <input id=\"search\" type=\"text\" placeholder=\"example_id, code, states...\" />
    </label>
    <label>
      Task ID
      <select id=\"task\"><option value=\"\">All</option></select>
    </label>
    <label>
      Entry Point
      <select id=\"entry\"><option value=\"\">All</option></select>
    </label>
    <label>
      Next Event
      <select id=\"event\"><option value=\"\">All</option></select>
    </label>
    <label>
      Page Size
      <select id=\"pageSize\">
        <option>10</option>
        <option selected>20</option>
        <option>50</option>
        <option>100</option>
      </select>
    </label>
    <label>
      Quick jump
      <input id=\"jump\" type=\"number\" min=\"1\" step=\"1\" placeholder=\"example #\" />
    </label>
  </div>
  <div class=\"header-row\" style=\"margin-top:10px\">
    <button id=\"apply\">Apply Filters</button>
    <button id=\"reset\" class=\"secondary\">Reset</button>
    <button id=\"prev\" class=\"secondary\">Prev Page</button>
    <button id=\"next\" class=\"secondary\">Next Page</button>
    <span id=\"status\"></span>
  </div>
</header>
<main>
  <div id=\"results\"></div>
  <div class=\"footer-controls\">
    <div class=\"meta\" id=\"pageInfo\"></div>
    <div class=\"meta\">Shortcuts: <code>/</code> search, <code>n</code>/<code>p</code> page, <code>j</code> jump.</div>
  </div>
</main>
<script>
const INITIAL_TITLE = {page_title};
const DATA = {data_json};

const state = {{
  page: 1,
  pageSize: 20,
  search: "",
  task: "",
  entry: "",
  event: "",
}};

const els = {{
  title: document.getElementById("title"),
  meta: document.getElementById("meta"),
  search: document.getElementById("search"),
  task: document.getElementById("task"),
  entry: document.getElementById("entry"),
  event: document.getElementById("event"),
  pageSize: document.getElementById("pageSize"),
  jump: document.getElementById("jump"),
  apply: document.getElementById("apply"),
  reset: document.getElementById("reset"),
  prev: document.getElementById("prev"),
  next: document.getElementById("next"),
  status: document.getElementById("status"),
  results: document.getElementById("results"),
  pageInfo: document.getElementById("pageInfo"),
}};

function fillSelect(select, values) {{
  values.forEach(value => {{
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = value;
    select.appendChild(opt);
  }});
}}

function normalize(text) {{
  return String(text || "").toLowerCase();
}}

function prettyStateText(raw) {{
  if (!raw) return "";
  const lines = String(raw).split("\\n");
  const last = lines.length > 0 ? lines[lines.length - 1].trim() : "";
  const hasSummaryToken = last === "<STATE_SUM>" || last === "<CTX_SUM>" || last === "<ACT_SUM>";
  if (!hasSummaryToken || lines.length < 2) return String(raw);

  const payload = lines.slice(0, -1).join("\\n").trim();
  if (!payload.startsWith("{{") || !payload.endsWith("}}")) return String(raw);

  try {{
    const obj = JSON.parse(payload);
    return JSON.stringify(obj, null, 2) + "\\n" + last;
  }} catch (_err) {{
    return String(raw);
  }}
}}

function makeField(label, text, openByDefault = false) {{
  const details = document.createElement("details");
  details.open = openByDefault;

  const summary = document.createElement("summary");
  summary.textContent = label;

  const pre = document.createElement("pre");
  pre.textContent = prettyStateText(text || "");

  details.appendChild(summary);
  details.appendChild(pre);
  return details;
}}

function getFiltered() {{
  const q = normalize(state.search);

  return DATA.examples.filter(ex => {{
    if (state.task && String(ex.task_id || "") !== state.task) return false;
    if (state.entry && String(ex.entry_point || "") !== state.entry) return false;
    if (state.event && String(ex.next_event_type || "") !== state.event) return false;

    if (!q) return true;
    const searchable = [
      ex.example_id,
      ex.task_id,
      ex.entry_point,
      ex.next_event_type,
      ex.code_context,
      ex.action,
      ex.current_state,
      ex.next_state,
    ].map(normalize).join("\\n");

    return searchable.includes(q);
  }});
}}

function render() {{
  const filtered = getFiltered();
  const total = filtered.length;

  if (state.pageSize < 1) state.pageSize = 20;
  const totalPages = Math.max(1, Math.ceil(total / state.pageSize));
  if (state.page > totalPages) state.page = totalPages;
  if (state.page < 1) state.page = 1;

  const start = (state.page - 1) * state.pageSize;
  const end = Math.min(start + state.pageSize, total);
  const pageItems = filtered.slice(start, end);

  els.results.textContent = "";

  pageItems.forEach((ex, idx) => {{
    const card = document.createElement("section");
    card.className = "card";

    const top = document.createElement("div");
    top.className = "card-top";

    const left = document.createElement("div");
    const title = document.createElement("strong");
    const absoluteIndex = start + idx + 1;
    title.textContent = `#${{absoluteIndex}} ${{ex.example_id || "(no id)"}}`;
    left.appendChild(title);

    const badges = document.createElement("div");
    badges.className = "badges";

    const badgeValues = [
      `task: ${{ex.task_id || "n/a"}}`,
      `entry: ${{ex.entry_point || "n/a"}}`,
      `step: ${{ex.step_index ?? "n/a"}}`,
      `line: ${{ex.line_no ?? "n/a"}}`,
      `next: ${{ex.next_event_type || "n/a"}}`,
    ];

    badgeValues.forEach(text => {{
      const b = document.createElement("span");
      b.className = "badge";
      b.textContent = text;
      badges.appendChild(b);
    }});

    top.appendChild(left);
    top.appendChild(badges);
    card.appendChild(top);

    card.appendChild(makeField("Code Context", ex.code_context, false));
    card.appendChild(makeField("Action", ex.action, true));
    card.appendChild(makeField("Current State", ex.current_state, false));
    card.appendChild(makeField("Next State", ex.next_state, false));

    els.results.appendChild(card);
  }});

  els.status.textContent = `Showing ${{total === 0 ? 0 : start + 1}}-${{end}} of ${{total}} filtered examples`;
  els.pageInfo.textContent = `Page ${{state.page}} / ${{totalPages}}`;

  els.prev.disabled = state.page <= 1;
  els.next.disabled = state.page >= totalPages;
}}

function applyFromInputs(resetPage = true) {{
  state.search = els.search.value.trim();
  state.task = els.task.value;
  state.entry = els.entry.value;
  state.event = els.event.value;
  state.pageSize = parseInt(els.pageSize.value, 10) || 20;

  if (resetPage) state.page = 1;

  const jumpValue = parseInt(els.jump.value, 10);
  if (!Number.isNaN(jumpValue) && jumpValue > 0) {{
    state.page = Math.ceil(jumpValue / state.pageSize);
  }}

  render();
}}

function resetAll() {{
  els.search.value = "";
  els.task.value = "";
  els.entry.value = "";
  els.event.value = "";
  els.pageSize.value = "20";
  els.jump.value = "";
  state.page = 1;
  applyFromInputs(false);
}}

function init() {{
  document.title = INITIAL_TITLE;
  els.title.textContent = INITIAL_TITLE;
  els.meta.textContent = `${{DATA.meta.count}} examples from ${{DATA.meta.input_path}}`;

  fillSelect(els.task, DATA.meta.task_ids);
  fillSelect(els.entry, DATA.meta.entry_points);
  fillSelect(els.event, DATA.meta.next_event_types);

  els.apply.addEventListener("click", () => applyFromInputs(true));
  els.reset.addEventListener("click", resetAll);

  els.prev.addEventListener("click", () => {{
    state.page -= 1;
    render();
  }});

  els.next.addEventListener("click", () => {{
    state.page += 1;
    render();
  }});

  [els.search, els.task, els.entry, els.event, els.pageSize, els.jump].forEach(el => {{
    el.addEventListener("keydown", evt => {{
      if (evt.key === "Enter") applyFromInputs(true);
    }});
  }});

  document.addEventListener("keydown", evt => {{
    if (evt.target && ["INPUT", "TEXTAREA", "SELECT"].includes(evt.target.tagName)) {{
      return;
    }}

    if (evt.key === "n") {{
      state.page += 1;
      render();
    }} else if (evt.key === "p") {{
      state.page -= 1;
      render();
    }} else if (evt.key === "/") {{
      evt.preventDefault();
      els.search.focus();
    }} else if (evt.key === "j") {{
      evt.preventDefault();
      els.jump.focus();
    }}
  }});

  render();
}}

init();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML viewer for transition dataset JSONL.",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to transition dataset JSONL",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output HTML path (default: <input_basename>.viewer.html)",
    )
    parser.add_argument(
        "--title",
        default="Transition Dataset Viewer",
        help="HTML title/header",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".viewer.html"

    examples, errors = load_jsonl(input_path)
    html = build_html(examples, input_path, args.title)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Wrote viewer: {output_path}")
    print(f"Loaded examples: {len(examples)}")
    if errors:
        print(f"Skipped malformed lines: {len(errors)}")
        for err in errors[:5]:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
