import os
import subprocess
import time
from pathlib import Path

html_content = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Architecture review — subject-frame-extractor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script type="module">
      import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
      mermaid.initialize({ startOnLoad: true, theme: "neutral", securityLevel: "loose" });
    </script>
    <style>
      .seam { stroke-dasharray: 4 4; }
      .leak { stroke: #dc2626; }
      .deep { background: linear-gradient(135deg, #0f172a, #1e293b); }
    </style>
  </head>
  <body class="bg-stone-50 text-slate-900 font-sans">
    <main class="max-w-5xl mx-auto px-6 py-12 space-y-12">
      <header class="border-b border-slate-200 pb-6">
        <h1 class="text-3xl font-serif font-bold text-slate-900">Architecture Review: subject-frame-extractor</h1>
        <p class="text-sm text-slate-500 mt-2">Date: June 21, 2026</p>
        <div class="mt-4 flex flex-wrap gap-4 text-xs font-mono">
          <div class="flex items-center gap-1.5">
            <span class="inline-block w-4 h-4 bg-white border border-slate-300 rounded"></span>
            <span>Module / File</span>
          </div>
          <div class="flex items-center gap-1.5">
            <span class="inline-block w-4 h-4 bg-white border-2 border-dashed border-slate-400 rounded"></span>
            <span>Seam</span>
          </div>
          <div class="flex items-center gap-1.5">
            <span class="inline-block w-4 h-4 bg-red-600 rounded"></span>
            <span class="text-red-600">Leakage</span>
          </div>
          <div class="flex items-center gap-1.5">
            <span class="inline-block w-4 h-4 deep rounded"></span>
            <span>Deep Module</span>
          </div>
        </div>
      </header>

      <section id="candidates" class="space-y-10">
        <!-- CANDIDATE 1 -->
        <article class="p-6 bg-white rounded-xl border border-slate-200 shadow-sm space-y-6">
          <div class="flex items-start justify-between">
            <div>
              <h2 class="text-xl font-serif font-semibold">1. Collapse the Tracker Selection Seam</h2>
              <p class="text-xs font-mono text-slate-500 mt-1">
                core/managers/sam2.py, core/managers/tracker_factory.py, core/managers/registry.py, core/managers/__init__.py
              </p>
            </div>
            <div class="flex gap-2">
              <span class="px-2.5 py-1 text-xs font-semibold rounded bg-emerald-100 text-emerald-800">Strong</span>
              <span class="px-2.5 py-1 text-xs font-semibold rounded bg-slate-100 text-slate-700">in-process</span>
            </div>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="rounded-lg border border-slate-200 bg-white p-4">
              <div class="text-xs uppercase tracking-wider text-slate-400 mb-2 font-semibold">Before</div>
              <pre class="mermaid">
                flowchart TD
                    Registry[ModelRegistry] --> TF[tracker_factory.py]
                    TF -.seam.-> SAM2[sam2.py <br/> retired - raises ValueError]
                    TF -.seam.-> SAM3[sam3.py <br/> active wrapper]
                    classDef default fill:#fff,stroke:#64748b,stroke-width:1px;
                    classDef active fill:#e2e8f0,stroke:#334155,stroke-width:2px;
                    classDef retired fill:#fee2e2,stroke:#ef4444,stroke-width:1px;
                    class SAM3 active;
                    class SAM2 retired;
              </pre>
            </div>
            <div class="rounded-lg border border-slate-200 bg-white p-4">
              <div class="text-xs uppercase tracking-wider text-slate-400 mb-2 font-semibold">After</div>
              <pre class="mermaid">
                flowchart TD
                    Registry[ModelRegistry] -.seam.-> SAM3[sam3.py <br/> active wrapper]
                    classDef default fill:#fff,stroke:#64748b,stroke-width:1px;
                    classDef active fill:#0f172a,stroke:#0f172a,stroke-width:2px,color:#fff;
                    class SAM3 active;
              </pre>
            </div>
          </div>

          <div class="space-y-3">
            <p><strong>Problem:</strong> Tracker selection seam is shallow; the factory is a pass-through because the SAM2.1 module is retired and disabled.</p>
            <p><strong>Solution:</strong> Delete the obsolete `sam2.py` and `tracker_factory.py` modules, and initialize `SAM3Wrapper` directly inside `ModelRegistry`.</p>
            <div>
              <strong class="text-sm text-slate-600 block mb-1">Wins:</strong>
              <ul class="list-disc list-inside text-sm text-slate-700 space-y-1">
                <li>Locality: tracking code concentrates in one module</li>
                <li>Delete 2 shallow adapters</li>
                <li>Shrink the interface surface area</li>
              </ul>
            </div>
          </div>
        </article>

        <!-- CANDIDATE 2 -->
        <article class="p-6 bg-white rounded-xl border border-slate-200 shadow-sm space-y-6">
          <div class="flex items-start justify-between">
            <div>
              <h2 class="text-xl font-serif font-semibold">2. Deepen Media & Session Management</h2>
              <p class="text-xs font-mono text-slate-500 mt-1">
                core/managers/video.py, core/managers/session.py, core/pipelines.py
              </p>
            </div>
            <div class="flex gap-2">
              <span class="px-2.5 py-1 text-xs font-semibold rounded bg-emerald-100 text-emerald-800">Strong</span>
              <span class="px-2.5 py-1 text-xs font-semibold rounded bg-slate-100 text-slate-700">local-substitutable</span>
            </div>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="rounded-lg border border-slate-200 bg-white p-4">
              <div class="text-xs uppercase tracking-wider text-slate-400 mb-2 font-semibold">Before</div>
              <pre class="mermaid">
                flowchart TD
                    pipelines.py --> VM[VideoManager <br/> download / metadata]
                    pipelines.py --> Session[session.py <br/> validate / load_session]
                    pipelines.py --> info[get_video_info]
                    classDef default fill:#fff,stroke:#64748b,stroke-width:1px;
              </pre>
            </div>
            <div class="rounded-lg border border-slate-200 bg-white p-4">
              <div class="text-xs uppercase tracking-wider text-slate-400 mb-2 font-semibold">After</div>
              <pre class="mermaid">
                flowchart TD
                    pipelines.py -.seam.-> MediaSession[MediaSession Module]
                    subgraph MediaSession [MediaSession Module]
                        video[Video ingestion]
                        session[Session validation/loading]
                        info[Metadata lookup]
                    end
                    classDef default fill:#fff,stroke:#64748b,stroke-width:1px;
                    classDef deep fill:#0f172a,stroke:#0f172a,stroke-width:2px,color:#fff;
                    class MediaSession deep;
              </pre>
            </div>
          </div>

          <div class="space-y-3">
            <p><strong>Problem:</strong> Media ingestion and session loading are split across shallow, procedural helper files, forcing the pipelines module to coordinate state.</p>
            <p><strong>Solution:</strong> Consolidate video validation, YouTube downloads, and scene ingestion into a single deep `MediaSession` module.</p>
            <div>
              <strong class="text-sm text-slate-600 block mb-1">Wins:</strong>
              <ul class="list-disc list-inside text-sm text-slate-700 space-y-1">
                <li>Locality: media lifecycle concentrates in one module</li>
                <li>Interface shrinks: single entry point for pipeline</li>
                <li>Leverage: pipeline delegates all prep logic</li>
              </ul>
            </div>
          </div>
        </article>

        <!-- CANDIDATE 3 -->
        <article class="p-6 bg-white rounded-xl border border-slate-200 shadow-sm space-y-6">
          <div class="flex items-start justify-between">
            <div>
              <h2 class="text-xl font-serif font-semibold">3. Deepen Operator Filtering & Discovery</h2>
              <p class="text-xs font-mono text-slate-500 mt-1">
                core/operators/registry.py, core/filtering.py
              </p>
            </div>
            <div class="flex gap-2">
              <span class="px-2.5 py-1 text-xs font-semibold rounded bg-amber-100 text-amber-800">Worth exploring</span>
              <span class="px-2.5 py-1 text-xs font-semibold rounded bg-slate-100 text-slate-700">in-process</span>
            </div>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="rounded-lg border border-slate-200 bg-white p-4">
              <div class="text-xs uppercase tracking-wider text-slate-400 mb-2 font-semibold">Before</div>
              <pre class="mermaid">
                flowchart TD
                    filtering.py -.leak.-> registry.py[run_operators]
                    filtering.py --> specific_keys[sharpness, contrast, yaw...]
                    classDef default fill:#fff,stroke:#64748b,stroke-width:1px;
                    classDef leak stroke:#dc2626,stroke-width:2px;
                    class specific_keys leak;
              </pre>
            </div>
            <div class="rounded-lg border border-slate-200 bg-white p-4">
              <div class="text-xs uppercase tracking-wider text-slate-400 mb-2 font-semibold">After</div>
              <pre class="mermaid">
                flowchart TD
                    filtering.py -.seam.-> OperatorRegistry[OperatorRegistry]
                    subgraph OperatorRegistry [OperatorRegistry]
                        ops[Operators run & filter themselves]
                    end
                    classDef default fill:#fff,stroke:#64748b,stroke-width:1px;
                    classDef deep fill:#0f172a,stroke:#0f172a,stroke-width:2px,color:#fff;
                    class OperatorRegistry deep;
              </pre>
            </div>
          </div>

          <div class="space-y-3">
            <p><strong>Problem:</strong> Filtering logic is highly coupled to specific operator metrics, causing metrics leakage across the filtering seam.</p>
            <p><strong>Solution:</strong> Deepen the `Operator` interface so each operator defines its own filter logic and default config, hiding details from the filter module.</p>
            <div>
              <strong class="text-sm text-slate-600 block mb-1">Wins:</strong>
              <ul class="list-disc list-inside text-sm text-slate-700 space-y-1">
                <li>Locality: metric filtering details stay within operators</li>
                <li>Leverage: registry runs and filters automatically</li>
                <li>Interface shrinks: less configuration coordination</li>
              </ul>
            </div>
          </div>
        </article>
      </section>

      <section id="top-recommendation" class="p-6 bg-slate-900 text-white rounded-xl border border-slate-800 shadow-lg space-y-4">
        <h2 class="text-2xl font-serif font-bold text-white">Top Recommendation</h2>
        <div class="text-emerald-400 font-semibold uppercase tracking-wider text-xs">Recommended First Step</div>
        <h3 class="text-lg font-semibold text-slate-200">1. Collapse the Tracker Selection Seam</h3>
        <p class="text-slate-400 text-sm">
          Retiring the obsolete and disabled SAM2.1 tracker allows us to delete two redundant, shallow files (<code class="text-emerald-300 font-mono">sam2.py</code> and <code class="text-emerald-300 font-mono">tracker_factory.py</code>). This directly collapses a shallow indirection layer, improves codebase leverage, and ensures all tracking flows go directly to the SAM3.1 wrapper.
        </p>
        <div class="pt-2">
          <a href="#candidates" class="inline-block text-emerald-400 hover:text-emerald-300 font-semibold text-sm underline">
            View details above &rarr;
          </a>
        </div>
      </section>
    </main>
  </body>
</html>
"""

timestamp = int(time.time())
temp_dir = os.environ.get("TMPDIR", "/tmp")
file_path = Path(temp_dir) / f"architecture-review-{timestamp}.html"
file_path.write_text(html_content, encoding="utf-8")

print(f"HTML report written to: {file_path}")
subprocess.run(["xdg-open", str(file_path)])
