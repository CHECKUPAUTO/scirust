// Pure rendering for the CCOS attentional shield — no `vscode` dependency, so it
// can be unit-tested against real `ccos focus --json` output (see render.test.js).

export interface FocusEntry {
  file: string;
  role: string; // "cause" | "symptom" | "context"
  score: number;
  content: string;
}

export interface FocusPayload {
  message: string;
  symptom_files: string[];
  workspace_files: number;
  reparsed_files?: number;
  tokens: number;
  entries: FocusEntry[];
}

const ROLE_RANK: Record<string, number> = { cause: 1, symptom: 2, context: 3 };
const ROLE_TAG: Record<string, string> = {
  cause: "◀ likely cause",
  symptom: "· symptom",
  context: "· related",
};

/** Cause first, then symptom, then the rest — the "skip to the root" ordering. */
export function orderEntries(entries: FocusEntry[]): FocusEntry[] {
  return [...(entries || [])].sort(
    (a, b) => (ROLE_RANK[a.role] ?? 9) - (ROLE_RANK[b.role] ?? 9)
  );
}

function esc(s: string): string {
  return String(s).replace(
    /[&<>"]/g,
    (c) => (({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" } as Record<string, string>)[c])
  );
}

/** Build the webview HTML. `csp`/`nonce` come from the host so inline JS is allowed. */
export function renderShieldHtml(p: FocusPayload, csp: string = "", nonce: string = ""): string {
  const ordered = orderEntries(p.entries);
  const header = `⚡ ${p.workspace_files ?? 0} files → ${ordered.length} in view · ~${p.tokens ?? 0} tokens`;
  const items = ordered
    .map((e) => {
      const cls = e.role === "cause" ? "entry cause" : "entry";
      const body = esc((e.content || "").split("\n").slice(0, 8).join("\n"));
      return `<div class="${cls}">
  <a class="file" data-file="${esc(e.file)}" href="#">${esc(e.file)}</a>
  <span class="tag">${esc(ROLE_TAG[e.role] || "")}</span>
  <pre>${body}</pre>
</div>`;
    })
    .join("\n");
  const msg = p.message ? `<p class="msg">${esc(p.message.replace(/\s+/g, " ").slice(0, 120))}</p>` : "";
  const cspMeta = csp
    ? `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${csp} 'unsafe-inline'; script-src 'nonce-${nonce}';">`
    : "";

  return `<!DOCTYPE html><html><head><meta charset="utf-8">${cspMeta}
<style>
  body { font-family: var(--vscode-editor-font-family, monospace); padding: 8px; color: var(--vscode-foreground); }
  h2 { font-size: 0.95em; font-weight: 600; margin: 0 0 4px; }
  .msg { opacity: 0.8; margin: 0 0 10px; }
  .entry { border-left: 2px solid var(--vscode-panel-border); padding: 2px 8px; margin: 8px 0; }
  .entry.cause { border-left-color: var(--vscode-editorError-foreground, #e51400); }
  .file { font-weight: 600; text-decoration: none; color: var(--vscode-textLink-foreground); }
  .tag { opacity: 0.7; margin-left: 8px; font-size: 0.85em; }
  pre { margin: 4px 0 0; white-space: pre-wrap; opacity: 0.9; font-size: 0.9em; }
</style></head><body>
<h2>${esc(header)}</h2>
${msg}
${items || "<p>No project source in the failure (all green?).</p>"}
<script nonce="${nonce}">
  const vscode = acquireVsCodeApi();
  for (const a of document.querySelectorAll('.file')) {
    a.addEventListener('click', (e) => {
      e.preventDefault();
      vscode.postMessage({ command: 'open', file: a.getAttribute('data-file') });
    });
  }
</script>
</body></html>`;
}
