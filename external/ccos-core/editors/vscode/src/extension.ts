// CCOS Attentional Shield — VS Code thin client.
//
// All intelligence lives in the `ccos focus` binary; this extension only:
//   run the test command → pipe its failure to `ccos focus --json` → render the
//   causal region (likely cause first), and open the file when a link is clicked.

import * as vscode from "vscode";
import * as cp from "child_process";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { renderShieldHtml, FocusPayload } from "./render";

let panel: vscode.WebviewPanel | undefined;

export function activate(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("ccos.focus", () => runFocus(context))
  );
}

function crateRoot(): string | undefined {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) {
    return undefined;
  }
  // Prefer the folder of the active file's nearest Cargo.toml, else the first folder.
  const active = vscode.window.activeTextEditor?.document.uri.fsPath;
  if (active) {
    let dir = path.dirname(active);
    while (dir && dir !== path.dirname(dir)) {
      if (fs.existsSync(path.join(dir, "Cargo.toml"))) {
        return dir;
      }
      dir = path.dirname(dir);
    }
  }
  return folders[0].uri.fsPath;
}

function cfg<T>(key: string, fallback: T): T {
  return vscode.workspace.getConfiguration("ccos").get<T>(key, fallback);
}

async function runFocus(context: vscode.ExtensionContext): Promise<void> {
  const root = crateRoot();
  if (!root) {
    vscode.window.showWarningMessage("CCOS: open a Rust workspace first.");
    return;
  }
  const testCmd = cfg("testCommand", "cargo test");

  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Window, title: "CCOS: running tests…" },
    () =>
      new Promise<void>((resolve) => {
        // cargo test exits non-zero on failure — that is exactly our input.
        cp.exec(testCmd, { cwd: root, maxBuffer: 16 * 1024 * 1024 }, (_err, stdout, stderr) => {
          const trace = path.join(os.tmpdir(), `ccos-trace-${Date.now()}.txt`);
          fs.writeFileSync(trace, (stdout || "") + "\n" + (stderr || ""));
          focus(context, root, trace).finally(resolve);
        });
      })
  );
}

function focus(context: vscode.ExtensionContext, root: string, trace: string): Promise<void> {
  const bin = cfg("binary", "ccos");
  const args = [
    "focus",
    cfg("src", "src"),
    "--input",
    trace,
    "--budget",
    String(cfg("budget", 2048)),
    "--workspace",
    cfg("workspace", ".ccos/ws.ccos"),
    "--json",
  ];
  return new Promise<void>((resolve) => {
    cp.execFile(bin, args, { cwd: root, maxBuffer: 16 * 1024 * 1024 }, (err, stdout) => {
      if (err && !stdout) {
        vscode.window.showErrorMessage(`CCOS focus failed: ${err.message}`);
        return resolve();
      }
      let payload: FocusPayload;
      try {
        payload = JSON.parse(stdout);
      } catch {
        vscode.window.showErrorMessage("CCOS: could not parse focus output.");
        return resolve();
      }
      show(context, root, payload);
      resolve();
    });
  });
}

function show(context: vscode.ExtensionContext, root: string, payload: FocusPayload): void {
  if (!panel) {
    panel = vscode.window.createWebviewPanel(
      "ccosShield",
      "CCOS Attentional Shield",
      { viewColumn: vscode.ViewColumn.Beside, preserveFocus: true },
      { enableScripts: true, retainContextWhenHidden: true }
    );
    panel.onDidDispose(() => (panel = undefined), null, context.subscriptions);
    panel.webview.onDidReceiveMessage(
      (m: { command: string; file?: string }) => {
        if (m.command === "open" && m.file) {
          const uri = vscode.Uri.file(path.join(root, m.file));
          vscode.window.showTextDocument(uri, { viewColumn: vscode.ViewColumn.One });
        }
      },
      undefined,
      context.subscriptions
    );
  }
  const nonce = Math.random().toString(36).slice(2);
  panel.webview.html = renderShieldHtml(payload, panel.webview.cspSource, nonce);
  panel.reveal(vscode.ViewColumn.Beside, true);
}

export function deactivate(): void {
  /* nothing to clean up */
}
