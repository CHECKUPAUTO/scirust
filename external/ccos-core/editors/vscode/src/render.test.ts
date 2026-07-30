// Pure render tests — no `vscode` dependency. Run: `npm run compile && node --test out/`.
// Verified live against real `ccos focus --json` output (see editors/README.md).
import assert from "node:assert";
import test from "node:test";
import { orderEntries, renderShieldHtml, FocusPayload } from "./render";

const sample: FocusPayload = {
  message: "thread 'tests::b' panicked at src/writer.rs:3",
  symptom_files: ["src/writer.rs"],
  workspace_files: 3,
  tokens: 100,
  entries: [
    { file: "src/writer.rs", role: "symptom", score: 0.72, content: "pub fn render() -> u8 { 0 }" },
    { file: "src/config.rs", role: "cause", score: 0.66, content: "pub fn buf() -> Vec<u8> { vec![] }" },
  ],
};

test("the likely cause is ordered first", () => {
  assert.equal(orderEntries(sample.entries)[0].role, "cause");
  assert.equal(orderEntries(sample.entries)[0].file, "src/config.rs");
});

test("html surfaces the cause, escapes content, and wires click-to-open", () => {
  const html = renderShieldHtml(sample, "vscode-resource:", "n0");
  assert.ok(html.includes('data-file="src/config.rs"'), "cause is a clickable file link");
  assert.ok(html.includes("nonce-n0") && html.includes("Content-Security-Policy"), "CSP + nonce");
  assert.ok(html.includes("Vec&lt;u8&gt;") && !html.includes("Vec<u8>"), "angle brackets escaped");
});
