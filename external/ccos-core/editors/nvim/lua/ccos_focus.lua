-- ccos_focus.lua — the CCOS attentional shield for Neovim.
--
-- On demand: run `cargo test`, feed the failure to `ccos focus --json`, and float
-- a minimal window showing the *likely root cause* first, then the symptom, then
-- the rest of the causal region — the backtrace noise and unrelated files hidden.
-- All the intelligence is in the `ccos` binary; this client only captures + renders.
--
-- Install (lazy.nvim):  { dir = "…/CCOS/editors/nvim", config = function() require("ccos_focus").setup() end }
-- Use:                  :CcosFocus   (or map it, see setup)

local M = {}

M.opts = {
  ccos = "ccos", -- path to the binary
  src = "src", -- source dir, relative to the crate root
  workspace = ".ccos/ws.ccos", -- persisted checkpoint → O(Δ) re-parse
  budget = 2048,
  test_cmd = { "cargo", "test" }, -- the command whose failure we focus
  keymap = "<leader>cf", -- set to false to skip the default mapping
}

-- Nearest ancestor of `start` that holds a Cargo.toml (the crate root).
local function crate_root(start)
  local dir = vim.fn.fnamemodify(start or vim.api.nvim_buf_get_name(0), ":p:h")
  while dir and dir ~= "/" do
    if vim.fn.filereadable(dir .. "/Cargo.toml") == 1 then
      return dir
    end
    dir = vim.fn.fnamemodify(dir, ":h")
  end
  return vim.fn.getcwd()
end

-- Float the focused view; <CR> opens the file on the current line, q/<Esc> closes.
local function render(payload)
  local entries = payload.entries or {}
  if #entries == 0 then
    vim.notify("CCOS: no project source in the failure (all green?)", vim.log.levels.INFO)
    return
  end
  -- Cause first, then symptom, then related — the "skip to the root" ordering.
  local rank = { cause = 1, symptom = 2, context = 3 }
  table.sort(entries, function(a, b)
    return (rank[a.role] or 9) < (rank[b.role] or 9)
  end)

  local lines, targets = {}, {}
  table.insert(lines, ("⚡ CCOS focus — %d files → %d in view (~%d tokens)")
    :format(payload.workspace_files or 0, #entries, payload.tokens or 0))
  if payload.message and payload.message ~= "" then
    table.insert(lines, "  " .. payload.message:gsub("%s+", " "):sub(1, 76))
  end
  table.insert(lines, "")
  local tag = { cause = "◀ likely cause", symptom = "· symptom", context = "· related" }
  for _, e in ipairs(entries) do
    targets[#lines + 1] = e.file
    table.insert(lines, ("  %s   %s"):format(e.file, tag[e.role] or ""))
    local n = 0
    for snippet in (e.content or ""):gmatch("[^\n]+") do
      if n >= 4 then
        table.insert(lines, "      …")
        break
      end
      table.insert(lines, "      " .. snippet)
      n = n + 1
    end
    table.insert(lines, "")
  end

  local buf = vim.api.nvim_create_buf(false, true)
  vim.api.nvim_buf_set_lines(buf, 0, -1, false, lines)
  vim.bo[buf].modifiable = false
  vim.bo[buf].filetype = "rust"
  local width = math.min(90, vim.o.columns - 4)
  local height = math.min(#lines + 1, vim.o.lines - 6)
  local win = vim.api.nvim_open_win(buf, true, {
    relative = "editor",
    width = width,
    height = height,
    row = 2,
    col = (vim.o.columns - width) / 2,
    style = "minimal",
    border = "rounded",
    title = " CCOS Attentional Shield ",
  })

  local root = payload._root
  local function open_under_cursor()
    local file = targets[vim.api.nvim_win_get_cursor(win)[1]]
    if file then
      vim.api.nvim_win_close(win, true)
      vim.cmd("edit " .. vim.fn.fnameescape(root .. "/" .. file))
    end
  end
  vim.keymap.set("n", "<CR>", open_under_cursor, { buffer = buf, nowait = true })
  for _, k in ipairs({ "q", "<Esc>" }) do
    vim.keymap.set("n", k, function() vim.api.nvim_win_close(win, true) end, { buffer = buf, nowait = true })
  end
end

-- Run `ccos focus` over a captured trace and render the result.
local function focus(root, trace_file)
  local o = M.opts
  local cmd = {
    o.ccos, "focus", o.src,
    "--input", trace_file,
    "--budget", tostring(o.budget),
    "--workspace", o.workspace,
    "--json",
  }
  local out = {}
  vim.fn.jobstart(cmd, {
    cwd = root,
    stdout_buffered = true,
    on_stdout = function(_, data) vim.list_extend(out, data or {}) end,
    on_exit = function(_, code)
      if code ~= 0 then
        vim.notify("CCOS focus failed (exit " .. code .. ")", vim.log.levels.ERROR)
        return
      end
      local ok, payload = pcall(vim.fn.json_decode, table.concat(out, "\n"))
      if not ok then
        vim.notify("CCOS: could not parse focus output", vim.log.levels.ERROR)
        return
      end
      payload._root = root
      vim.schedule(function() render(payload) end)
    end,
  })
end

-- Run the test command, capture its output, then focus the failure.
function M.run()
  local root = crate_root()
  local out = {}
  vim.notify("CCOS: running tests…", vim.log.levels.INFO)
  vim.fn.jobstart(M.opts.test_cmd, {
    cwd = root,
    stdout_buffered = true,
    stderr_buffered = true,
    on_stdout = function(_, d) vim.list_extend(out, d or {}) end,
    on_stderr = function(_, d) vim.list_extend(out, d or {}) end,
    on_exit = function()
      local trace = vim.fn.tempname()
      vim.fn.writefile(out, trace)
      focus(root, trace)
    end,
  })
end

function M.setup(opts)
  M.opts = vim.tbl_extend("force", M.opts, opts or {})
  vim.api.nvim_create_user_command("CcosFocus", M.run, { desc = "CCOS attentional shield" })
  if M.opts.keymap then
    vim.keymap.set("n", M.opts.keymap, M.run, { desc = "CCOS Focus Shield" })
  end
end

-- Exposed for a headless render smoke-test (see editors/README.md): float a given
-- payload directly, bypassing the async test run.
M.render = render

return M
