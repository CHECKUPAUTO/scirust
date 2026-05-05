#!/usr/bin/env node
'use strict';

const { execSync, spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');
const https = require('https');

// ── Config ────────────────────────────────────────────────────────────────
const REPO_URL       = 'https://github.com/CHECKUPAUTO/scirust.git';
const REPO_API       = 'https://api.github.com/repos/CHECKUPAUTO/scirust';
const CARGO_BIN      = process.env.CARGO_BIN || 'cargo';
const INSTALL_DIR    = process.env.SCIRUST_HOME || path.join(os.homedir(), '.scirust');
const SRC_DIR        = path.join(INSTALL_DIR, 'src');
const BIN_DIR        = path.join(INSTALL_DIR, 'bin');
const BIN_PATH       = path.join(BIN_DIR, 'scirust');
const VERSION_FILE   = path.join(INSTALL_DIR, '.version');
const PROFILE_PATH   = path.join(os.homedir(), '.bashrc');
const ZSH_PROFILE    = path.join(os.homedir(), '.zshrc');
const FISH_PROFILE   = path.join(os.homedir(), '.config', 'fish', 'config.fish');

// ── Helpers ────────────────────────────────────────────────────────────────

function log(msg)   { console.log(`[scirust] ${msg}`); }
function warn(msg)  { console.warn(`[scirust] ⚠  ${msg}`); }
function error(msg) { console.error(`[scirust] ❌ ${msg}`); process.exit(1); }
function ok(msg)    { console.log(`[scirust] ✅ ${msg}`); }

function run(cmd, opts = {}) {
    log(`→ ${cmd}`);
    const r = spawnSync('bash', ['-c', cmd], { stdio: 'inherit', ...opts });
    if (r.status !== 0 && !opts.ignoreError) {
        error(`Command failed: ${cmd}`);
    }
    return r;
}

function runSilent(cmd) {
    return execSync(cmd, { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] }).trim();
}

function hasCommand(cmd) {
    try { execSync(`which ${cmd}`, { stdio: 'ignore' }); return true; } catch { return false; }
}

function httpGetJSON(url) {
    return new Promise((resolve, reject) => {
        https.get(url, { headers: { 'User-Agent': 'scirust-npm' } }, (res) => {
            let data = '';
            res.on('data', c => data += c);
            res.on('end', () => {
                try { resolve(JSON.parse(data)); } catch (e) { reject(e); }
            });
        }).on('error', reject);
    });
}

// ── Install ────────────────────────────────────────────────────────────────

async function install() {
    console.log('╔══════════════════════════════════════════════╗');
    console.log('║   SciRust Installer v0.13 — Pure Rust ML     ║');
    console.log('╚══════════════════════════════════════════════╝\n');

    // Prerequisites
    const missing = [];
    if (!hasCommand('cargo')) missing.push('cargo (Rust toolchain)');
    if (!hasCommand('git'))   missing.push('git');

    if (missing.length > 0) {
        error(`Missing prerequisites:\n  ${missing.join('\n  ')}\n\nInstall Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`);
    }

    log(`Install directory: ${INSTALL_DIR}`);

    // Create directories
    fs.mkdirSync(INSTALL_DIR, { recursive: true });
    fs.mkdirSync(BIN_DIR, { recursive: true });

    // Clone or pull repo
    if (fs.existsSync(SRC_DIR)) {
        log('Repository exists, pulling latest...');
        run(`cd "${SRC_DIR}" && git fetch origin && git checkout feat/portable-simd-and-views && git pull origin feat/portable-simd-and-views`, { ignoreError: false });
    } else {
        log('Cloning repository...');
        run(`git clone --branch feat/portable-simd-and-views "${REPO_URL}" "${SRC_DIR}"`);
    }

    // Build release
    log('Building SciRust (release mode, this may take a few minutes)...');
    run(`cd "${SRC_DIR}" && ${CARGO_BIN} build --release -p scirust`);

    // Copy binary
    const builtBin = path.join(SRC_DIR, 'target', 'release', 'scirust');
    if (!fs.existsSync(builtBin)) {
        error(`Build succeeded but binary not found at: ${builtBin}`);
    }
    fs.copyFileSync(builtBin, BIN_PATH);
    fs.chmodSync(BIN_PATH, 0o755);

    // Record version
    const version = runSilent(`cd "${SRC_DIR}" && git rev-parse --short HEAD`);
    fs.writeFileSync(VERSION_FILE, version);

    ok(`SciRust installed to ${BIN_PATH}`);
    ok(`Version: ${version}`);

    // Add to PATH
    addToPath();
    printNextSteps();
}

// ── Update ─────────────────────────────────────────────────────────────────

async function update() {
    if (!fs.existsSync(SRC_DIR)) {
        warn('SciRust is not installed. Running install instead...');
        return install();
    }

    const oldVersion = fs.existsSync(VERSION_FILE) ? fs.readFileSync(VERSION_FILE, 'utf8').trim() : 'unknown';
    log(`Current version: ${oldVersion}`);
    log('Checking for updates...');

    // Check latest commit on remote
    try {
        const apiData = await httpGetJSON(`${REPO_API}/branches/feat/portable-simd-and-views`);
        const latestSha = apiData.commit.sha.substring(0, 7);

        const localSha = runSilent(`cd "${SRC_DIR}" && git rev-parse --short HEAD`);

        if (localSha === latestSha) {
            ok(`Already up to date (${latestSha})`);
            return;
        }

        log(`Update available: ${localSha} → ${latestSha}`);
    } catch (e) {
        warn(`Could not check remote version: ${e.message}`);
        log('Proceeding with local rebuild...');
    }

    // Pull and rebuild
    run(`cd "${SRC_DIR}" && git fetch origin && git checkout feat/portable-simd-and-views && git pull origin feat/portable-simd-and-views`);
    run(`cd "${SRC_DIR}" && ${CARGO_BIN} build --release -p scirust`);

    const builtBin = path.join(SRC_DIR, 'target', 'release', 'scirust');
    if (!fs.existsSync(builtBin)) {
        error(`Build succeeded but binary not found at: ${builtBin}`);
    }
    fs.copyFileSync(builtBin, BIN_PATH);
    fs.chmodSync(BIN_PATH, 0o755);

    const newVersion = runSilent(`cd "${SRC_DIR}" && git rev-parse --short HEAD`);
    fs.writeFileSync(VERSION_FILE, newVersion);

    ok(`SciRust updated: ${oldVersion} → ${newVersion}`);
}

// ── PATH management ────────────────────────────────────────────────────────

function addToPath() {
    const exportLine = `\n# SciRust\nexport PATH="${BIN_DIR}:$PATH"`;

    for (const pf of [PROFILE_PATH, ZSH_PROFILE, FISH_PROFILE]) {
        if (!fs.existsSync(pf)) continue;

        let content = fs.readFileSync(pf, 'utf8');
        if (content.includes(BIN_DIR)) {
            log(`PATH already configured in ${pf}`);
            continue;
        }

        // For fish shell, use fish syntax
        if (pf === FISH_PROFILE) {
            const fishLine = `\n# SciRust\nfish_add_path "${BIN_DIR}"`;
            if (!content.includes(BIN_DIR)) {
                fs.appendFileSync(pf, fishLine);
                ok(`Added to ${pf}`);
            }
        } else {
            fs.appendFileSync(pf, exportLine);
            ok(`Added to ${pf}`);
        }
    }
}

function printNextSteps() {
    console.log(`\n  🚀 SciRust is ready!`);
    console.log(`\n  Add to current shell:`);
    console.log(`    export PATH="${BIN_DIR}:$PATH"`);
    console.log(`\n  Or restart your terminal, then:`);
    console.log(`    scirust              # show capabilities`);
    console.log(`    scirust simd         # SIMD benchmark`);
    console.log(`    scirust autodiff     # XOR classifier demo`);
    console.log(`    scirust symbolic     # symbolic math`);
    console.log(`    scirust bench        # all benchmarks\n`);
}

// ── Postinstall (npm postinstall hook) ─────────────────────────────────────

function postinstall() {
    // npm postinstall: only print guidance, don't auto-build
    console.log('');
    console.log('╔══════════════════════════════════════════════╗');
    console.log('║   SciRust installed via npm!                ║');
    console.log('╚══════════════════════════════════════════════╝');
    console.log('');
    console.log('  To install the SciRust binary:');
    console.log('    npx scirust-install install');
    console.log('');
    console.log('  To update:');
    console.log('    npx scirust-install update');
    console.log('');
}

// ── Main ───────────────────────────────────────────────────────────────────

const command = process.argv[2] || 'help';

(async () => {
    switch (command) {
        case 'install':
            await install();
            break;
        case 'update':
            await update();
            break;
        case 'postinstall':
            postinstall();
            break;
        case 'version':
        case '--version':
        case '-v':
            console.log('scirust npm installer v0.13.0');
            break;
        default:
            console.log('SciRust CLI Installer\n');
            console.log('Usage:');
            console.log('  npx scirust-install install    Install SciRust from source');
            console.log('  npx scirust-install update     Update to latest version');
            console.log('  scirust                        Run SciRust (after install)\n');
            console.log('After install, run:  scirust');
    }
})();
