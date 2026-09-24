#!/usr/bin/env node
/**
 * 妙算 Launcher — 一键启动器
 * =========================================================
 * 职责:
 *   1. 启动后端 (FastAPI on 127.0.0.1:8686, Python 3.14)
 *   2. 等待后端就绪 (轮询 /api/meta)
 *   3. 用系统默认浏览器打开 http://127.0.0.1:8686
 *   4. 监控后端进程，退出时提示
 *
 * 用法:  node launcher/run.js
 *
 * 注: 不再使用 Electron — 无 GPU 环境下 GPU 进程会崩溃，
 *     改用系统浏览器可兼容所有环境。
 */
'use strict';

const { spawn, exec } = require('child_process');
const http = require('http');
const path = require('path');
const fs = require('fs');

const ROOT = path.resolve(__dirname, '..');
const PORT = 8686;
const FRONT_URL = `http://127.0.0.1:${PORT}`;
const PYTHON = findPython();
const LOG_FILE = path.join(ROOT, 'tmp', 'launcher_backend.log');

// ── 1. 定位 Python ──
function findPython() {
  // .venv 优先（miaosuan 及所有依赖装在这里），系统 Python 作为兜底
  const candidates = [
    path.join(ROOT, '.venv', 'Scripts', 'python.exe'),
    'C:\\Python314\\python.exe',
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return process.platform === 'win32' ? 'python' : 'python3';
}

// ── 2. 探测后端端口是否在监听 ──
function portUp(timeout = 1000) {
  return new Promise((resolve) => {
    const req = http.request({
      host: '127.0.0.1', port: PORT, path: '/api/meta',
      method: 'GET', timeout,
    }, (res) => {
      res.resume();
      resolve(true);
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
    req.end();
  });
}

// ── 3. 杀掉占用端口的旧进程 ──
function killPort() {
  try {
    const out = require('child_process').execFileSync(
      'netstat', ['-ano'], { encoding: 'utf8', windowsHide: true });
    const lines = out.split(/\r?\n/);
    const listeners = lines.filter((l) =>
      l.includes(`:${PORT}`) && l.includes('LISTENING'));
    for (const l of listeners) {
      const parts = l.trim().split(/\s+/);
      const pid = parts[parts.length - 1];
      if (pid && pid !== '0') {
        try { process.kill(Number(pid)); } catch (e) {}
      }
    }
  } catch (e) {}
}

// ── 4. 启动后端 ──
function startBackend() {
  return new Promise((resolve) => {
    // 确保 tmp 目录存在
    const tmpDir = path.join(ROOT, 'tmp');
    if (!fs.existsSync(tmpDir)) fs.mkdirSync(tmpDir, { recursive: true });

    const logStream = fs.openSync(LOG_FILE, 'a');
    const env = {
      ...process.env,
      PYTHONPATH: `${ROOT};${process.env.PYTHONPATH || ''}`,
      PYTHONIOENCODING: 'utf-8',
    };

    const args = ['-m', 'miaosuan.cli', 'ui', '--host', '127.0.0.1', '--port', String(PORT)];
    console.log(`[launcher] Python:  ${PYTHON}`);
    console.log(`[launcher] 启动命令: python ${args.join(' ')}`);
    console.log(`[launcher] 日志文件: ${LOG_FILE}`);

    const child = spawn(PYTHON, args, {
      cwd: ROOT, env, windowsHide: true,
      stdio: ['ignore', logStream, logStream],
    });

    child.on('error', (err) => {
      console.error('[launcher] 后端启动失败:', err.message);
      resolve(false);
    });

    child.on('exit', (code) => {
      console.log(`[launcher] 后端进程退出 (code=${code})`);
    });

    // 等待端口就绪（最多 30s）
    const t0 = Date.now();
    const iv = setInterval(async () => {
      if (await portUp(500)) {
        clearInterval(iv);
        resolve(true);
      } else if (Date.now() - t0 > 30000) {
        clearInterval(iv);
        console.error('[launcher] 后端 30s 内未就绪，请检查日志:', LOG_FILE);
        resolve(false);
      }
    }, 600);
  });
}

// ── 5. 打开系统浏览器 ──
function openBrowser() {
  const cmd = process.platform === 'win32'
    ? `start "" "${FRONT_URL}"`
    : process.platform === 'darwin'
      ? `open "${FRONT_URL}"`
      : `xdg-open "${FRONT_URL}"`;
  exec(cmd, (err) => {
    if (err) {
      console.error('[launcher] 无法打开浏览器:', err.message);
      console.log(`[launcher] 请手动访问: ${FRONT_URL}`);
    } else {
      console.log(`[launcher] 浏览器已打开: ${FRONT_URL}`);
    }
  });
}

// ── 6. 主流程 ──
async function main() {
  console.log('');
  console.log('================================');
  console.log('  妙算 XAUUSD 量化交易系统');
  console.log('  一键启动器 v2.0 (浏览器模式)');
  console.log('================================');
  console.log('');

  // 检查后端是否已在运行
  if (await portUp(700)) {
    console.log('[launcher] 后端已在运行，直接打开浏览器...');
    openBrowser();
    return;
  }

  // 端口被占用但非后端 → 清理
  killPort();
  await new Promise((r) => setTimeout(r, 800));

  // 启动后端
  console.log('[launcher] 正在启动后端...');
  const ok = await startBackend();
  if (ok) {
    console.log('[launcher] 后端就绪 ✓');
    console.log(`[launcher] 访问地址: ${FRONT_URL}`);
    openBrowser();
  } else {
    console.error('[launcher] 后端启动失败 ✗');
    console.error(`[launcher] 请查看日志: ${LOG_FILE}`);
    process.exit(1);
  }
}

// Run when executed directly
if (require.main === module) {
  main().catch(err => {
    console.error('[launcher] 未捕获的错误:', err);
    process.exit(1);
  });
}