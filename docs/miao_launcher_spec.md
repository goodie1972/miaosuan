# One-Click Launcher Specification for MiaOSuan

## Goal
Create a standalone Windows executable that launches the MiaOSuan CLI UI web server and opens the browser automatically.

## Deliverables
- `miao_launcher.exe` (≤ 50 MB)
- Starts `python -m miaosuan.cli ui --host 127.0.0.1 --port 8686` without a console window
- Shows a simple splash screen while starting
- Opens the default browser to `http://127.0.0.1:8686` after the server is ready
- Terminates all child processes cleanly on exit

## Functional Requirements
1. **Hide the command‑line console console.**
2. Launch a background process running `python -m miaosuan.cli ui --host 127.0.0.1 --port 8686`.
3. After detecting that the server is ready (port open), open the default browser to `http://127.0.0.1:8686`.
3.1 Include a minimal HTML progress page that can be displayed while waiting.
4. When the user closes the browser window or kills the exe, stop the background Python process and detach from any remaining sockets.
5. Gracefully handle errors (e.g., port in use) and return a friendly message box.

## Non‑Functional Requirements
| Requirement | Description | Constraint |
|-----------|-------------|----------|
| **File size** | ≤ 50 MB | All dependencies must be packaged |
| **Startup time** | ≤ 5 seconds after double‑click | Including splash screen display |
| **Interface** | No console window; only a splash screen | Only a single UI dialog shown |
| **Compatibility** | Windows 10 (64‑bit) with WebView2 Runtime installed | Must work on clean Windows images |
| **Exit behavior** | All child processes terminated cleanly | No orphan processes left |

## Assumptions
- Target machine has Internet access only for downloading the WebView2 Runtime if it is not present.
- The executable will be distributed together with the MiaOSuan source (or as a separate package).

## Open Questions
- Should we bundle a custom HTML splash page or use a native dialog?
- How to handle port conflicts when another MiaOSuan instance is already running?
- Should the executable include bundled data (e.g., a default `spec.json`)?

## Implementation Approach
| Step | Description |
|------|-------------|
| **1. Generate spec** | Finalize functional & non‑functional requirements |
| **2. Choose tech stack** | Prefer `windows-webview2-python-launcher` skill for native launcher; fallback to PyInstaller + custom script |
| **3. Create project skeleton** | Add `launcher/` folder with `main.py`, `service.py`, `ui_template.html`, `pyinstaller.spec` |
| **4. Implement launcher** | - Hide console, start Python process as child, wait for port, open browser |
| **5. Test on clean Windows VM** | Verify size, startup time, clean exit |
| **6. Package** | Use PyInstaller or C# build to produce `miao_launcher.exe` |
| **7. Document usage** | Add README with double‑click instructions |

## Success Metrics
- Executable launches and opens browser to MiaOSuan UI within 5 seconds.
- All background processes terminate when user exits the browser.
- No leftover sockets or orphan processes.
- File size ≤ 50 MB.

---  

*Prepared for the MiaOSuan development team to enable a one‑click deployment experience.[Agent ID: agent-82cfff49]*