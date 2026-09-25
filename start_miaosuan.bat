@echo off
chcp 65001 >nul
title MiaoSuan Launcher
cd /d "%~dp0"
.venv\Scripts\python.exe launcher.py