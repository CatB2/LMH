@echo off
chcp 65001 >nul
echo 关闭 SSH 隧道...
ssh -S NONE -O exit root@106.38.203.153 -p 36754 -L 8000:localhost:8000 2>nul

REM 也尝试通过进程名杀掉
taskkill /f /im ssh.exe 2>nul

echo [完成] 隧道已关闭
pause
