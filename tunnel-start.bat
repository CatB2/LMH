@echo off
chcp 65001 >nul 2>&1
echo ====================================
echo  SSH - Jie Guo VLLM
echo ====================================
echo.
echo : 106.38.203.153:36754 ^-^> localhost:8000
echo.

REM Kill existing tunnel
ssh -S NONE -O exit root@connect.bjb1.seetacloud.com -p 36754 -L 8000:localhost:8000 2>nul

REM Create tunnel (runs in background after password)
ssh -L 8000:localhost:8000 root@connect.bjb1.seetacloud.com -p 36754 -N -f

if errorlevel 1 (
    echo [Error] Tunnel failed. Check password and network.
    pause
    exit /b 1
)

echo [OK] Tunnel established!
echo.
echo Test: curl http://localhost:8000/v1/models
echo Stop: tunnel-stop.bat
echo.
pause