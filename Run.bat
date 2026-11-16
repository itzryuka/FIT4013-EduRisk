@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
title EduRisk - Du doan sinh vien bo hoc
cd /d "%~dp0"

rem ===== Tim Python (uu tien moi truong ao .venv neu co) =====
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" --version >nul 2>&1
if errorlevel 1 (
    echo [LOI] Khong tim thay Python. Hay cai Python 3.10+ tu https://www.python.org/downloads/
    pause
    exit /b 1
)

rem ===== Tham so dong lenh: Run.bat [1-6] chay thang lua chon, khong hien menu =====
set "ARGMODE="
if not "%~1"=="" (
    set "ARGMODE=1"
    set "choice=%~1"
    goto dispatch
)

:menu
cls
echo ==========================================================
echo   EduRisk - He thong canh bao som sinh vien bo hoc
echo   FIT4013 - De tai 8
echo ==========================================================
for /f "delims=" %%v in ('"%PY%" --version') do echo   Python: %%v  ^(%PY%^)
if exist "data\data.csv" (echo   Du lieu : OK) else (echo   Du lieu : CHUA CO - dat data.csv vao thu muc data\)
if exist "models\full.joblib" (echo   Mo hinh : DA HUAN LUYEN) else (echo   Mo hinh : CHUA HUAN LUYEN)
echo ----------------------------------------------------------
echo   [1] Chay nhanh ^(tu cai dat + huan luyen neu thieu, mo web^)
echo   [2] Cai dat thu vien ^(requirements.txt^)
echo   [3] Tao moi truong ao .venv va cai thu vien
echo   [4] Huan luyen lai toan bo mo hinh ^(~3 phut^)
echo   [5] Chay web app
echo   [6] Kiem thu nhanh cac mo hinh da luu ^(in ket qua ra man hinh^)
echo   [0] Thoat
echo   ^(Meo: chay "Run.bat 6" de chon thang khong qua menu^)
echo ----------------------------------------------------------
set "choice="
set /p "choice=Chon [1]: "
if "%choice%"=="" set "choice=1"

:dispatch
if "%choice%"=="1" goto quick
if "%choice%"=="2" goto install
if "%choice%"=="3" goto venv
if "%choice%"=="4" goto train
if "%choice%"=="5" goto app
if "%choice%"=="6" goto test
if "%choice%"=="0" exit /b 0
if defined ARGMODE (echo [LOI] Lua chon khong hop le: %choice% & exit /b 1)
goto menu

rem ===== Ket thuc mot lua chon: o che do tham so thi thoat, nguoc lai quay ve menu =====
:done
if defined ARGMODE exit /b %errorlevel%
pause
goto menu

:check_data
if not exist "data\data.csv" (
    echo [LOI] Thieu data\data.csv. Tai tu https://archive.ics.uci.edu/dataset/697 va giai nen vao thu muc data\
    pause
    exit /b 1
)
exit /b 0

:install
echo.
echo === Cai dat thu vien ===
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (echo [LOI] Cai dat that bai. & goto done)
echo [OK] Da cai dat xong.
goto done

:venv
echo.
echo === Tao moi truong ao .venv ===
python -m venv .venv
if errorlevel 1 (echo [LOI] Khong tao duoc .venv & goto done)
set "PY=.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements.txt
echo [OK] Da tao .venv va cai thu vien.
goto done

:train
call :check_data || goto done
echo.
echo === Huan luyen toan bo thi nghiem ===
"%PY%" -m src.train
if errorlevel 1 (echo [LOI] Huan luyen that bai. & goto done)
echo [OK] Ket qua luu trong results\ va models\
goto done

:test
if not exist "models\full.joblib" (echo [LOI] Chua co mo hinh, chon [4] de huan luyen. & goto done)
echo.
"%PY%" -m src.test_models
goto done

:quick
call :check_data || goto done
"%PY%" -c "import sklearn, flask, pandas, matplotlib, joblib" >nul 2>&1
if errorlevel 1 (
    echo Thieu thu vien, dang cai dat...
    "%PY%" -m pip install -r requirements.txt || (echo [LOI] Cai dat that bai. & goto done)
)
if not exist "models\full.joblib" (
    echo Chua co mo hinh, dang huan luyen ^(~3 phut^)...
    "%PY%" -m src.train || (echo [LOI] Huan luyen that bai. & goto done)
)
goto app

:app
if not exist "models\full.joblib" (echo [LOI] Chua co mo hinh, chon [4] de huan luyen. & goto done)
echo.
echo === Khoi dong web app tai http://127.0.0.1:5000  ^(Ctrl+C de dung^) ===
start "" cmd /c "timeout /t 4 >nul & start http://127.0.0.1:5000"
"%PY%" -m app.app
goto done
