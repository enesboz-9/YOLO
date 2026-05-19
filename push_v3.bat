@echo off

set HF_KULLANICI=enesboz9
set HF_SPACE=yolo
set KAYNAK=C:\Users\enesb\YOLO
set HEDEF=C:\Users\enesb\YOLO\spaces_repo
set AUTH_URL=https://enesboz9:TOKEN_BURAYA@huggingface.co/spaces/enesboz9/yolo

echo.
echo === Hugging Face Spaces Push ===
echo.

git --version >nul 2>&1
if errorlevel 1 (
    echo [HATA] Git yuklu degil. https://git-scm.com/download/win
    pause
    exit /b 1
)
echo [OK] Git bulundu.

if not exist "%KAYNAK%\app.py" (
    echo [HATA] app.py bulunamadi: %KAYNAK%\app.py
    pause
    exit /b 1
)
echo [OK] app.py bulundu.

if exist "%HEDEF%" (
    echo Eski repo siliniyor...
    rmdir /s /q "%HEDEF%"
)

echo Repo klonlaniyor...
git clone "%AUTH_URL%" "%HEDEF%"
if errorlevel 1 (
    echo [HATA] Klonlama basarisiz. Token ve Space adini kontrol et.
    pause
    exit /b 1
)
echo [OK] Klonlama tamam.

echo Dosyalar kopyalaniyor...
copy /Y "%KAYNAK%\app.py"           "%HEDEF%\app.py"
copy /Y "%KAYNAK%\requirements.txt" "%HEDEF%\requirements.txt"
copy /Y "%KAYNAK%\README.md"        "%HEDEF%\README.md"
if exist "%KAYNAK%\.gitignore" copy /Y "%KAYNAK%\.gitignore" "%HEDEF%\.gitignore"
echo [OK] Dosyalar kopyalandi.

cd /d "%HEDEF%"
git config user.email "enesboz9@hf.co"
git config user.name "enesboz9"
git add .
git commit -m "deploy: YOLOv8 arac tespiti"
git push "%AUTH_URL%" main
if errorlevel 1 (
    echo [HATA] Push basarisiz!
    pause
    exit /b 1
)

echo.
echo === BASARILI! Space hazirlanıyor: https://huggingface.co/spaces/enesboz9/yolo ===
echo.
pause
