@echo off
title Sistema de Alerta Temprana - Iniciador
chcp 65001 > nul
cls

echo ======================================================================
echo           SISTEMA DE ALERTA TEMPRANA - INICIADOR AUTOMÁTICO (WINDOWS)
echo ======================================================================
echo.

:: 1. Verificar si Python está instalado
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python no está instalado o no se encuentra en el PATH.
    echo Por favor, instala Python 3.9 o superior y asegúrate de marcar la casilla:
    echo "Add Python to PATH" durante la instalación.
    echo.
    echo Abriendo la página oficial de descargas de Python...
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Mostrar versión de Python detectada
echo [INFO] Python detectado:
python --version
echo.

:: 2. Crear y configurar entorno virtual .venv
if not exist ".venv" (
    echo [INFO] No se encontró el entorno virtual. Creando .venv...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
    echo [INFO] Entorno virtual creado exitosamente.
)

:: 3. Instalar/Actualizar dependencias
echo [INFO] Validando e instalando dependencias (requirements.txt)...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Error durante la instalación de dependencias.
    pause
    exit /b 1
)

:: 4. Verificar si existe el modelo entrenado
if not exist "model_registry\registry.json" (
    echo [INFO] No se encontró un modelo registrado. Iniciando entrenamiento inicial...
    .venv\Scripts\python.exe train.py
    if %errorlevel% neq 0 (
        echo [ERROR] Error al entrenar el modelo.
        pause
        exit /b 1
    )
)

:: 5. Iniciar la aplicación Streamlit
echo.
echo [INFO] Iniciando el servidor Streamlit en segundo plano...
echo La aplicación se abrirá automáticamente en tu navegador web.
echo Si no se abre, ingresa manualmente a: http://localhost:8501
echo.
echo Presiona Ctrl+C en esta consola para detener la aplicación.
echo ======================================================================
echo.

.venv\Scripts\streamlit.exe run app.py

pause
