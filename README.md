# Predicción de Deserción Estudiantil

Aplicación Streamlit que predice la deserción estudiantil usando una red neuronal.

---

# Instalación Paso a Paso

## 1. Verificar Python

### Linux / macOS

```bash
python3 --version
```

### Windows

```cmd
python --version
```

Debe mostrar una versión entre:

```text
Python 3.9
Python 3.10
Python 3.11
Python 3.12
```

---

# Linux (Ubuntu / Debian)

## Instalar dependencias necesarias

```bash
sudo apt update
```

```bash
sudo apt install -y git python3 python3-pip python3-venv
```

## Clonar el proyecto

```bash
git clone https://github.com/blackrose99/Tesisfiles.git
```

## Entrar al proyecto

```bash
cd Tesisfiles
```

## Crear entorno virtual

```bash
python3 -m venv .venv
```

## Activar entorno virtual

```bash
source .venv/bin/activate
```

## Actualizar pip

```bash
pip install --upgrade pip
```

## Instalar dependencias

```bash
pip install -r requirements.txt
```

## Verificar instalación

```bash
pip list
```

---

# Windows PowerShell

## Clonar el proyecto

```powershell
git clone https://github.com/blackrose99/Tesisfiles.git
```

## Entrar al proyecto

```powershell
cd Tesisfiles
```

## Crear entorno virtual

```powershell
python -m venv .venv
```

## Activar entorno virtual

```powershell
.venv\Scripts\Activate.ps1
```

Si aparece un error de ejecución:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

y vuelve a ejecutar:

```powershell
.venv\Scripts\Activate.ps1
```

## Actualizar pip

```powershell
python -m pip install --upgrade pip
```

## Instalar dependencias

```powershell
pip install -r requirements.txt
```

---

# Windows CMD

## Clonar el proyecto

```cmd
git clone https://github.com/blackrose99/Tesisfiles.git
```

## Entrar al proyecto

```cmd
cd Tesisfiles
```

## Crear entorno virtual

```cmd
python -m venv .venv
```

## Activar entorno virtual

```cmd
.venv\Scripts\activate.bat
```

## Actualizar pip

```cmd
python -m pip install --upgrade pip
```

## Instalar dependencias

```cmd
pip install -r requirements.txt
```

---

# Procesamiento y Entrenamiento

Después de instalar todo:

## Copiar archivo de datos

Coloque el archivo:

```text
Base de datos estudiantes.xlsx
```

en la carpeta raíz del proyecto:

```text
Tesisfiles/
```

## Limpiar datos

```bash
python "Limpieza de datos.py"
```

## Entrenar modelo

```bash
python train.py
```

Al finalizar se generarán:

```text
modelo_desercion_nn.keras
scaler.joblib
```

## Ejecutar aplicación

```bash
streamlit run app.py
```

---

# Solución de Problemas

## Error: ensurepip is not available

Ubuntu/Debian:

```bash
sudo apt install python3-venv
```

o

```bash
sudo apt install python3.10-venv
```

según la versión instalada.

---

## Error: python command not found

Verifique:

```bash
python3 --version
```

Si no existe:

```bash
sudo apt install python3
```

---

## Error: pip command not found

Ubuntu/Debian:

```bash
sudo apt install python3-pip
```

---

## Error: No module named ...

Active nuevamente el entorno virtual:

Linux:

```bash
source .venv/bin/activate
```

Windows:

```cmd
.venv\Scripts\activate.bat
```

o

```powershell
.venv\Scripts\Activate.ps1
```

Luego reinstale:

```bash
pip install -r requirements.txt
```

---

## Error al abrir Excel

Verifique que exista exactamente:

```text
Base de datos estudiantes.xlsx
```

en:

```text
Tesisfiles/Base de datos estudiantes.xlsx
```
