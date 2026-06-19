import os
import shutil
import sys
import streamlit.web.cli as stcli

# Archivos y carpetas que deben persistir entre ejecuciones del .exe
# (modelo entrenado, datos cargados, reportes generados, explicabilidad SHAP).
PERSISTENT_ENTRIES = [
    "dataset",
    "model_registry",
    "model.joblib",
    "scaler.joblib",
    "shap_background.npy",
    "shap_feature_names.npy",
]

if __name__ == '__main__':
    is_frozen = getattr(sys, "frozen", False)

    if is_frozen:
        # PyInstaller --onefile extrae los recursos de solo lectura (app.py, src/,
        # copia inicial del modelo) a una carpeta temporal (_MEIPASS) que Windows
        # borra al cerrar el programa. Si trabajamos ahí, el modelo reentrenado,
        # el dataset actualizado y los reportes generados desaparecerían en cada
        # apertura del .exe, a diferencia de "streamlit run app.py" en el repo.
        bundle_dir = sys._MEIPASS
        # Carpeta junto al .exe: vive mientras el usuario no la borre, igual que
        # la carpeta del proyecto cuando se ejecuta en modo desarrollador.
        data_dir = os.path.dirname(os.path.abspath(sys.executable))

        for entry in PERSISTENT_ENTRIES:
            src_path = os.path.join(bundle_dir, entry)
            dst_path = os.path.join(data_dir, entry)
            if os.path.exists(dst_path) or not os.path.exists(src_path):
                continue
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

        sys.path.append(bundle_dir)
        os.chdir(data_dir)
        app_path = os.path.join(bundle_dir, "app.py")
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        os.chdir(current_dir)
        app_path = os.path.join(current_dir, "app.py")

    sys.argv = ["streamlit", "run", app_path, "--global.developmentMode=false"]
    sys.exit(stcli.main())
