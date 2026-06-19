import os
import sys
from src.infrastructure.repositories.student_repository_impl import StudentRepositoryImpl
from src.infrastructure.repositories.model_repository_impl import ModelRepositoryImpl
from src.application.train_use_case import TrainUseCase

def main():
    # Windows consoles default to cp1252, which can't encode the emoji/check
    # marks below and would otherwise crash the script after training already succeeded.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print("==================================================")
    # Correct Python path to find src/
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    
    # Instantiate repositories
    student_repo = StudentRepositoryImpl()
    model_repo = ModelRepositoryImpl()
    
    # Instantiate Use Case
    train_use_case = TrainUseCase(student_repo, model_repo)
    
    # Raw dataset path
    data_path = "dataset/student_database.xlsx"
    if not os.path.exists(data_path):
        print(f"Error: No se encontró el archivo '{data_path}' en el directorio dataset/.")
        return
        
    print(f"Iniciando entrenamiento del modelo desde: {data_path}")
    
    # We will exclude ANIO_INGRESO and MES_INGRESO to prevent temporal bias 
    # and make the model generalizable to new cohorts of students.
    exclude_features = ["ANIO_INGRESO", "MES_INGRESO"]
    
    # Let's train using all programs as requested (allowed_programs=None uses all)
    # We'll set n_trials to 15 for hyperparameter tuning to keep it fast but effective.
    try:
        results = train_use_case.execute(
            raw_data_path=data_path,
            allowed_programs=None,
            exclude_features=exclude_features,
            n_trials=15
        )
        
        print("\n==================================================")
        print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print(f"Modelo Campeón: {results['model_name']}")
        print(f"Versión Registrada: {results['version']}")
        print(f"Número de Features: {len(results['features_trained'])}")
        print("Métricas en Test:")
        print(f"  Accuracy:  {results['metrics']['accuracy']:.4f}")
        print(f"  Precision: {results['metrics']['precision']:.4f}")
        print(f"  Recall:    {results['metrics']['recall']:.4f}")
        print(f"  F1-Score:  {results['metrics']['f1']:.4f}")
        print(f"  ROC-AUC:   {results['metrics']['roc_auc']:.4f}")
        print("==================================================")
        
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
