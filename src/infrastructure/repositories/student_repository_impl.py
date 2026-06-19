import pandas as pd
import csv
import io
from typing import List
from src.domain.entities.student import Student
from src.domain.entities.prediction import Prediction
from src.domain.repositories.student_repository import StudentRepository

class StudentRepositoryImpl(StudentRepository):
    """
    Concrete implementation of StudentRepository for Excel and CSV data sources.
    """
    def load_from_excel(self, file_path: str) -> List[Student]:
        df = pd.read_excel(file_path)
        # Normalize column names
        df.columns = df.columns.astype(str).str.strip()
        students = []
        for _, row in df.iterrows():
            students.append(Student.from_dict(row.to_dict()))
        return students

    def load_from_csv(self, file_path: str, delimiter: str = None) -> List[Student]:
        if delimiter is None:
            # Detect delimiter automatically
            with open(file_path, "rb") as f:
                content = f.read(2048)
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = content.decode("latin-1")
            try:
                dialect = csv.Sniffer().sniff(text[:1024], delimiters=[",", ";", "\t", "|"])
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ","

        df = pd.read_csv(file_path, sep=delimiter, engine="python")
        df.columns = df.columns.astype(str).str.strip()
        students = []
        for _, row in df.iterrows():
            students.append(Student.from_dict(row.to_dict()))
        return students

    def save_predictions(self, predictions: List[Prediction], output_path: str) -> None:
        data = []
        for pred in predictions:
            row = {
                "identificador": pred.code_student,
                "p_desercion": round(pred.probability, 4),
                "nivel_riesgo": pred.risk_level,
                "resultado_modelo": "Deserta" if pred.is_dropout else "No deserta"
            }
            # Flatten SHAP values if they are present
            for k, v in pred.shap_values.items():
                row[f"shap_{k}"] = round(v, 6)
            data.append(row)
        
        df = pd.DataFrame(data)
        if output_path.endswith(".xlsx"):
            df.to_excel(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
