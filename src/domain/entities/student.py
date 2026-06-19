from typing import Dict, Any, Optional
from datetime import datetime

class Student:
    """
    Domain entity representing a student and their academic/socioeconomic features.
    """
    def __init__(self,
                 code_student: str,
                 fecha_ingreso: Optional[datetime],
                 creditos_aprobados: float,
                 ubicacion_semestral: int,
                 promedio_general: float,
                 programa: str,
                 jornada: str,
                 genero: str,
                 fecha_nacimiento: Optional[datetime],
                 ciudad_residencia: str,
                 estrato: Optional[float],
                 tiene_sisben: Optional[float],
                 vive_con_familia: Optional[float],
                 situacion_padres: Optional[str],
                 numero_familiares: Optional[float],
                 numero_hermanos: Optional[float],
                 posicion_en_hermanos: Optional[float],
                 miembros_trabajan: Optional[float],
                 situacion: Optional[str] = None,
                 target: Optional[int] = None):
        self.code_student = code_student
        self.fecha_ingreso = fecha_ingreso
        self.creditos_aprobados = creditos_aprobados
        self.ubicacion_semestral = ubicacion_semestral
        self.promedio_general = promedio_general
        self.programa = programa
        self.jornada = jornada
        self.genero = genero
        self.fecha_nacimiento = fecha_nacimiento
        self.ciudad_residencia = ciudad_residencia
        self.estrato = estrato
        self.tiene_sisben = tiene_sisben
        self.vive_con_familia = vive_con_familia
        self.situacion_padres = situacion_padres
        self.numero_familiares = numero_familiares
        self.numero_hermanos = numero_hermanos
        self.posicion_en_hermanos = posicion_en_hermanos
        self.miembros_trabajan = miembros_trabajan
        self.situacion = situacion
        self.target = target

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Student":
        """
        Creates a Student instance from a dictionary containing raw Excel/CSV values.
        """
        # Parse dates
        def parse_date(val) -> Optional[datetime]:
            if pd_isna(val):
                return None
            if isinstance(val, datetime):
                return val
            try:
                return datetime.fromisoformat(str(val))
            except Exception:
                return None

        def to_float(val) -> Optional[float]:
            if pd_isna(val):
                return None
            try:
                val_str = str(val).strip().replace(" ", "").replace(",", ".")
                return float(val_str)
            except (ValueError, TypeError):
                return None

        def to_int(val) -> Optional[int]:
            if pd_isna(val):
                return None
            try:
                val_str = str(val).strip().replace(" ", "").replace(",", ".")
                return int(float(val_str))
            except (ValueError, TypeError):
                return None

        def pd_isna(val) -> bool:
            # Simple check for null
            if val is None:
                return True
            import pandas as pd
            return pd.isna(val)

        return cls(
            code_student=str(data.get("CODESTUDIANTE", "")).strip(),
            fecha_ingreso=parse_date(data.get("ESTP_FECHAINGRESO")),
            creditos_aprobados=to_float(data.get("CREDITOSAPROBADOS")) or 0.0,
            ubicacion_semestral=to_int(data.get("UBICACION_SEMESTRAL")) or 1,
            promedio_general=to_float(data.get("PROMEDIO_GENERAL")) or 0.0,
            programa=str(data.get("PROGRAMA", "")).strip().upper(),
            jornada=str(data.get("JORNADA", "")).strip().upper(),
            genero=str(data.get("GENERO", "")).strip().upper(),
            fecha_nacimiento=parse_date(data.get("FECHA_NACIMIENTO")),
            ciudad_residencia=str(data.get("CIUDADRESIDENCIA", "")).strip().upper(),
            estrato=to_float(data.get("ESTRATO")),
            tiene_sisben=to_float(data.get("TIENE_SISBEN")),
            vive_con_familia=to_float(data.get("INFE_VIVECONFAMILIA")),
            situacion_padres=data.get("INFE_SITUACIONPADRES") if not pd_isna(data.get("INFE_SITUACIONPADRES")) else None,
            numero_familiares=to_float(data.get("INFE_NUMEROFAMILIARES")),
            numero_hermanos=to_float(data.get("INFE_NUMEROHERMANOS")),
            posicion_en_hermanos=to_float(data.get("INFE_POSICIONENHERMANOS")),
            miembros_trabajan=to_float(data.get("INFE_NUMMIEMBROSTRABAJA")),
            situacion=data.get("SITUACION") if not pd_isna(data.get("SITUACION")) else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the Student instance back to a dictionary.
        """
        return {
            "CODESTUDIANTE": self.code_student,
            "ESTP_FECHAINGRESO": self.fecha_ingreso,
            "CREDITOSAPROBADOS": self.creditos_aprobados,
            "UBICACION_SEMESTRAL": self.ubicacion_semestral,
            "PROMEDIO_GENERAL": self.promedio_general,
            "PROGRAMA": self.programa,
            "JORNADA": self.jornada,
            "GENERO": self.genero,
            "FECHA_NACIMIENTO": self.fecha_nacimiento,
            "CIUDADRESIDENCIA": self.ciudad_residencia,
            "ESTRATO": self.estrato,
            "TIENE_SISBEN": self.tiene_sisben,
            "INFE_VIVECONFAMILIA": self.vive_con_familia,
            "INFE_SITUACIONPADRES": self.situacion_padres,
            "INFE_NUMEROFAMILIARES": self.numero_familiares,
            "INFE_NUMEROHERMANOS": self.numero_hermanos,
            "INFE_POSICIONENHERMANOS": self.posicion_en_hermanos,
            "INFE_NUMMIEMBROSTRABAJA": self.miembros_trabajan,
            "SITUACION": self.situacion,
            "TARGET": self.target
        }
