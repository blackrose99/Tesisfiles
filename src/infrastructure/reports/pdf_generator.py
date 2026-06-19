import os
from fpdf import FPDF
from datetime import datetime
from typing import Dict, Any, List
from src.domain.entities.student import Student
from src.domain.entities.prediction import Prediction

class AcademicPDFReport(FPDF):
    """
    FPDF subclass to generate standardized academic risk reports.
    """
    def header(self):
        # Header banner
        self.set_fill_color(26, 26, 46) # Dark Indigo #1a1a2e
        self.rect(0, 0, 210, 40, "F")
        
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 18)
        self.cell(0, 5, "REPORTE DE ALERTA TEMPRANA ACADÉMICA", ln=True, align="C")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 8, "Sistema Inteligente de Predicción de Deserción", ln=True, align="C")
        self.ln(15)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Página {self.page_no()} | Reporte generado el {datetime.now().strftime('%d/%m/%Y %H:%M')}", align="C")


class PDFGenerator:
    """
    Service class to compile student prediction reports into PDFs.
    """
    @staticmethod
    def generate_student_report(student: Student, prediction: Prediction, output_dir: str = "archivos_procesados/reportes") -> str:
        os.makedirs(output_dir, exist_ok=True)
        pdf = AcademicPDFReport()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # 1. Student Info Section
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(26, 26, 46)
        pdf.cell(0, 8, "1. Información General del Estudiante", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        
        # Columns info
        col_w = 45
        val_w = 50
        
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(col_w, 6, "Código Estudiante:")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(val_w, 6, str(student.code_student))
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(col_w, 6, "Programa Académico:")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(val_w, 6, str(student.programa), ln=True)
        
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(col_w, 6, "Ubicación Semestral:")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(val_w, 6, f"Semestre {student.ubicacion_semestral}")
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(col_w, 6, "Jornada:")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(val_w, 6, str(student.jornada), ln=True)
        
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(col_w, 6, "Promedio General:")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(val_w, 6, f"{student.promedio_general:.2f}")
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(col_w, 6, "Créditos Aprobados:")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(val_w, 6, f"{student.creditos_aprobados} créditos", ln=True)
        
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(col_w, 6, "Estrato Social:")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(val_w, 6, f"Estrato {int(student.estrato)}" if student.estrato is not None else "No reportado")
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(col_w, 6, "Género:")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(val_w, 6, str(student.genero), ln=True)
        
        pdf.ln(5)
        
        # 2. Risk Assessment Box
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(26, 26, 46)
        pdf.cell(0, 8, "2. Diagnóstico de Riesgo de Deserción", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        # Color Box for Risk
        # Green: Bajo, Yellow: Moderado, Red: Alto
        prob_pct = prediction.probability * 100
        risk = prediction.risk_level
        
        if risk == "Bajo":
            r, g, b = 230, 245, 230 # Soft Green
            tr, tg, tb = 39, 174, 96 # Dark Green text
        elif risk == "Moderado":
            r, g, b = 255, 249, 219 # Soft Yellow
            tr, tg, tb = 217, 131, 0 # Dark Yellow text
        else:
            r, g, b = 253, 237, 237 # Soft Red
            tr, tg, tb = 231, 76, 60 # Dark Red text
            
        pdf.set_fill_color(r, g, b)
        pdf.rect(10, pdf.get_y(), 190, 24, "F")
        
        pdf.set_xy(15, pdf.get_y() + 3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(tr, tg, tb)
        pdf.cell(0, 6, f"NIVEL DE RIESGO: {risk.upper()}", ln=True)
        
        pdf.set_x(15)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(0, 6, f"Probabilidad Calculada de Deserción: {prob_pct:.2f}% (Umbral del modelo: 50.00%)", ln=True)
        
        pdf.ln(12)
        
        # 3. Factor Analysis (SHAP)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(26, 26, 46)
        pdf.cell(0, 8, "3. Análisis de Factores de Riesgo (SHAP)", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 5, "Las siguientes variables son las que más influyeron en la predicción del modelo. Los valores positivos incrementan el riesgo, mientras que los valores negativos lo disminuyen.")
        pdf.ln(3)
        
        # Table of Top 5 SHAP values
        pdf.set_fill_color(240, 240, 240)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(50, 50, 50)
        
        pdf.cell(90, 8, "Variable", 1, 0, "C", fill=True)
        pdf.cell(50, 8, "Impacto (SHAP)", 1, 0, "C", fill=True)
        pdf.cell(50, 8, "Efecto", 1, 1, "C", fill=True)
        
        pdf.set_font("Helvetica", "", 10)
        
        sorted_shap = sorted(prediction.shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        # Translate feature names for Spanish report
        feat_translations = {
            "PROMEDIO_GENERAL": "Promedio Académico General",
            "CREDITOSAPROBADOS": "Créditos Aprobados Totales",
            "UBICACION_SEMESTRAL": "Ubicación Semestral (Semestre)",
            "ESTRATO": "Estrato Socioeconómico",
            "TIENE_SISBEN": "Tiene Sisbén",
            "EDAD_INGRESO": "Edad al Ingreso",
            "CREDITOS_POR_SEMESTRE": "Ritmo de Aprobación Académica",
            "RENDIMIENTO_GENERAL": "Rendimiento Académico General",
            "TASA_VULNERABILIDAD": "Índice de Vulnerabilidad Social",
            "APOYO_FAMILIAR": "Índice de Apoyo Familiar",
            "JORNADA_NOCTURNA": "Jornada Nocturna",
            "JORNADA_DIURNA": "Jornada Diurna",
            "GENERO_M": "Género Masculino",
            "GENERO_F": "Género Femenino",
            "CIUDADRESIDENCIA_OTROS": "Reside fuera del Área Metropolitana",
            "CIUDADRESIDENCIA_BUCARAMANGA": "Residente en Bucaramanga",
            "CIUDADRESIDENCIA_FLORIDABLANCA": "Residente en Floridablanca",
            "CIUDADRESIDENCIA_GIRON": "Residente en Girón",
            "CIUDADRESIDENCIA_PIEDECUESTA": "Residente en Piedecuesta",
            "CATEGORIA_NUEVO REGULAR": "Estudiante Nuevo Regular",
            "CATEGORIA_ANTIGUO": "Estudiante Antiguo",
            "CATEGORIA_NUEVO REINGRESO": "Estudiante de Reingreso"
        }
        
        for feat, val in sorted_shap:
            friendly_name = feat_translations.get(feat, feat)
            effect_text = "Incrementa Riesgo" if val > 0 else "Reduce Riesgo"
            
            # Text color depending on risk
            if val > 0:
                pdf.set_text_color(200, 50, 50)
            else:
                pdf.set_text_color(50, 150, 50)
                
            pdf.cell(90, 8, f" {friendly_name}", 1, 0, "L")
            pdf.cell(50, 8, f" {val:+.4f}", 1, 0, "C")
            pdf.cell(50, 8, f" {effect_text}", 1, 1, "C")
            
        pdf.set_text_color(50, 50, 50)
        pdf.ln(5)
        
        # 4. Intervention Plan
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(26, 26, 46)
        pdf.cell(0, 8, "4. Plan de Intervención Recomendado", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)
        
        pdf.set_font("Helvetica", "", 10)
        
        # Logic-based recommendation rules
        recs = []
        if student.promedio_general < 3.3:
            recs.append("Asignar tutoría académica prioritaria en asignaturas críticas y vincular al programa de refuerzo escolar.")
        if student.creditos_aprobados / max(1, student.ubicacion_semestral) < 10:
            recs.append("Revisar carga académica. Se sugiere consejería para planeación de matrícula y reducción de materias por semestre.")
        if student.estrato is not None and student.estrato <= 2:
            recs.append("Derivar a Bienestar Universitario para evaluar elegibilidad de apoyos económicos, subsidio de transporte o comedor.")
        if student.jornada == "NOCTURNA":
            recs.append("Facilitar canales virtuales de atención y flexibilizar horarios de tutorías académicas para estudiantes trabajadores.")
            
        if not recs:
            recs.append("Monitorear el rendimiento semestral estándar. El estudiante cuenta actualmente con indicadores estables.")
            
        for r_idx, rec in enumerate(recs):
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(10, 6, f"{r_idx+1}.", 0, 0, "R")
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(180, 6, rec)
            
        # Write PDF to file
        output_filename = f"reporte_estudiante_{student.code_student}.pdf"
        output_path = os.path.join(output_dir, output_filename)
        pdf.output(output_path)
        
        return output_path
