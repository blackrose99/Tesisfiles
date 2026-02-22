# 📋 INSTRUCCIONES PARA CARGA DE DATOS - MODELO DE DESERCIÓN

## 🎯 Columnas Obligatorias para Subir

Tu archivo CSV o Excel **DEBE** contener las siguientes columnas (los nombres deben ser exactos):

### ✅ Columnas Requeridas (18 columnas):

| Columna | Tipo | Descripción | Ejemplo |
|---------|------|-------------|---------|
| **ESTP_FECHAINGRESO** | Fecha | Fecha de ingreso del estudiante | 2023-01-15 |
| **SITUACION** | Texto | Situación académica actual | ACTIVO, PFI, INACTIVO |
| **CREDITOSAPROBADOS** | Número | Créditos aprobados | 45 |
| **UBICACION_SEMESTRAL** | Número | Semestre actual | 5 |
| **PROMEDIO_GENERAL** | Número | Promedio académico (0-5) | 3.8 |
| **PROGRAMA** | Texto | Nombre del programa | INGENIERIA DE SISTEMAS |
| **JORNADA** | Texto | Jornada de estudio | DIURNA, NOCTURNA |
| **GENERO** | Texto | Género del estudiante | M, F |
| **FECHA_NACIMIENTO** | Fecha | Fecha de nacimiento | 2000-05-20 |
| **CIUDADRESIDENCIA** | Texto | Ciudad donde reside | BUCARAMANGA, FLORIDABLANCA |
| **ESTRATO** | Número | Estrato socioeconómico (1-6) | 3 |
| **TIENE_SISBEN** | Número | Tiene SISBEN (0=No, 1=Sí) | 1 |
| **INFE_VIVECONFAMILIA** | Texto | Vive con familia | SI, NO |
| **INFE_SITUACIONPADRES** | Texto | Situación de los padres | VIVOS Y CONVIVEN |
| **INFE_NUMEROFAMILIARES** | Número | Número de familiares | 4 |
| **INFE_NUMEROHERMANOS** | Número | Número de hermanos | 2 |
| **INFE_POSICIONENHERMANOS** | Número | Posición entre hermanos | 1 |
| **INFE_NUMMIEMBROSTRABAJA** | Número | Miembros de familia que trabajan | 2 |

### 🔴 Columnas que NO debes incluir (se eliminan automáticamente):

Estas columnas NO son necesarias y se eliminan durante la limpieza:
- ❌ CODESTUDIANTE
- ❌ CODIGOCIUDADR
- ❌ NIVEL_SISBEN
- ❌ CATEGORIA
- ❌ CODMATRICULA
- ❌ SEDE
- ❌ INFE_HERMANOSESTUDIANDOU

---

## 📝 Reglas de Formato

### Fechas:
- **Formato aceptado**: YYYY-MM-DD, DD/MM/YYYY, o cualquier formato que Excel reconozca
- **Ejemplo**: 2023-01-15 o 15/01/2023

### Números:
- **PROMEDIO_GENERAL**: Entre 0.0 y 5.0
- **ESTRATO**: Entre 1 y 6 (si está fuera de rango, se marcará como 0="No reportado")
- **CREDITOSAPROBADOS**: Número entero positivo
- **Valores vacíos**: Se llenarán con códigos especiales (-1 para conteos, 0 para ESTRATO)

### Programas Aceptados:
Solo se procesarán estudiantes de estos programas:
- ✅ INGENIERIA DE SISTEMAS
- ✅ TECNOLOGIA EN DESARROLLO DE SISTEMAS INFORMATICOS

Estudiantes de otros programas serán **automáticamente eliminados**.

### SITUACION - Codificación Automática:
El sistema automáticamente convertirá SITUACION a binario:
- **1 (Desertor)**: EXCLUIDO NO RENOVACION DE MATRICULA, PFI, EXCLUIDO CANCELACION SEMESTRE, INACTIVO
- **0 (Activo)**: Cualquier otra situación

### CIUDADRESIDENCIA - Codificación Automática:
- BUCARAMANGA → 1
- FLORIDABLANCA → 2
- GIRON → 3
- PIEDECUESTA → 4
- Otras ciudades → 5

---

## 🔄 Proceso de Limpieza Automática

Tu archivo pasará por estos pasos:

1. **Eliminación de columnas** no necesarias
2. **Conversión de fechas** a variables numéricas (EDAD_INGRESO, ANIO_INGRESO, MES_INGRESO)
3. **Validación de ESTRATO** (1-6, fuera de rango → 0)
4. **Filtrado de programas** (solo Ingeniería de Sistemas y Tecnología)
5. **Codificación de SITUACION** (binario 0/1)
6. **Recodificación de CIUDADRESIDENCIA** (1-5)
7. **One-Hot Encoding** de variables categóricas (PROGRAMA, JORNADA, GENERO, etc.)
8. **Llenado de valores faltantes** con códigos especiales
9. **Conversión a formato final** de 26 columnas para el modelo

---

## 📊 Ejemplo de Archivo CSV

```csv
ESTP_FECHAINGRESO,SITUACION,CREDITOSAPROBADOS,UBICACION_SEMESTRAL,PROMEDIO_GENERAL,PROGRAMA,JORNADA,GENERO,FECHA_NACIMIENTO,CIUDADRESIDENCIA,ESTRATO,TIENE_SISBEN,INFE_VIVECONFAMILIA,INFE_SITUACIONPADRES,INFE_NUMEROFAMILIARES,INFE_NUMEROHERMANOS,INFE_POSICIONENHERMANOS,INFE_NUMMIEMBROSTRABAJA
2023-01-15,ACTIVO,45,5,3.8,INGENIERIA DE SISTEMAS,DIURNA,M,2000-05-20,BUCARAMANGA,3,1,SI,VIVOS Y CONVIVEN,4,2,1,2
2022-08-10,ACTIVO,60,7,4.2,TECNOLOGIA EN DESARROLLO DE SISTEMAS INFORMATICOS,NOCTURNA,F,1999-11-15,FLORIDABLANCA,2,0,NO,VIVOS Y SEPARADOS,3,1,1,1
```

---

## 📁 Estructura de Archivos del Sistema

El sistema guardará automáticamente:

```
archivos_procesados/
├── base/           ← Tu archivo original (copia de seguridad)
├── limpio/         ← Archivo después de limpieza (26 columnas)
└── resultados/     ← Predicciones del modelo (ID, resultado, probabilidad)
```

---

## ⚠️ Errores Comunes

| Error | Causa | Solución |
|-------|-------|----------|
| "Columna X no encontrada" | Falta una columna obligatoria | Verifica que todas las 18 columnas estén presentes |
| "Tipo de dato incorrecto" | Fecha o número mal formateado | Revisa el formato de fechas (YYYY-MM-DD) y números |
| "Archivo vacío después de limpieza" | Programas no válidos | Solo usa INGENIERIA DE SISTEMAS o TECNOLOGIA EN... |
| "Demasiados valores faltantes" | Muchas celdas vacías | Completa al menos las columnas principales |

---

## 💡 Recomendaciones

1. ✅ **Descarga la plantilla** desde el dashboard antes de crear tu archivo
2. ✅ **Copia y pega** tus datos en la plantilla descargada
3. ✅ **Verifica** que los nombres de columnas coincidan exactamente (mayúsculas, guiones bajos)
4. ✅ **Revisa** el log del dashboard después de subir para detectar advertencias
5. ✅ **Guarda** una copia de tu archivo original antes de subirlo

---

## 📞 Soporte

Si tienes dudas sobre el formato o encuentras errores, revisa el **panel de logs** en el dashboard que muestra cada paso del proceso de limpieza.
