#!/usr/bin/env python3
# ruff: noqa: ANN401, CPY001, D103, DOC201, E501, I001, PIE808, PLR0914, PLR0915, SLF001, T201
"""Build the advisor-facing final experiment report from accepted local artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont


BLUE = "2E6F9E"
ORANGE = "D97934"
INK = "1F2933"
MUTED = "52606D"
LIGHT = "E8EDF2"
NORMAL_LABEL = "VAE no equivariante"
SO2_LABEL = "VAE equivariante SO(2)"
HEATMAP_LIGHT_TEXT_THRESHOLD = 0.55
FIXED25_COUNT = 25
FIRST_ORBIT_SHEET_COUNT = 15


SPANISH_REPLACEMENTS = {
    "Aun": "Aún",
    "Area": "Área",
    "Atribucion": "Atribución",
    "Comparacion": "Comparación",
    "Conclusion": "Conclusión",
    "Cuadricula": "Cuadrícula",
    "Descripcion": "Descripción",
    "Diagnostico": "Diagnóstico",
    "Donde": "Dónde",
    "Entropia": "Entropía",
    "Definicion": "Definición",
    "Desviacion": "Desviación",
    "Dimension": "Dimensión",
    "Direccion": "Dirección",
    "Grafica": "Gráfica",
    "Graficas": "Gráficas",
    "Interpretacion": "Interpretación",
    "Implementacion": "Implementación",
    "Metricas": "Métricas",
    "Metrica": "Métrica",
    "Parametros": "Parámetros",
    "Poblacion": "Población",
    "Orbita": "Órbita",
    "Perdida": "Pérdida",
    "Reconstruccion": "Reconstrucción",
    "Razon": "Razón",
    "Relacion": "Relación",
    "Seccion": "Sección",
    "Senal": "Señal",
    "Senales": "Señales",
    "Si": "Sí",
    "Simetria": "Simetría",
    "Sintesis": "Síntesis",
    "Tamano": "Tamaño",
    "Validacion": "Validación",
    "Visualizacion": "Visualización",
    "amplio": "amplió",
    "agregacion": "agregación",
    "analisis": "análisis",
    "almaceno": "almacenó",
    "angulos": "ángulos",
    "aritmetica": "aritmética",
    "acompana": "acompaña",
    "aplico": "aplicó",
    "automaticamente": "automáticamente",
    "biologica": "biológica",
    "area": "área",
    "arquitectonica": "arquitectónica",
    "arquitectonico": "arquitectónico",
    "asi": "así",
    "atencion": "atención",
    "atribucion": "atribución",
    "clasificacion": "clasificación",
    "ciclico": "cíclico",
    "ciclicos": "cíclicos",
    "compactacion": "compactación",
    "comparacion": "comparación",
    "compensacion": "compensación",
    "conclusion": "conclusión",
    "contribucion": "contribución",
    "cientificamente": "científicamente",
    "cuadratico": "cuadrático",
    "diagnostico": "diagnóstico",
    "diagnosticos": "diagnósticos",
    "descripcion": "descripción",
    "deberia": "debería",
    "despues": "después",
    "desviacion": "desviación",
    "direccion": "dirección",
    "dispersion": "dispersión",
    "entropia": "entropía",
    "estandar": "estándar",
    "establecio": "estableció",
    "estan": "están",
    "estratifico": "estratificó",
    "estadistica": "estadística",
    "evaluacion": "evaluación",
    "evaluo": "evaluó",
    "explicacion": "explicación",
    "explicita": "explícita",
    "especifica": "específica",
    "especificas": "específicas",
    "entreno": "entrenó",
    "experimentacion": "experimentación",
    "fragiles": "frágiles",
    "fluctuan": "fluctúan",
    "grafica": "gráfica",
    "graficas": "gráficas",
    "histopatologia": "histopatología",
    "histopatologico": "histopatológico",
    "guardo": "guardó",
    "incluyo": "incluyó",
    "inspeccion": "inspección",
    "implementacion": "implementación",
    "interpretacion": "interpretación",
    "laminas": "láminas",
    "limitacion": "limitación",
    "linea": "línea",
    "maximo": "máximo",
    "mas": "más",
    "mejoro": "mejoró",
    "metrica": "métrica",
    "metricas": "métricas",
    "mostro": "mostró",
    "ningun": "ningún",
    "numero": "número",
    "numerica": "numérica",
    "normalizacion": "normalización",
    "numericamente": "numéricamente",
    "observo": "observó",
    "organizacion": "organización",
    "oraculo": "oráculo",
    "orbita": "órbita",
    "parametrica": "paramétrica",
    "parametros": "parámetros",
    "patron": "patrón",
    "pequena": "pequeña",
    "pequeno": "pequeño",
    "perdida": "pérdida",
    "periodica": "periódica",
    "pixel": "píxel",
    "pixeles": "píxeles",
    "poblacion": "población",
    "proposito": "propósito",
    "probabilistica": "probabilística",
    "presentacion": "presentación",
    "prediccion": "predicción",
    "reconstruccion": "reconstrucción",
    "representacion": "representación",
    "recupero": "recuperó",
    "razon": "razón",
    "resumenes": "resúmenes",
    "reduccion": "reducción",
    "regimenes": "regímenes",
    "restriccion": "restricción",
    "rotacion": "rotación",
    "segun": "según",
    "seleccion": "selección",
    "senal": "señal",
    "senales": "señales",
    "simetria": "simetría",
    "simultaneo": "simultáneo",
    "simultaneos": "simultáneos",
    "simultaneamente": "simultáneamente",
    "solicito": "solicitó",
    "sintesis": "síntesis",
    "estadisticas": "estadísticas",
    "tamano": "tamaño",
    "tambien": "también",
    "unica": "única",
    "unico": "único",
    "uso": "usó",
    "validacion": "validación",
    "utiles": "útiles",
    "variacion": "variación",
    "visualizacion": "visualización",
    "verificacion": "verificación",
}


def es(text: str) -> str:
    """Apply conservative Spanish accent restoration to display text."""
    for source, target in SPANISH_REPLACEMENTS.items():
        text = re.sub(rf"\b{re.escape(source)}\b", target, text)
    return text


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_orbit_population(  # noqa: C901, PLR0912
    document: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    """Reject stale or internally inconsistent dense-orbit summaries.

    Raises:
        TypeError: A required mapping has the wrong type.
        ValueError: An identity, value, median, or paired count is inconsistent.

    """
    if document.get("schema") != "spec0038.fixed25_dense_orbit_population.v1":
        message = "La evidencia poblacional de rotación usa un esquema inesperado"
        raise ValueError(message)
    if document.get("angles_degrees") != list(range(360)):
        message = "La evidencia poblacional de rotación no contiene 0°--359°"
        raise ValueError(message)

    metric_names = (
        "local_linearity_ratio",
        "step_size_cv",
        "pca_explained_variance",
    )
    values_by_branch: dict[str, dict[str, list[float]]] = {}
    medians_by_branch: dict[str, dict[str, float]] = {}
    for branch in ("normal", "so2"):
        section = document.get(branch)
        if not isinstance(section, dict):
            message = f"Falta la rama {branch} en la evidencia poblacional"
            raise TypeError(message)
        values_by_branch[branch] = {}
        for metric_name in metric_names:
            values = section.get(metric_name)
            if not isinstance(values, list) or len(values) != FIXED25_COUNT:
                message = f"{branch}.{metric_name} no contiene 25 valores"
                raise ValueError(message)
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            ):
                message = f"{branch}.{metric_name} contiene valores no finitos"
                raise ValueError(message)
            values_by_branch[branch][metric_name] = [float(value) for value in values]

        stored_medians = section.get("median")
        if not isinstance(stored_medians, dict):
            message = f"Faltan medianas para la rama {branch}"
            raise TypeError(message)
        medians_by_branch[branch] = {}
        for metric_name in metric_names:
            stored = stored_medians.get(metric_name)
            if (
                isinstance(stored, bool)
                or not isinstance(stored, (int, float))
                or not math.isfinite(float(stored))
            ):
                message = f"Mediana inválida para {branch}.{metric_name}"
                raise ValueError(message)
            recomputed = float(median(values_by_branch[branch][metric_name]))
            if not math.isclose(float(stored), recomputed, rel_tol=0.0, abs_tol=1e-12):
                message = f"Mediana inconsistente para {branch}.{metric_name}"
                raise ValueError(message)
            medians_by_branch[branch][metric_name] = recomputed

    normal_values = values_by_branch["normal"]
    so2_values = values_by_branch["so2"]
    expected_counts = {
        "so2_lower_local_linearity_ratio": sum(
            so2 < normal
            for normal, so2 in zip(
                normal_values["local_linearity_ratio"],
                so2_values["local_linearity_ratio"],
                strict=True,
            )
        ),
        "so2_lower_step_size_cv": sum(
            so2 < normal
            for normal, so2 in zip(
                normal_values["step_size_cv"],
                so2_values["step_size_cv"],
                strict=True,
            )
        ),
        "so2_higher_pca_explained_variance": sum(
            so2 > normal
            for normal, so2 in zip(
                normal_values["pca_explained_variance"],
                so2_values["pca_explained_variance"],
                strict=True,
            )
        ),
    }
    if document.get("paired_favorable_counts") != expected_counts:
        message = "Los conteos pareados de las órbitas no coinciden con sus 25 valores"
        raise ValueError(message)

    source = document.get("source")
    patch_hashes = source.get("patch_sha256") if isinstance(source, dict) else None
    if (
        not isinstance(patch_hashes, list)
        or len(patch_hashes) != FIXED25_COUNT
        or len(set(patch_hashes)) != FIXED25_COUNT
        or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in patch_hashes
        )
    ):
        message = "La identidad de los 25 parches no es válida o no es única"
        raise ValueError(message)
    for key in ("normal_checkpoint_sha256", "so2_checkpoint_sha256"):
        value = source.get(key) if isinstance(source, dict) else None
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            message = f"Hash de checkpoint inválido: {key}"
            raise ValueError(message)

    return medians_by_branch["normal"], medians_by_branch["so2"], expected_counts


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = [
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def draw_orbit_population_report_sheets(
    source: Path,
    first_output: Path,
    second_output: Path,
) -> None:
    """Reflow the accepted 25-patch orbit grid into two readable report sheets.

    Raises:
        ValueError: The immutable source image has an unexpected size.

    """
    with Image.open(source) as opened:
        source_image = opened.convert("RGB")
    if source_image.size != (3000, 2400):
        message = f"Tamaño inesperado para el mosaico 25x360: {source_image.size}"
        raise ValueError(message)

    def render(indices: range, output: Path, subtitle: str) -> None:
        canvas = Image.new("RGB", (1800, 1680), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (70, 24),
            "Órbitas latentes de los 25 parches fijos",
            fill="#1F2933",
            font=font(38, bold=True),
        )
        draw.text((70, 76), subtitle, fill="#52606D", font=font(25))
        is_first_sheet = len(indices) == FIRST_ORBIT_SHEET_COUNT
        columns = 3 if is_first_sheet else 2
        x_positions = [70, 640, 1210] if is_first_sheet else [350, 930]
        label_font = font(30, bold=True)
        for position, patch_index in enumerate(indices):
            source_row, source_column = divmod(patch_index, 5)
            source_left = 65 + source_column * 585
            source_top = 170 + source_row * 330
            tile = source_image.crop(
                (source_left, source_top, source_left + 520, source_top + 285),
            )
            tile_draw = ImageDraw.Draw(tile)
            tile_draw.rectangle((0, 0, 520, 40), fill="white")
            tile_draw.text(
                (8, 3),
                f"P{patch_index:02d} · normal",
                fill=f"#{BLUE}",
                font=label_font,
            )
            tile_draw.text(
                (278, 3),
                f"P{patch_index:02d} · SO(2)",
                fill=f"#{ORANGE}",
                font=label_font,
            )
            destination_column = position % columns
            destination_row = position // columns
            canvas.paste(
                tile,
                (x_positions[destination_column], 130 + destination_row * 300),
            )
        draw.text(
            (70, 1640),
            "Cada curva contiene 360 posiciones (0°--359°); la escala es isótropa y propia de cada panel.",
            fill="#52606D",
            font=font(27),
        )
        canvas.save(output, dpi=(180, 180))

    render(range(0, 15), first_output, "Parte A · parches 00--14 · muestreo cada 1°")
    render(range(15, 25), second_output, "Parte B · parches 15--24 · muestreo cada 1°")


def draw_mil_chart(normal: dict[str, Any], so2: dict[str, Any], output: Path) -> None:
    width, height = 1500, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, axis_font, tick_font = font(38, bold=True), font(25), font(22)
    draw.text(
        (80, 38),
        es("Diagnostico WSI en el test sellado"),
        fill="#1F2933",
        font=title_font,
    )
    left, top, right, bottom = 130, 150, 1430, 610
    draw.line((left, top, left, bottom), fill="#8392A5", width=3)
    draw.line((left, bottom, right, bottom), fill="#8392A5", width=3)
    for i in range(0, 8):
        value = i / 10
        y = bottom - int(value / 0.7 * (bottom - top))
        draw.line((left, y, right, y), fill="#E5E9EF", width=2)
        draw.text((54, y - 13), f"{value:.1f}", fill="#52606D", font=tick_font)
    metrics = [
        ("F1 macro", normal["metrics"]["macro_f1"], so2["metrics"]["macro_f1"]),
        (
            "Exactitud\nbalanceada",
            normal["metrics"]["balanced_accuracy"],
            so2["metrics"]["balanced_accuracy"],
        ),
        ("Exactitud", normal["metrics"]["accuracy"], so2["metrics"]["accuracy"]),
    ]
    centers = [360, 790, 1220]
    bar_width = 110
    for center, (label, nval, sval) in zip(centers, metrics, strict=True):
        for x, val, color in ((center - 125, nval, BLUE), (center + 20, sval, ORANGE)):
            y = bottom - int(val / 0.7 * (bottom - top))
            draw.rounded_rectangle(
                (x, y, x + bar_width, bottom),
                radius=8,
                fill=f"#{color}",
            )
            text = f"{val:.3f}"
            bbox = draw.textbbox((0, 0), text, font=tick_font)
            draw.text(
                (x + (bar_width - (bbox[2] - bbox[0])) / 2, y - 34),
                text,
                fill="#1F2933",
                font=tick_font,
            )
        for line_no, line in enumerate(label.split("\n")):
            bbox = draw.textbbox((0, 0), line, font=axis_font)
            draw.text(
                (center - (bbox[2] - bbox[0]) / 2, bottom + 22 + line_no * 28),
                es(line),
                fill="#1F2933",
                font=axis_font,
            )
    legend_y = 700
    draw.rounded_rectangle(
        (395, legend_y, 435, legend_y + 24),
        radius=4,
        fill=f"#{BLUE}",
    )
    draw.text((450, legend_y - 4), NORMAL_LABEL, fill="#1F2933", font=tick_font)
    draw.rounded_rectangle(
        (850, legend_y, 890, legend_y + 24),
        radius=4,
        fill=f"#{ORANGE}",
    )
    draw.text((905, legend_y - 4), SO2_LABEL, fill="#1F2933", font=tick_font)
    image.save(output, dpi=(180, 180))


def draw_tissue_chart(table: dict[str, Any], output: Path) -> None:
    width, height = 1500, 800
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, axis_font, tick_font = font(38, bold=True), font(25), font(22)
    draw.text(
        (80, 35),
        "Eficiencia de etiquetas en tejido: F1 macro",
        fill="#1F2933",
        font=title_font,
    )
    left, top, right, bottom = 135, 150, 1435, 635
    draw.line((left, top, left, bottom), fill="#8392A5", width=3)
    draw.line((left, bottom, right, bottom), fill="#8392A5", width=3)
    y_min, y_max = 0.45, 0.80
    for value in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        y = bottom - int((value - y_min) / (y_max - y_min) * (bottom - top))
        draw.line((left, y, right, y), fill="#E5E9EF", width=2)
        draw.text((55, y - 13), f"{value:.2f}", fill="#52606D", font=tick_font)
    rows = table["rows"]
    xs = [
        left + 95 + i * ((right - left - 190) / (len(rows) - 1))
        for i in range(len(rows))
    ]

    def y_coord(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    for key, color in (("normal_vae", BLUE), ("so2_vae", ORANGE)):
        points = [
            (x, y_coord(row[key]["metrics"]["macro_f1"]))
            for x, row in zip(xs, rows, strict=True)
        ]
        draw.line(points, fill=f"#{color}", width=7, joint="curve")
        for x, y in points:
            marker = (x - 10, y - 10, x + 10, y + 10)
            if key == "normal_vae":
                draw.ellipse(marker, fill=f"#{color}", outline="white", width=3)
            else:
                draw.rectangle(marker, fill=f"#{color}", outline="white", width=3)
    for x, row in zip(xs, rows, strict=True):
        label = f"{row['labels_per_class']:,}".replace(",", ".")
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text(
            (x - (bbox[2] - bbox[0]) / 2, bottom + 20),
            label,
            fill="#1F2933",
            font=tick_font,
        )
    sig = rows[1]
    sx = xs[1]
    sy = (
        min(
            y_coord(sig["normal_vae"]["metrics"]["macro_f1"]),
            y_coord(sig["so2_vae"]["metrics"]["macro_f1"]),
        )
        - 38
    )
    draw.text((sx - 10, sy), "*", fill="#7C3AED", font=font(38, bold=True))
    draw.text((left + 455, 690), "Etiquetas por clase", fill="#1F2933", font=axis_font)
    draw.rounded_rectangle((845, 683, 885, 707), radius=4, fill=f"#{BLUE}")
    draw.text((900, 679), NORMAL_LABEL, fill="#1F2933", font=tick_font)
    draw.rounded_rectangle((845, 728, 885, 752), radius=4, fill=f"#{ORANGE}")
    draw.text((900, 724), SO2_LABEL, fill="#1F2933", font=tick_font)
    draw.text(
        (150, 715),
        es("* intervalo simultaneo del contraste excluye cero"),
        fill="#7C3AED",
        font=tick_font,
    )
    image.save(output, dpi=(180, 180))


def draw_wsi_class_chart(  # noqa: C901
    normal: dict[str, Any],
    so2: dict[str, Any],
    output: Path,
) -> None:
    width, height = 1800, 1450
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(40, bold=True)
    subtitle_font = font(25)
    axis_font = font(24, bold=True)
    tick_font = font(21)
    cell_font = font(19, bold=True)
    class_labels = [item["diagnosis"] for item in normal["metrics"]["per_class"]]
    supports = [item["support"] for item in normal["metrics"]["per_class"]]

    draw.text(
        (70, 35),
        "Diagnóstico WSI: errores y F1 por clase",
        fill="#1F2933",
        font=title_font,
    )
    draw.text(
        (70, 90),
        "Test sellado: n=23 WSI; las celdas muestran conteo y porcentaje dentro de la clase real",
        fill="#52606D",
        font=subtitle_font,
    )

    def heat_color(value: float) -> str:
        start = (239, 246, 251)
        end = (46, 111, 158)
        rgb = tuple(round(a + value * (b - a)) for a, b in zip(start, end, strict=True))
        return "#" + "".join(f"{channel:02X}" for channel in rgb)

    def draw_confusion(metrics: dict[str, Any], x0: int, title: str) -> None:
        matrix = metrics["confusion_matrix"]
        cell = 92
        y0 = 260
        draw.text((x0 + 95, 150), title, fill="#1F2933", font=axis_font)
        draw.text((x0 + 150, 190), "Predicción", fill="#52606D", font=tick_font)
        for col, label in enumerate(class_labels):
            bbox = draw.textbbox((0, 0), label, font=tick_font)
            draw.text(
                (x0 + col * cell + (cell - (bbox[2] - bbox[0])) / 2, y0 - 34),
                label,
                fill="#1F2933",
                font=tick_font,
            )
        for row, (values, support) in enumerate(zip(matrix, supports, strict=True)):
            label = f"{class_labels[row]} (n={support})"
            bbox = draw.textbbox((0, 0), label, font=tick_font)
            draw.text(
                (x0 - (bbox[2] - bbox[0]) - 14, y0 + row * cell + 32),
                label,
                fill="#1F2933",
                font=tick_font,
            )
            for col, count in enumerate(values):
                share = count / support if support else 0.0
                left = x0 + col * cell
                top = y0 + row * cell
                draw.rectangle(
                    (left, top, left + cell, top + cell),
                    fill=heat_color(share),
                    outline="#FFFFFF",
                    width=3,
                )
                label = f"{count}\n{share:.0%}"
                color = "white" if share >= HEATMAP_LIGHT_TEXT_THRESHOLD else "#1F2933"
                for line_no, line in enumerate(label.split("\n")):
                    bbox = draw.textbbox((0, 0), line, font=cell_font)
                    draw.text(
                        (
                            left + (cell - (bbox[2] - bbox[0])) / 2,
                            top + 20 + line_no * 25,
                        ),
                        line,
                        fill=color,
                        font=cell_font,
                    )

    draw_confusion(normal["metrics"], 235, NORMAL_LABEL)
    draw_confusion(so2["metrics"], 1120, SO2_LABEL)

    left, top, right, bottom = 120, 880, 1710, 1280
    draw.text((70, 785), "F1 por clase", fill="#1F2933", font=axis_font)
    draw.line((left, top, left, bottom), fill="#8392A5", width=3)
    draw.line((left, bottom, right, bottom), fill="#8392A5", width=3)
    for value in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = bottom - int(value * (bottom - top))
        draw.line((left, y, right, y), fill="#E5E9EF", width=2)
        draw.text((55, y - 12), f"{value:.2f}", fill="#52606D", font=tick_font)
    normal_f1 = [item["f1"] for item in normal["metrics"]["per_class"]]
    so2_f1 = [item["f1"] for item in so2["metrics"]["per_class"]]
    group_width = (right - left) / len(class_labels)
    bar_width = 92
    for index, (label, support, nval, sval) in enumerate(
        zip(class_labels, supports, normal_f1, so2_f1, strict=True),
    ):
        center = left + group_width * (index + 0.5)
        for x, value, color in (
            (center - bar_width - 8, nval, BLUE),
            (center + 8, sval, ORANGE),
        ):
            y = bottom - value * (bottom - top)
            draw.rounded_rectangle(
                (x, y, x + bar_width, bottom),
                radius=7,
                fill=f"#{color}",
            )
            text = f"{value:.2f}"
            bbox = draw.textbbox((0, 0), text, font=tick_font)
            draw.text(
                (x + (bar_width - (bbox[2] - bbox[0])) / 2, y - 30),
                text,
                fill="#1F2933",
                font=tick_font,
            )
        group_label = f"{label}\n(n={support})"
        for line_no, line in enumerate(group_label.split("\n")):
            bbox = draw.textbbox((0, 0), line, font=tick_font)
            draw.text(
                (center - (bbox[2] - bbox[0]) / 2, bottom + 18 + line_no * 27),
                line,
                fill="#1F2933",
                font=tick_font,
            )
    draw.rounded_rectangle((520, 1380, 560, 1404), radius=4, fill=f"#{BLUE}")
    draw.text((575, 1376), NORMAL_LABEL, fill="#1F2933", font=tick_font)
    draw.rounded_rectangle((1040, 1380, 1080, 1404), radius=4, fill=f"#{ORANGE}")
    draw.text((1095, 1376), SO2_LABEL, fill="#1F2933", font=tick_font)
    image.save(output, dpi=(180, 180))


def draw_tissue_class_chart(table: dict[str, Any], output: Path) -> None:
    width, height = 1800, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(40, bold=True)
    axis_font = font(23, bold=True)
    tick_font = font(19)
    draw.text(
        (70, 35),
        "Eficiencia de etiquetas: F1 por tipo de tejido",
        fill="#1F2933",
        font=title_font,
    )
    draw.text(
        (70, 90),
        "Test sellado: 31.572 parches de 23 WSI; puntos descriptivos por presupuesto",
        fill="#52606D",
        font=font(24),
    )
    rows = table["rows"]
    tissues = [
        ("tumor", "Tumor", "n=21.796 parches; 23 WSI"),
        ("stroma", "Estroma", "n=8.500 parches; 21 WSI"),
        ("necrosis", "Necrosis", "n=1.276 parches; 5 WSI"),
    ]

    for panel, (key, label, support) in enumerate(tissues):
        panel_left = 75 + panel * 575
        left, top, right, bottom = panel_left + 60, 240, panel_left + 515, 700
        draw.text((panel_left + 150, 160), label, fill="#1F2933", font=axis_font)
        draw.text((panel_left + 110, 195), support, fill="#52606D", font=tick_font)
        draw.line((left, top, left, bottom), fill="#8392A5", width=3)
        draw.line((left, bottom, right, bottom), fill="#8392A5", width=3)
        for value in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y = bottom - int(value * (bottom - top))
            draw.line((left, y, right, y), fill="#E5E9EF", width=2)
            draw.text(
                (left - 48, y - 10),
                f"{value:.2f}",
                fill="#52606D",
                font=tick_font,
            )
        xs = [left + i * (right - left) / (len(rows) - 1) for i in range(len(rows))]
        for branch, color in (("normal_vae", BLUE), ("so2_vae", ORANGE)):
            values = []
            for row in rows:
                by_tissue = {
                    item["tissue"]: item["f1"]
                    for item in row[branch]["metrics"]["per_class"]
                }
                values.append(by_tissue[key])
            points = [
                (x, bottom - value * (bottom - top))
                for x, value in zip(xs, values, strict=True)
            ]
            draw.line(points, fill=f"#{color}", width=6, joint="curve")
            for x, y in points:
                marker = (x - 8, y - 8, x + 8, y + 8)
                if branch == "normal_vae":
                    draw.ellipse(marker, fill=f"#{color}", outline="white", width=2)
                else:
                    draw.rectangle(marker, fill=f"#{color}", outline="white", width=2)
        for x, row in zip(xs, rows, strict=True):
            value = f"{row['labels_per_class']:,}".replace(",", ".")
            bbox = draw.textbbox((0, 0), value, font=tick_font)
            draw.text(
                (x - (bbox[2] - bbox[0]) / 2, bottom + 18),
                value,
                fill="#1F2933",
                font=tick_font,
            )
    draw.text((710, 765), "Etiquetas por clase", fill="#1F2933", font=axis_font)
    draw.rounded_rectangle((490, 835, 530, 859), radius=4, fill=f"#{BLUE}")
    draw.text((545, 831), NORMAL_LABEL, fill="#1F2933", font=tick_font)
    draw.rounded_rectangle((1030, 835, 1070, 859), radius=4, fill=f"#{ORANGE}")
    draw.text((1085, 831), SO2_LABEL, fill="#1F2933", font=tick_font)
    image.save(output, dpi=(180, 180))


def draw_supervised_development_chart(  # noqa: C901
    mil_train: dict[str, list[dict[str, str]]],
    mil_validation: dict[str, list[dict[str, str]]],
    mil_summaries: dict[str, dict[str, Any]],
    tissue_validation: list[dict[str, float]],
    output: Path,
) -> None:
    width, height = 2200, 1200
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(46, bold=True)
    axis_font = font(30, bold=True)
    tick_font = font(26)
    draw.text(
        (70, 35),
        "Desarrollo de los clasificadores supervisados",
        fill="#1F2933",
        font=title_font,
    )
    draw.text(
        (70, 90),
        "F1 de selección y brecha train-validación; el test sellado permanece separado",
        fill="#52606D",
        font=font(28),
    )

    def y_at(
        value: float,
        top: int,
        bottom: int,
        minimum: float,
        maximum: float,
    ) -> float:
        return bottom - (value - minimum) / (maximum - minimum) * (bottom - top)

    def x_at(boundary: float, left: int, right: int, maximum: float) -> float:
        return left + boundary / maximum * (right - left)

    def axes(  # noqa: PLR0913, PLR0917
        left: int,
        top: int,
        right: int,
        bottom: int,
        title: str,
        y_values: list[float],
    ) -> None:
        draw.text((left, top - 62), title, fill="#1F2933", font=axis_font)
        draw.line((left, top, left, bottom), fill="#8392A5", width=3)
        draw.line((left, bottom, right, bottom), fill="#8392A5", width=3)
        y_min, y_max = min(y_values), max(y_values)
        for value in y_values:
            y = y_at(value, top, bottom, y_min, y_max)
            draw.line((left, y, right, y), fill="#E5E9EF", width=2)
            draw.text(
                (left - 58, y - 10),
                f"{value:.2f}",
                fill="#52606D",
                font=tick_font,
            )

    def dashed_line(
        points: list[tuple[float, float]],
        color: str,
        width: int = 4,
        dash: int = 18,
        gap: int = 10,
    ) -> None:
        for (x1, y1), (x2, y2) in pairwise(points):
            distance = max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1.0)
            cursor = 0.0
            while cursor < distance:
                segment_end = min(cursor + dash, distance)
                start_fraction = cursor / distance
                end_fraction = segment_end / distance
                draw.line(
                    (
                        x1 + (x2 - x1) * start_fraction,
                        y1 + (y2 - y1) * start_fraction,
                        x1 + (x2 - x1) * end_fraction,
                        y1 + (y2 - y1) * end_fraction,
                    ),
                    fill=color,
                    width=width,
                )
                cursor += dash + gap

    max_boundary = max(
        float(row["boundary"]) for rows in mil_validation.values() for row in rows
    )
    left, top, right, bottom = 105, 230, 1030, 630
    axes(
        left,
        top,
        right,
        bottom,
        "MIL WSI: F1 macro de validación",
        [0.0, 0.25, 0.5, 0.75, 1.0],
    )
    for branch, color in (("normal_vae", BLUE), ("so2_vae", ORANGE)):
        points = [
            (
                x_at(float(row["boundary"]), left, right, max_boundary),
                y_at(float(row["macro_f1"]), top, bottom, 0.0, 1.0),
            )
            for row in mil_validation[branch]
        ]
        draw.line(points, fill=f"#{color}", width=4, joint="curve")
        summary = mil_summaries[branch]
        best_x = x_at(summary["best_boundary"], left, right, max_boundary)
        best_y = y_at(
            summary["best_metrics"]["macro_f1"],
            top,
            bottom,
            0.0,
            1.0,
        )
        draw.ellipse(
            (best_x - 11, best_y - 11, best_x + 11, best_y + 11),
            fill=f"#{color}",
            outline="white",
            width=3,
        )
        label = f"{summary['best_metrics']['macro_f1']:.3f} @ {summary['best_boundary']:,}".replace(
            ",",
            ".",
        )
        draw.text((best_x + 12, best_y - 12), label, fill=f"#{color}", font=tick_font)
    for value in [0, 2000, 4000, 6000, 8000]:
        x = x_at(value, left, right, max_boundary)
        draw.text(
            (x - 25, bottom + 18),
            f"{value // 1000}k",
            fill="#52606D",
            font=tick_font,
        )
    draw.text(
        (left + 365, bottom + 58),
        "Actualización",
        fill="#52606D",
        font=tick_font,
    )

    left2, top2, right2, bottom2 = 1190, 230, 2115, 630
    axes(
        left2,
        top2,
        right2,
        bottom2,
        "MIL WSI: entropía cruzada train-validación",
        [0.5, 1.0, 1.5, 2.0, 2.5],
    )
    for branch, color in (("normal_vae", BLUE), ("so2_vae", ORANGE)):
        previous_boundary = 0
        train_points: list[tuple[float, float]] = []
        validation_points: list[tuple[float, float]] = []
        for validation_row in mil_validation[branch]:
            boundary = int(validation_row["boundary"])
            losses = [
                float(row["unweighted_loss"])
                for row in mil_train[branch]
                if previous_boundary < int(row["committed_update"]) <= boundary
            ]
            if not losses:
                message = f"Missing MIL train-loss window ending at {boundary}"
                raise ValueError(message)
            train_points.append(
                (
                    x_at(boundary, left2, right2, max_boundary),
                    y_at(sum(losses) / len(losses), top2, bottom2, 0.5, 2.5),
                ),
            )
            validation_points.append(
                (
                    x_at(boundary, left2, right2, max_boundary),
                    y_at(
                        float(validation_row["mean_ce"]),
                        top2,
                        bottom2,
                        0.5,
                        2.5,
                    ),
                ),
            )
            previous_boundary = boundary
        draw.line(train_points, fill=f"#{color}", width=5, joint="curve")
        dashed_line(validation_points, f"#{color}")
    for value in [0, 2000, 4000, 6000, 8000]:
        x = x_at(value, left2, right2, max_boundary)
        draw.text(
            (x - 25, bottom2 + 18),
            f"{value // 1000}k",
            fill="#52606D",
            font=tick_font,
        )
    draw.text(
        (left2 + 365, bottom2 + 58),
        "Actualización",
        fill="#52606D",
        font=tick_font,
    )

    draw.rounded_rectangle((500, 730, 540, 754), radius=4, fill=f"#{BLUE}")
    draw.text((555, 726), NORMAL_LABEL, fill="#1F2933", font=tick_font)
    draw.rounded_rectangle((1090, 730, 1130, 754), radius=4, fill=f"#{ORANGE}")
    draw.text((1145, 726), SO2_LABEL, fill="#1F2933", font=tick_font)
    draw.line((590, 790, 650, 790), fill="#52606D", width=5)
    draw.text(
        (665, 780),
        "train online (media por ventana)",
        fill="#1F2933",
        font=tick_font,
    )
    dashed_line([(1260, 790), (1320, 790)], "#52606D", width=4)
    draw.text((1335, 780), "validación", fill="#1F2933", font=tick_font)

    left3, top3, right3, bottom3 = 420, 920, 1780, 1060
    axes(
        left3,
        top3,
        right3,
        bottom3,
        "Tejido: mejor F1 macro de validación",
        [0.0, 0.5, 1.0],
    )
    xs = [
        left3 + i * (right3 - left3) / (len(tissue_validation) - 1)
        for i in range(len(tissue_validation))
    ]
    for key, color in (("normal_vae", BLUE), ("so2_vae", ORANGE)):
        points = [
            (x, y_at(row[key], top3, bottom3, 0.0, 1.0))
            for x, row in zip(xs, tissue_validation, strict=True)
        ]
        draw.line(points, fill=f"#{color}", width=6, joint="curve")
        for x, y in points:
            if key == "normal_vae":
                draw.ellipse(
                    (x - 8, y - 8, x + 8, y + 8),
                    fill=f"#{color}",
                    outline="white",
                    width=2,
                )
            else:
                draw.rectangle(
                    (x - 8, y - 8, x + 8, y + 8),
                    fill=f"#{color}",
                    outline="white",
                    width=2,
                )
    for x, row in zip(xs, tissue_validation, strict=True):
        label = f"{int(row['budget']):,}".replace(",", ".")
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text(
            (x - (bbox[2] - bbox[0]) / 2, bottom3 + 18),
            label,
            fill="#52606D",
            font=tick_font,
        )
    draw.text(
        (left3 + 540, bottom3 + 58),
        "Etiquetas por clase",
        fill="#52606D",
        font=tick_font,
    )
    image.save(output, dpi=(180, 180))


def set_cell_shading(cell: Any, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_border(cell: Any, color: str = "CDD4DC", size: str = "6") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_borders = tc_pr.first_child_found_in("w:tcBorders")
    if tc_borders is None:
        tc_borders = OxmlElement("w:tcBorders")
        tc_pr.append(tc_borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        element = tc_borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            tc_borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:color"), color)


def set_cell_margins(
    cell: Any,
    top: int = 80,
    start: int = 90,
    bottom: int = 80,
    end: int = 90,
) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for key, value in (
        ("top", top),
        ("start", start),
        ("bottom", bottom),
        ("end", end),
    ):
        element = tc_mar.find(qn(f"w:{key}"))
        if element is None:
            element = OxmlElement(f"w:{key}")
            tc_mar.append(element)
        element.set(qn("w:w"), str(value))
        element.set(qn("w:type"), "dxa")


def set_cell_width(cell: Any, inches: float) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(int(inches * 1440)))
    tc_w.set(qn("w:type"), "dxa")


def set_repeat_table_header(row: Any) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_row_cant_split(row: Any) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def set_paragraph_keep(
    paragraph: Any,
    *,
    keep_next: bool = False,
    keep_lines: bool = True,
) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    if keep_next:
        p_pr.append(OxmlElement("w:keepNext"))
    if keep_lines:
        p_pr.append(OxmlElement("w:keepLines"))


def add_page_number(paragraph: Any) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fld_char = OxmlElement("w:fldChar")
    fld_char.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char, instr_text, fld_char2])


def set_picture_alt(paragraph: Any, description: str) -> None:
    doc_prs = paragraph._p.xpath(".//wp:docPr")
    if doc_prs:
        doc_prs[-1].set("descr", description)
        doc_prs[-1].set("title", description[:80])


def add_table(
    doc: Document,
    headers: list[str],
    rows: list[list[str]],
    *,
    widths: list[float] | None = None,
    font_size: float = 8.7,
) -> Any:
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.style = "Table Grid"
    header = table.rows[0]
    set_repeat_table_header(header)
    for i, label in enumerate(headers):
        cell = header.cells[i]
        set_cell_shading(cell, INK)
        set_cell_border(cell)
        set_cell_margins(cell)
        if widths:
            set_cell_width(cell, widths[i])
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(es(label))
        run.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        run.font.size = Pt(font_size)
    for row_no, values in enumerate(rows):
        row = table.add_row()
        set_row_cant_split(row)
        for i, value in enumerate(values):
            cell = row.cells[i]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_border(cell)
            set_cell_margins(cell)
            if widths:
                set_cell_width(cell, widths[i])
            if row_no % 2:
                set_cell_shading(cell, "F5F7F9")
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
            )
            run = p.add_run(es(str(value)))
            run.font.size = Pt(font_size)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_body(doc: Document, text: str, *, bold_prefix: str | None = None) -> Any:
    p = doc.add_paragraph()
    p.style = doc.styles["Body Text"]
    if bold_prefix and text.startswith(bold_prefix):
        p.add_run(es(bold_prefix)).bold = True
        p.add_run(es(text[len(bold_prefix) :]))
    else:
        p.add_run(es(text))
    return p


def add_bullet(doc: Document, text: str, *, level: int = 0) -> Any:
    p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    p.add_run(es(text))
    return p


def add_figure(doc: Document, path: Path, width: float, caption: str, alt: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(3)
    set_paragraph_keep(p, keep_next=True)
    p.add_run().add_picture(str(path), width=Inches(width))
    set_picture_alt(p, es(alt))
    cap = doc.add_paragraph(es(caption), style="Caption")
    cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_paragraph_keep(cap)


def add_heading(doc: Document, text: str, level: int = 1) -> Any:
    p = doc.add_heading(es(text), level=level)
    set_paragraph_keep(p, keep_next=True)
    return p


def set_document_language(doc: Document, language: str = "es-CO") -> None:
    styles_element = doc.styles.element
    doc_defaults = styles_element.find(qn("w:docDefaults"))
    if doc_defaults is None:
        doc_defaults = OxmlElement("w:docDefaults")
        styles_element.insert(0, doc_defaults)
    rpr_default = doc_defaults.find(qn("w:rPrDefault"))
    if rpr_default is None:
        rpr_default = OxmlElement("w:rPrDefault")
        doc_defaults.append(rpr_default)
    default_rpr = rpr_default.find(qn("w:rPr"))
    if default_rpr is None:
        default_rpr = OxmlElement("w:rPr")
        rpr_default.append(default_rpr)
    default_lang = default_rpr.find(qn("w:lang"))
    if default_lang is None:
        default_lang = OxmlElement("w:lang")
        default_rpr.append(default_lang)
    default_lang.set(qn("w:val"), language)
    default_lang.set(qn("w:eastAsia"), language)
    default_lang.set(qn("w:bidi"), language)

    for style in doc.styles:
        style_rpr = style._element.get_or_add_rPr()
        style_lang = style_rpr.find(qn("w:lang"))
        if style_lang is None:
            style_lang = OxmlElement("w:lang")
            style_rpr.append(style_lang)
        style_lang.set(qn("w:val"), language)
        style_lang.set(qn("w:eastAsia"), language)
        style_lang.set(qn("w:bidi"), language)

    settings = doc.settings.element
    theme_lang = settings.find(qn("w:themeFontLang"))
    if theme_lang is None:
        theme_lang = OxmlElement("w:themeFontLang")
        settings.append(theme_lang)
    theme_lang.set(qn("w:val"), language)
    theme_lang.set(qn("w:eastAsia"), language)
    theme_lang.set(qn("w:bidi"), language)


def configure_document(doc: Document) -> None:
    set_document_language(doc)
    section = doc.sections[0]
    section.orientation = WD_ORIENT.PORTRAIT
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.68)
    section.bottom_margin = Inches(0.67)
    section.left_margin = Inches(0.72)
    section.right_margin = Inches(0.72)
    section.header_distance = Inches(0.28)
    section.footer_distance = Inches(0.3)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Liberation Sans"
    normal.font.size = Pt(10.3)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Liberation Sans")
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE

    body = styles["Body Text"]
    body.font.name = "Liberation Sans"
    body.font.size = Pt(10.3)
    body.font.color.rgb = RGBColor.from_string(INK)
    body._element.rPr.rFonts.set(qn("w:eastAsia"), "Liberation Sans")
    body.paragraph_format.space_after = Pt(5)
    body.paragraph_format.line_spacing = 1.08
    body.paragraph_format.left_indent = Inches(0)
    body.paragraph_format.right_indent = Inches(0)
    body.paragraph_format.first_line_indent = Inches(0)

    for level, size, before, after in ((1, 17, 8, 5), (2, 13, 7, 4), (3, 11, 5, 3)):
        style = styles[f"Heading {level}"]
        style.font.name = "Liberation Sans"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(INK)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Liberation Sans")
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    caption = styles["Caption"]
    caption.font.name = "Liberation Sans"
    caption.font.size = Pt(8.2)
    caption.font.italic = False
    caption.font.color.rgb = RGBColor.from_string(MUTED)
    caption._element.rPr.rFonts.set(qn("w:eastAsia"), "Liberation Sans")
    caption.paragraph_format.space_after = Pt(5)

    for style_name in ("List Bullet", "List Bullet 2"):
        style = styles[style_name]
        style.font.name = "Liberation Sans"
        style.font.size = Pt(10.1)
        style.paragraph_format.space_after = Pt(3)

    header = section.header.paragraphs[0]
    header.text = "Informe final | VAE no equivariante vs. VAE SO(2)"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in header.runs:
        run.font.name = "Liberation Sans"
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor.from_string(MUTED)
    add_page_number(section.footer.paragraphs[0])
    for run in section.footer.paragraphs[0].runs:
        run.font.name = "Liberation Sans"
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor.from_string(MUTED)

    props = doc.core_properties
    props.title = "Informe final de resultados experimentales"
    props.subject = es(
        "Comparacion entre VAE no equivariante y VAE equivariante continuo SO(2)",
    )
    props.author = "Maximiliano Garavito Chtefan"
    props.language = "es-CO"
    props.keywords = es("VAE; SO(2); histopatologia; reconstruccion; MIL; tejido")
    props.comments = "Generado desde artefactos experimentales aceptados."


def add_cover(doc: Document) -> None:
    for _ in range(3):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("INFORME FINAL DE\nRESULTADOS EXPERIMENTALES")
    run.bold = True
    run.font.name = "Liberation Sans"
    run.font.size = Pt(26)
    run.font.color.rgb = RGBColor.from_string(INK)
    p.paragraph_format.space_after = Pt(24)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(
        es(
            "Comparacion entre un VAE no equivariante y un VAE\nequivariante continuo SO(2) para histopatologia",
        ),
    )
    run.font.name = "Liberation Sans"
    run.font.size = Pt(16)
    run.font.color.rgb = RGBColor.from_string(MUTED)
    p.paragraph_format.space_after = Pt(52)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Proyecto de tesis\n").bold = True
    p.add_run("Maximiliano Garavito Chtefan\n")
    p.add_run("Universidad Industrial de Santander")
    for run in p.runs:
        run.font.name = "Liberation Sans"
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor.from_string(INK)
    p.paragraph_format.space_after = Pt(70)
    p = doc.add_paragraph("8 de septiembre de 2026")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in p.runs:
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor.from_string(MUTED)
    doc.add_page_break()


def build_report(repo: Path, output_docx: Path, work_dir: Path) -> None:
    fixed = read_json(
        repo
        / "runs/local/professor_metrics_v1/metrics/reconstruction_fixed25_summary.json",
    )
    recon = read_json(
        repo
        / "runs/local/vae_test_reconstruction_scored_v1/metrics/overall_summary.json",
    )
    mil_normal = read_json(
        repo
        / "runs/local/ubc_ocean_mil_test_scored_v1/normal_vae/scored_predictions_and_metrics.json",
    )
    mil_so2 = read_json(
        repo
        / "runs/local/ubc_ocean_mil_test_scored_v1/so2_vae/scored_predictions_and_metrics.json",
    )
    mil_pair = read_json(
        repo / "runs/local/ubc_ocean_mil_test_scored_v1/paired_bootstrap.json",
    )
    tissue = read_json(
        repo / "runs/local/tissue_test_scored_v1/label_efficiency_table.json",
    )
    mil_training_root = (
        repo / "runs/kaggle/ubc_ocean_mil_training_v7/spec0036_mil_training"
    )
    mil_train = {
        branch: read_csv(mil_training_root / branch / "metrics/train.csv")
        for branch in ("normal_vae", "so2_vae")
    }
    mil_validation = {
        branch: read_csv(mil_training_root / branch / "metrics/validation.csv")
        for branch in ("normal_vae", "so2_vae")
    }
    mil_summaries = {
        branch: read_json(mil_training_root / branch / "final_summary.json")
        for branch in ("normal_vae", "so2_vae")
    }
    tissue_training_root = (
        repo / "runs/kaggle/tissue_label_efficiency_training_v0003/"
        "tissue_label_efficiency_training/branches"
    )
    tissue_validation = []
    for budget in (250, 500, 1000, 2500, 5671):
        row: dict[str, float] = {"budget": float(budget)}
        for branch in ("normal_vae", "so2_vae"):
            result = read_json(
                tissue_training_root
                / branch
                / f"budget_{budget:04d}_per_class/result.json",
            )
            row[branch] = result["branches"][branch]["best"]["macro_f1"]
        tissue_validation.append(row)

    work_dir.mkdir(parents=True, exist_ok=True)
    output_docx.parent.mkdir(parents=True, exist_ok=True)
    mil_chart = work_dir / "mil_test_summary.png"
    mil_class_chart = work_dir / "mil_test_class_detail.png"
    tissue_chart = work_dir / "tissue_label_efficiency.png"
    tissue_class_chart = work_dir / "tissue_label_efficiency_by_class.png"
    supervised_development_chart = work_dir / "supervised_development.png"
    draw_mil_chart(mil_normal, mil_so2, mil_chart)
    draw_wsi_class_chart(mil_normal, mil_so2, mil_class_chart)
    draw_tissue_chart(tissue, tissue_chart)
    draw_tissue_class_chart(tissue, tissue_class_chart)
    draw_supervised_development_chart(
        mil_train,
        mil_validation,
        mil_summaries,
        tissue_validation,
        supervised_development_chart,
    )

    professor_figures = repo / "runs/local/professor_metrics_v1/figures"
    recon_figures = repo / "runs/local/vae_test_reconstruction_scored_v1/figures"
    orbit_figures = repo / "runs/local/frozen_vae_rotation_orbits"
    orbit_population_json = orbit_figures / "07-all25-latent-orbits.json"
    required = [
        professor_figures / "training_dashboard.png",
        professor_figures / "metrics_boxplots_fixed25.png",
        professor_figures / "reconstructions_fixed25.png",
        professor_figures / "rotated_input_vs_latent.png",
        orbit_figures / "01-latent-orbits.png",
        orbit_figures / "07-all25-latent-orbits.png",
        orbit_population_json,
        orbit_figures / "05-paper-style-latent-pca.png",
        recon_figures / "patch_metric_boxplots.png",
        recon_figures / "paired_wsi_mae.png",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Faltan figuras aceptadas:\n" + "\n".join(missing))
    orbit_population = read_json(orbit_population_json)
    normal_orbit_median, so2_orbit_median, orbit_favorable = validate_orbit_population(
        orbit_population,
    )
    orbit_population_sheet_a = work_dir / "orbit_population_patches_00_14.png"
    orbit_population_sheet_b = work_dir / "orbit_population_patches_15_24.png"
    draw_orbit_population_report_sheets(
        orbit_figures / "07-all25-latent-orbits.png",
        orbit_population_sheet_a,
        orbit_population_sheet_b,
    )

    doc = Document()
    configure_document(doc)
    add_cover(doc)

    add_heading(doc, "1. Resumen ejecutivo", 1)
    add_body(
        doc,
        "Este informe responde las solicitudes de evaluacion registradas por el profesor en los issues #2, #3, #4 y #6: curvas y graficas de resultados, MAE, MSE, PSNR, SSIM, desviacion estandar, tamano muestral, diagramas de caja, 25 reconstrucciones fijas, rotaciones de 90/180/270 grados y una visualizacion PCA del espacio latente para ambos modelos.",
    )
    add_body(
        doc,
        "Conclusion principal. La evidencia completa es mixta y no permite declarar un ganador universal.",
        bold_prefix="Conclusion principal.",
    )
    add_bullet(
        doc,
        "Reconstruccion sellada: el VAE no equivariante obtuvo un MAE promedio ligeramente menor. La diferencia primaria normal menos SO(2) fue -0,001358, pero su IC 95% por bootstrap de WSI [-0,002213; 0,000046] incluye cero; no se establecio una diferencia.",
    )
    add_bullet(
        doc,
        "Diagnostico WSI: SO(2) fue numericamente mejor en F1 macro (0,483 frente a 0,389) y exactitud (0,609 frente a 0,522), pero el IC 95% de la diferencia de F1 macro tambien incluye cero.",
    )
    add_bullet(
        doc,
        "Tejido: SO(2) tuvo un F1 macro numericamente mayor en los cuatro presupuestos bajos; solo 500 etiquetas por clase tuvo un intervalo simultaneo que excluye cero. El area global bajo la curva de aprendizaje (AULC) fue inconclusa y, en el presupuesto maximo, el modelo normal tuvo mayor F1 macro.",
    )
    add_bullet(
        doc,
        f"Rotaciones: el parche fijo de rango 12 muestra una orbita PCA de SO(2) visualmente mas regular. En el nuevo barrido a 1 grado sobre los 25 parches se observo una menor razon de linealidad local para SO(2) en {orbit_favorable['so2_lower_local_linearity_ratio']}/25 casos (medianas {normal_orbit_median['local_linearity_ratio']:.3f} normal y {so2_orbit_median['local_linearity_ratio']:.3f} SO(2)). No es una ventaja universal: la velocidad de recorrido fue menos uniforme para SO(2) en 25/25 y otros proxies no lo favorecen consistentemente.",
    )

    add_heading(doc, "Lectura para la presentacion", 2)
    add_table(
        doc,
        ["Pregunta", "Respuesta respaldada"],
        [
            [
                "¿Reconstruye mejor SO(2)?",
                "No. El modelo normal es ligeramente mejor en promedio, sin diferencia primaria establecida.",
            ],
            [
                "¿Ayuda en tareas posteriores?",
                "Hay senales favorables a SO(2) en WSI y en bajo etiquetado, pero no una superioridad general.",
            ],
            [
                "¿La equivarianza es visible?",
                "Si: la orbita densa de un ejemplo es mas regular con SO(2); los proxies comparables no confirman una ventaja global.",
            ],
            [
                "¿Falta experimentar?",
                "No para responder los issues. La fase pendiente es integrar y presentar la evidencia.",
            ],
        ],
        widths=[2.1, 4.75],
        font_size=9,
    )

    add_heading(doc, "2. Pregunta experimental y comparacion", 1)
    add_body(
        doc,
        "Se compararon dos codificadores variacionales entrenados sobre el mismo dominio histopatologico: una linea base convolucional no equivariante y una arquitectura con equivarianza continua SO(2). Ambos completaron 60.000 actualizaciones. Las particiones, los checkpoints y el cierre del entrenamiento precedieron al test; sin embargo, la agregacion exacta y el bootstrap de reconstruccion se formalizaron despues de abrir las etiquetas de los tests posteriores. Esos resultados no se usaron para cambiar modelos ni seleccionar checkpoints.",
    )
    add_table(
        doc,
        ["Propiedad", NORMAL_LABEL, SO2_LABEL],
        [
            [
                "Simetria arquitectonica",
                "Sin restriccion rotacional explicita",
                "Equivarianza continua SO(2)",
            ],
            ["Actualizaciones", "60.000", "60.000"],
            ["Parametros aprendibles", "3.958.435", "1.180.035"],
            ["Tamano relativo", "100%", "29,8% del modelo normal"],
        ],
        widths=[1.7, 2.55, 2.55],
    )
    add_body(
        doc,
        "La reduccion aproximada de 70,2% en parametros es relevante para eficiencia, pero tambien es una limitacion causal: la comparacion cambia simultaneamente la simetria y la capacidad del modelo.",
    )

    add_heading(doc, "Implementacion SO(2) verificada antes del experimento", 2)
    add_body(
        doc,
        "La implementacion continua SO(2) fue validada por componentes y como VAE completo antes del entrenamiento sobre datos reales. Esta verificacion responde la parte de implementacion del issue #6.",
    )
    add_table(
        doc,
        ["Componente", "Evidencia de verificacion"],
        [
            [
                "No linealidades y normalizacion",
                "34 compuertas radiales F0/F1 y 40 normalizaciones: aritmetica FP32, gradientes y actualizaciones finitas.",
            ],
            [
                "Remuestreo y upsampling",
                "Seis downsamplers y seis upsamplers fijos; formas 256-128-64-32 y 32-64-128-256 verificadas.",
            ],
            [
                "Muestreo VAE y estadisticas latentes",
                "mu, logvar, z con epsilon controlado, decodificador y reconstruccion comprobados en angulos cardinales y no cardinales; mu tiene forma (B,16,32,32).",
            ],
            [
                "Referencia y runtime",
                "escnn se uso solo como oraculo de pruebas focalizadas; no es dependencia del runtime. La implementacion propia compilable pasó el prelaunch batch-25 dual T4 sin skips AMP ni recompilaciones.",
            ],
        ],
        widths=[2.05, 4.65],
        font_size=8.4,
    )

    add_heading(doc, "3. Protocolo de evaluacion", 1)
    add_table(
        doc,
        ["Capa de evidencia", "Poblacion", "Unidad", "Uso"],
        [
            [
                "Validacion fija",
                "25 parches predeterminados",
                "Parche",
                "Descripcion y control visual; no es test sellado",
            ],
            [
                "Reconstruccion sellada",
                "67.138 parches de 23 WSI",
                "Parche; bootstrap por WSI",
                "Comparacion primaria de MAE y endpoints secundarios",
            ],
            [
                "Diagnostico WSI sellado",
                "23 WSI",
                "WSI",
                "F1 macro, exactitudes y entropia cruzada",
            ],
            [
                "Tejido sellado",
                "31.572 parches de 23 WSI",
                "Parche; remuestreo por WSI",
                "Eficiencia de etiquetas en cinco presupuestos",
            ],
        ],
        widths=[1.35, 1.7, 1.65, 2.1],
        font_size=8.4,
    )
    add_body(
        doc,
        "Regla de inferencia. En reconstruccion, MAE normalizado fue el endpoint confirmatorio. MSE, PSNR y SSIM fueron secundarios. Las desviaciones estandar de parches describen dispersion y no sustituyen los intervalos de incertidumbre por WSI.",
        bold_prefix="Regla de inferencia.",
    )
    add_body(
        doc,
        "Direccion de contrastes. Salvo que se indique lo contrario, las diferencias se informan como normal menos SO(2): valores negativos favorecen al modelo normal en errores (MAE/MSE) y favorecen a SO(2) en metricas donde un valor mayor es mejor (F1/exactitud).",
        bold_prefix="Direccion de contrastes.",
    )

    doc.add_page_break()
    add_heading(doc, "4. Entrenamiento y validacion fija de 25 parches", 1)
    add_body(
        doc,
        "La validacion fija permite comprobar convergencia, estabilidad y calidad visual con exactamente los mismos parches para ambos modelos. Sus resultados son descriptivos; no reemplazan la evaluacion sellada.",
    )
    add_figure(
        doc,
        professor_figures / "training_dashboard.png",
        6.75,
        "Figura 1. Panel de entrenamiento hasta 60.000 actualizaciones: objetivo, Charbonnier/L1, 1-SSIM, PSNR, diagnostico latente de equivarianza y tasa de aprendizaje. Validacion fija de 25 parches; no es test sellado.",
        "Panel de seis graficas de entrenamiento para los modelos normal y SO2 hasta 60000 actualizaciones.",
    )

    normal_fixed = fixed["models"]["normal"]
    so2_fixed = fixed["models"]["so2"]
    metric_rows = []
    for label, key in (
        ("MAE (menor es mejor)", "mae_norm"),
        ("MSE (menor es mejor)", "mse_norm"),
        ("PSNR dB (mayor es mejor)", "psnr_img_db"),
        ("SSIM (mayor es mejor)", "ssim_img"),
    ):
        nstat, sstat = normal_fixed[key], so2_fixed[key]
        decimals = 3 if key == "psnr_img_db" else 4
        metric_rows.append([
            label,
            f"{nstat['mean']:.{decimals}f} ({nstat['std_population']:.{decimals}f})",
            f"{sstat['mean']:.{decimals}f} ({sstat['std_population']:.{decimals}f})",
            "25 / 25",
        ])
    add_table(
        doc,
        ["Metrica", "Normal: media (DEp)", "SO(2): media (DEp)", "n"],
        metric_rows,
        widths=[2.2, 1.8, 1.8, 0.9],
        font_size=8.7,
    )
    add_body(
        doc,
        "En esta muestra descriptiva, el modelo normal tuvo menor MAE/MSE y mayor PSNR/SSIM. No se aplico inferencia estadistica a la validacion fija.",
    )
    add_figure(
        doc,
        professor_figures / "metrics_boxplots_fixed25.png",
        6.65,
        "Figura 2. Diagramas de caja solicitados para MAE, MSE, PSNR y SSIM en los 25 parches fijos. Cada caja resume n=25 por modelo; DEp significa desviacion estandar poblacional.",
        "Cuatro diagramas de caja comparan MAE, MSE, PSNR y SSIM en 25 parches para los dos modelos.",
    )

    doc.add_page_break()
    add_heading(doc, "5. Reconstrucciones cualitativas fijas", 1)
    add_body(
        doc,
        "Las reconstrucciones permiten verificar que ambos modelos preservan estructura y color generales. La inspeccion visual complementa, pero no reemplaza, las metricas poblacionales.",
    )
    add_figure(
        doc,
        professor_figures / "reconstructions_fixed25.png",
        6.9,
        "Figura 3. Originales y reconstrucciones de los 25 parches predeterminados para ambos modelos. Poblacion: validacion fija; n=25; seleccion fijada antes del analisis.",
        "Mosaico de 25 parches originales junto con sus reconstrucciones por el VAE normal y el VAE SO2.",
    )
    add_heading(doc, "Cómo leer esta figura", 2)
    add_bullet(
        doc,
        "Buscar perdida de bordes celulares, suavizado excesivo, cambios de color y artefactos repetitivos.",
    )
    add_bullet(
        doc,
        "No elegir un modelo por un parche aislado: las metricas selladas agregan 67.138 parches.",
    )
    add_bullet(
        doc,
        "La seleccion fija evita escoger ejemplos favorables despues de observar los resultados.",
    )

    doc.add_page_break()
    add_heading(doc, "6. Reconstruccion en el test sellado", 1)
    add_body(
        doc,
        "La evaluacion principal amplio la comparacion a 67.138 parches procedentes de 23 WSI. Los intervalos se calcularon con 10.000 remuestras por conglomerado WSI; cubren variacion entre WSI, no nuevas semillas de entrenamiento ni una nueva seleccion de parches.",
    )
    recon_rows = []
    keys = (
        ("MAE", "mae_norm"),
        ("MSE", "mse_norm"),
        ("PSNR (dB)", "psnr_img"),
        ("SSIM", "ssim_img"),
    )
    for label, key in keys:
        nstat = recon["branches"]["normal"]["patch_distribution"][key]
        sstat = recon["branches"]["so2"]["patch_distribution"][key]
        decimals = 3 if key == "psnr_img" else 4
        delta = recon["paired_normal_minus_so2"][key]["point_difference"]
        recon_rows.append([
            label,
            f"{nstat['mean']:.{decimals}f} ({nstat['population_sd']:.{decimals}f})",
            f"{sstat['mean']:.{decimals}f} ({sstat['population_sd']:.{decimals}f})",
            f"{delta:+.{decimals}f}",
        ])
    add_table(
        doc,
        ["Metrica", "Normal: media (DEp)", "SO(2): media (DEp)", "Normal - SO(2)"],
        recon_rows,
        widths=[1.55, 1.9, 1.9, 1.35],
        font_size=8.6,
    )
    add_body(
        doc,
        "Resultado primario. MAE: 0,062872 para el modelo normal y 0,064230 para SO(2). Diferencia normal menos SO(2) = -0,001358; IC 95% [-0,002213; 0,000046]. Como el intervalo incluye cero, no se establecio una diferencia de reconstruccion primaria.",
        bold_prefix="Resultado primario.",
    )
    add_body(
        doc,
        "Endpoints secundarios. PSNR y SSIM favorecen descriptivamente al modelo normal, y sus intervalos exploratorios no cruzan cero. Se mantienen como resultados secundarios: no sustituyen la conclusion del endpoint primario.",
        bold_prefix="Endpoints secundarios.",
    )
    add_figure(
        doc,
        recon_figures / "patch_metric_boxplots.png",
        6.6,
        "Figura 4. Distribuciones por parche en el test sellado: 67.138 parches por modelo, agrupados en 23 WSI. Las cajas describen dispersion de parches; no son intervalos inferenciales. La tabla superior proporciona la escala numerica.",
        "Diagramas de caja del test sellado comparan las cuatro metricas de reconstruccion entre ambos modelos.",
    )

    doc.add_page_break()
    add_heading(doc, "7. Consistencia entre WSI", 1)
    add_body(
        doc,
        "La vista pareada por WSI evita que el gran numero de parches oculte la unidad biologica. En 22 de los 23 WSI, el MAE medio del modelo normal fue menor; un WSI mostro el patron contrario. Aun asi, la incertidumbre del estimando primario ponderado por parches alcanza el cero.",
    )
    add_figure(
        doc,
        recon_figures / "paired_wsi_mae.png",
        6.8,
        "Figura 5. MAE medio pareado por WSI en el test sellado (n=23 WSI). Cada linea conecta exactamente el mismo WSI bajo ambos modelos.",
        "Grafica pareada con 23 lineas que conectan el MAE de cada WSI para el modelo normal y el modelo SO2.",
    )
    add_heading(doc, "Interpretacion", 2)
    add_body(
        doc,
        "El patron por WSI sugiere una pequena ventaja de fidelidad para el modelo normal, coherente con los endpoints secundarios. Sin embargo, el objetivo del experimento no era optimizar solo pixeles: tambien se evaluo si las representaciones resultaban utiles para diagnostico y tejido.",
    )

    doc.add_page_break()
    add_heading(doc, "8. Rotaciones y equivarianza", 1)
    add_body(
        doc,
        "El profesor solicito comparar rotaciones de 90, 180 y 270 grados por dos rutas: reconstruir la entrada ya rotada y transformar espacialmente el latente antes de decodificar. La figura siguiente usa el parche predeterminado de rango 12 y muestra ambas rutas para los dos modelos.",
    )
    add_figure(
        doc,
        professor_figures / "rotated_input_vs_latent.png",
        6.5,
        "Figura 6. Comparacion cualitativa a 0, 90, 180 y 270 grados: verdad objetivo, reconstruccion de la entrada rotada y reconstruccion despues de transformar el latente. n=1, selector rank 12 del fixed-25; no es resultado poblacional ni test sellado.",
        "Cuadricula de rotaciones que compara imagen objetivo, reconstruccion de entrada rotada y reconstruccion de latente rotado para ambos modelos.",
    )
    add_heading(doc, "Resultado cuantitativo complementario", 2)
    add_body(
        doc,
        "En el único parche mostrado, la diferencia MAE entre las dos rutas de decodificación fue 0,1002/0,1097/0,0988 para el VAE normal y 0,0118/0,0130/0,0109 para SO(2) a 90/180/270 grados. Esto respalda una mayor consistencia entre rutas en ese ejemplo, pero n=1 impide generalizarlo.",
    )
    add_body(
        doc,
        "En el proxy enmascarado de residuo latente para los 25 parches, menor es mejor. Las medianas normal fueron 1,327/1,303/1,327 a 90/180/270 grados, frente a 1,473/1,418/1,487 para SO(2). Este resultado no favorece a SO(2) en la salida posterior final. Por eso la evidencia de equivarianza se considera mixta.",
    )
    add_body(
        doc,
        "El barrido denso de 0 a 359 grados y los campos internos F1 explican el mecanismo arquitectonico, pero son diagnosticos exploratorios y no cambian la conclusion de rendimiento.",
    )
    add_figure(
        doc,
        orbit_figures / "01-latent-orbits.png",
        6.45,
        "Figura 7. Orbita latente del parche predeterminado de rango 12 durante un barrido de 0 a 359 grados. Cada modelo usa una PCA propia ajustada una sola vez sobre todos los angulos; los dos componentes explican 3,75% de la varianza normal y 4,21% de SO(2). La trayectoria de SO(2) forma un ciclo visualmente mas regular, pero bases, signos y escalas difieren entre modelos. Los paneles inferiores mantienen visible el residuo directo y el resumen exacto de los 25 parches.",
        "Comparacion de orbitas PCA durante una rotacion completa: el ejemplo SO2 forma un ciclo mas regular, acompañado por graficas de residuo latente.",
    )
    add_body(
        doc,
        "Lectura de suavidad y predictibilidad. En este ejemplo preseleccionado, el recorrido SO(2) es mas cercano a una trayectoria cerrada, continua y periodica, mientras que el recorrido normal presenta mas pliegues. Esto sugiere una respuesta rotacional de menor complejidad en las dos componentes dominantes del parche mostrado. No establece por si solo equivarianza global: las PCA son separadas y, en el residuo directo comparable de los 25 parches, el modelo normal obtuvo valores menores.",
        bold_prefix="Lectura de suavidad y predictibilidad.",
    )
    add_heading(doc, "Comprobación poblacional a resolución de 1 grado", 2)
    add_body(
        doc,
        "Para comprobar que la forma del parche 12 no fuera un caso afortunado, se repitió el ciclo completo para los 25 parches fijos: 360 inferencias por parche y modelo, de 0 a 359 grados. Cada panel ajusta una sola PCA por parche y modelo, con escala isótropa. Las métricas inferiores se calculan en el posterior medio crudo enmascarado, no sobre la apariencia de la PCA.",
    )
    add_figure(
        doc,
        orbit_population_sheet_a,
        6.75,
        "Figura 8a. Órbitas de rotación de los parches fijos 00--14 a intervalos de 1 grado. Cada par usa una PCA propia y escala isótropa; la geometría puede compararse dentro de cada panel, no como coordenadas absolutas entre modelos.",
        "Primera de dos cuadriculas legibles con orbitas PCA normal y SO2 para los parches fijos 00 a 14, cada una formada por 360 rotaciones.",
    )
    doc.add_page_break()
    add_figure(
        doc,
        orbit_population_sheet_b,
        6.75,
        f"Figura 8b. Órbitas de rotación de los parches fijos 15--24 a intervalos de 1 grado. En los 25 pares, SO(2) obtuvo menor razón de linealidad local en {orbit_favorable['so2_lower_local_linearity_ratio']}/25 (medianas {so2_orbit_median['local_linearity_ratio']:.3f} frente a {normal_orbit_median['local_linearity_ratio']:.3f} normal). Esta razón divide el RMS cíclico de la segunda diferencia de mu entre el RMS cíclico de la primera diferencia a 1 grado; menor implica mayor suavidad local. El CV del paso —desviación estándar dividida por la media de los pasos RMS cíclicos de mu a 1 grado— favoreció a SO(2) en {orbit_favorable['so2_lower_step_size_cv']}/25 (medianas {so2_orbit_median['step_size_cv']:.3f} y {normal_orbit_median['step_size_cv']:.3f}); las dos primeras PC explicaron más varianza para SO(2) en {orbit_favorable['so2_higher_pca_explained_variance']}/25. Es evidencia descriptiva de esta validación fija, no una prueba de equivarianza global ni un resultado de test sellado.",
        "Segunda cuadricula legible con orbitas PCA normal y SO2 para los parches fijos 15 a 24, seguida por la interpretacion cuantitativa de los 25 pares.",
    )
    add_body(
        doc,
        "Interpretación. La menor razón de linealidad local de SO(2) apareció en los 25 parches bajo la métrica predefinida; por tanto, el patrón no fue exclusivo del parche 12 dentro de esta muestra fija. Sin embargo, una curva puede ser localmente suave y recorrerla a velocidad poco uniforme; además, la planitud PCA y el residuo de rotación exacta responden preguntas distintas. La conclusión queda restringida a mayor suavidad local del recorrido a resolución de 1 grado.",
        bold_prefix="Interpretacion.",
    )

    doc.add_page_break()
    add_heading(doc, "9. Visualizacion PCA del espacio latente", 1)
    add_body(
        doc,
        "La visualizacion tipo EQ-VAE proyecta descriptores latentes espaciales a tres componentes principales y los presenta como falso color. Su proposito es mostrar organizacion espacial dentro del latente; PCA es un sistema de coordenadas, no una metrica de calidad ni de equivarianza.",
    )
    add_figure(
        doc,
        orbit_figures / "05-paper-style-latent-pca.png",
        6.45,
        "Figura 9. Visualizacion PCA-RGB de cuatro parches mostrados (ranks 0, 8, 12 y 24) de la fuente fixed-25. Los modelos usan ajustes PCA separados; los colores y rangos no deben compararse como magnitudes absolutas entre modelos.",
        "Visualizacion PCA en falso color compara mapas latentes espaciales del modelo normal y del modelo SO2.",
    )
    add_body(
        doc,
        "La coherencia local mediana r_edge fue 1,4567 para el modelo normal y 1,4676 para SO(2); menor es mas coherente bajo este proxy. La diferencia no respalda una ventaja de coherencia local para SO(2). Un lector RGB local exploratorio tambien obtuvo R2=0,286 para normal y R2=0,189 para SO(2). Ninguno de estos diagnosticos es un endpoint confirmatorio.",
    )

    doc.add_page_break()
    add_heading(doc, "10. Diagnostico WSI en el test sellado", 1)
    nm, sm = mil_normal["metrics"], mil_so2["metrics"]
    add_heading(doc, "Desarrollo supervisado y seleccion", 2)
    add_body(
        doc,
        "Los clasificadores se ajustaron después de congelar cada representación. En MIL se usaron 106 WSI de entrenamiento y 23 WSI de validación; ambos entrenamientos terminaron por parada temprana. La pérdida de train se registró en línea antes de cada actualización y se resume por ventanas de 53 actualizaciones, una WSI por actualización, mientras que el F1 y la entropía cruzada de validación se calcularon con el modelo fijo en cada frontera. Para tejido se entrenó un clasificador por representación y presupuesto con subconjuntos balanceados. La validación sirvió para seleccionar checkpoints y no se reutiliza como evidencia del test sellado.",
    )
    add_figure(
        doc,
        supervised_development_chart,
        6.9,
        "Figura 10. Desarrollo supervisado. Arriba izquierda: F1 macro de validación MIL; los puntos resaltan los checkpoints seleccionados (0,448 normal en la actualización 3.127 y 0,591 SO(2) en 5.671). Arriba derecha: entropía cruzada de train registrada en línea y promediada por cada ventana de 53 actualizaciones, junto con la entropía cruzada de validación en cada frontera. Como el modelo cambia dentro de cada ventana, la curva de train diagnostica la optimización y no equivale a evaluar un checkpoint fijo sobre todo train. Abajo: mejor F1 macro de validación tisular por presupuesto. Ningún panel es test sellado ni variación entre semillas.",
        "Tres paneles muestran F1 macro de validacion WSI, entropia cruzada de train y validacion WSI, y el mejor F1 macro de validacion tisular por presupuesto.",
    )
    add_heading(doc, "Resultado WSI sellado", 2)
    add_body(
        doc,
        "Se entreno una unica trayectoria de clasificador MIL por representacion y se evaluo una sola vez sobre 23 WSI sellados. La tarea contiene cinco diagnosticos; LGSC y MC tienen solo dos WSI cada uno, por lo que las conclusiones por clase son fragiles.",
    )
    add_table(
        doc,
        ["Metrica", "Normal", "SO(2)", "Normal - SO(2) [IC 95%]"],
        [
            [
                "F1 macro",
                f"{nm['macro_f1']:.3f}",
                f"{sm['macro_f1']:.3f}",
                f"{mil_pair['macro_f1']['difference']:+.3f} [{mil_pair['macro_f1']['confidence_low']:+.3f}; {mil_pair['macro_f1']['confidence_high']:+.3f}]",
            ],
            [
                "Exactitud balanceada",
                f"{nm['balanced_accuracy']:.3f}",
                f"{sm['balanced_accuracy']:.3f}",
                f"{mil_pair['balanced_accuracy']['difference']:+.3f} [{mil_pair['balanced_accuracy']['confidence_low']:+.3f}; {mil_pair['balanced_accuracy']['confidence_high']:+.3f}]",
            ],
            [
                "Exactitud",
                f"{nm['accuracy']:.3f}",
                f"{sm['accuracy']:.3f}",
                f"{mil_pair['accuracy']['difference']:+.3f} [{mil_pair['accuracy']['confidence_low']:+.3f}; {mil_pair['accuracy']['confidence_high']:+.3f}]",
            ],
            [
                "Entropia cruzada",
                f"{nm['mean_ce']:.3f}",
                f"{sm['mean_ce']:.3f}",
                f"{mil_pair['mean_ce']['difference']:+.3f} [{mil_pair['mean_ce']['confidence_low']:+.3f}; {mil_pair['mean_ce']['confidence_high']:+.3f}]",
            ],
        ],
        widths=[1.65, 1.0, 1.0, 3.05],
        font_size=8.5,
    )
    add_figure(
        doc,
        mil_chart,
        6.55,
        "Figura 11. Metricas de diagnostico WSI en el test sellado (n=23 WSI). SO(2) es numericamente mayor en F1 macro y exactitud; la diferencia de F1 macro no está establecida porque su IC 95% incluye cero.",
        "Grafica de barras de F1 macro, exactitud balanceada y exactitud para 23 WSI bajo ambos modelos.",
    )
    add_body(
        doc,
        "La entropia cruzada, endpoint secundario, fue menor para SO(2) y su intervalo pareado favorece a SO(2). Este resultado debe interpretarse junto con el unico entrenamiento del clasificador, el test pequeno y el bajo soporte de dos clases.",
    )

    doc.add_page_break()
    add_heading(doc, "Desglose por clase en diagnostico WSI", 2)
    add_body(
        doc,
        "Las matrices permiten ver que la diferencia global no se distribuye uniformemente. SO(2) obtuvo F1 mayor en CC, EC, HGSC y LGSC, mientras que el modelo normal obtuvo F1 mayor en MC. En particular, el VAE normal no recupero casos EC ni LGSC y SO(2) no recupero MC. Estas observaciones son descriptivas: los soportes por clase son CC=5, EC=6, HGSC=8, LGSC=2 y MC=2 WSI.",
    )
    add_figure(
        doc,
        mil_class_chart,
        6.45,
        "Figura 12. Matrices de confusion normalizadas por clase real y F1 por diagnostico en el test sellado. Cada celda incluye conteo y porcentaje dentro de su fila; los soportes 5/6/8/2/2 muestran por que LGSC y MC no permiten conclusiones estables por clase.",
        "Dos matrices de confusion y barras de F1 por clase comparan los clasificadores WSI normal y SO2 en 23 laminas.",
    )
    add_heading(doc, "Atribucion espacial a parches", 2)
    add_body(
        doc,
        "No se incluyo un mapa de atencion por WSI porque la corrida sellada almaceno logits, prediccion, tamano de bolsa e identidad y resumen del grafo, pero no compuertas por parche. La atencion local describe intercambio entre vecinos; las compuertas sigmoidales del resumen global no son probabilidades ni atribuciones causales o especificas de clase. Generar un mapa exige una nueva inferencia instrumentada sobre latentes y pesos congelados y, para superponerlo sobre tejido real, acceso al thumbnail o WSI. Un analisis futuro deberia preseleccionar el WSI antes de inspeccionar el mapa, usar la misma escala espacial para ambos modelos y etiquetar el resultado como intensidad de lectura interna o sensibilidad post-hoc, no como explicacion causal.",
    )

    doc.add_page_break()
    add_heading(doc, "11. Eficiencia de etiquetas en tejido", 1)
    add_body(
        doc,
        "La prueba de tejido uso 31.572 parches de 23 WSI: 21.796 tumorales, 8.500 de estroma y 1.276 de necrosis. La necrosis aparece en solo cinco WSI. Se compararon cinco presupuestos de etiquetas por clase con intervalos simultaneos para controlar la familia de contrastes. El bootstrap se estratifico por soporte tisular observado: 2 WSI con solo tumor, 16 con tumor y estroma, y 5 con los tres tejidos.",
    )
    tissue_rows = []
    for row in tissue["rows"]:
        sim = row["primary_simultaneous_95"]
        tissue_rows.append([
            f"{row['labels_per_class']:,}".replace(",", "."),
            f"{row['normal_vae']['metrics']['macro_f1']:.3f}",
            f"{row['so2_vae']['metrics']['macro_f1']:.3f}",
            f"{sim['estimate']:+.3f} [{sim['lower']:+.3f}; {sim['upper']:+.3f}]",
        ])
    add_table(
        doc,
        [
            "Etiquetas/clase",
            "F1 macro normal",
            "F1 macro SO(2)",
            "Normal - SO(2), IC simultaneo 95%",
        ],
        tissue_rows,
        widths=[1.3, 1.35, 1.35, 2.7],
        font_size=8.4,
    )
    add_figure(
        doc,
        tissue_chart,
        6.55,
        "Figura 13. Curvas de F1 macro por presupuesto en el test sellado de tejido: 31.572 parches de 23 WSI. Solo 500 etiquetas por clase presenta un intervalo simultaneo que excluye cero y favorece a SO(2).",
        "Grafica lineal de F1 macro a cinco presupuestos de etiquetas por clase para ambos modelos; 500 está marcado como diferencia establecida.",
    )
    aulc = tissue["aulc"]
    add_body(
        doc,
        f"Resumen global. La AULC trapezoidal normalizada sobre log10(etiquetas por clase), sin suavizado, fue {aulc['normal_vae']['estimate']:.3f} para normal y {aulc['so2_vae']['estimate']:.3f} para SO(2). Diferencia normal menos SO(2) = {aulc['primary_simultaneous_95']['estimate']:+.3f}; IC simultaneo 95% [{aulc['primary_simultaneous_95']['lower']:+.3f}; {aulc['primary_simultaneous_95']['upper']:+.3f}]. El intervalo incluye cero: la ventaja global es inconclusa. Esta AULC resume el F1 macro medio en el rango logaritmico preespecificado; no demuestra que un modelo necesite menos etiquetas.",
        bold_prefix="Resumen global.",
    )
    add_figure(
        doc,
        tissue_class_chart,
        6.55,
        "Figura 14. F1 por tipo de tejido a cada presupuesto de etiquetas. Son estimaciones puntuales descriptivas; la inferencia primaria simultanea se definio para F1 macro, no para seleccionar retrospectivamente una clase. Necrosis tiene 1.276 parches pero aparece en solo cinco WSI.",
        "Tres curvas comparan F1 de tumor, estroma y necrosis a cinco presupuestos para los modelos normal y SO2.",
    )
    add_body(
        doc,
        "El desglose muestra que las diferencias cambian segun tejido y presupuesto. SO(2) presenta su senal mas consistente en tumor a presupuestos bajos; estroma y, especialmente, necrosis fluctuan mas. En el presupuesto maximo, la ventaja de F1 macro del modelo normal se explica en parte por un F1 de necrosis mayor. Esta lectura no convierte las curvas por clase en contrastes confirmatorios.",
    )

    doc.add_page_break()
    add_heading(doc, "12. Sintesis integrada", 1)
    add_table(
        doc,
        ["Dimension", "Senal observada", "Conclusion permitida"],
        [
            [
                "Fidelidad de reconstruccion",
                "Normal ligeramente mejor",
                "Sin diferencia primaria establecida",
            ],
            [
                "Diagnostico WSI",
                "SO(2) mayor en F1 macro y exactitud",
                "Senal favorable, no superioridad general",
            ],
            [
                "Tejido con pocas etiquetas",
                "SO(2) mayor en 250-2.500",
                "Evidencia puntual a 500; AULC inconclusa",
            ],
            [
                "Presupuesto maximo de tejido",
                "Normal mayor en F1 macro",
                "El beneficio de SO(2) no persiste uniformemente",
            ],
            [
                "Equivarianza/PCA",
                "Orbita mas regular en el ejemplo; proxies poblacionales mixtos",
                "Suavidad descriptiva visible; ventaja global no demostrada",
            ],
            [
                "Eficiencia parametrica",
                "SO(2) usa 29,8% de parametros",
                "Ventaja de compactacion; confusor de capacidad",
            ],
        ],
        widths=[1.6, 2.15, 3.0],
        font_size=8.4,
    )
    add_heading(doc, "Mensaje central para el profesor", 2)
    add_body(
        doc,
        "La equivarianza continua SO(2) no mejoro automaticamente la reconstruccion ni todos los diagnosticos latentes. Si produjo un modelo mucho mas compacto en numero de parametros y senales de utilidad en diagnostico WSI y en ciertos regimenes de bajo etiquetado. La conclusion cientificamente responsable es una posible compensacion: el punto estimado de MAE sugiere una pequena desventaja de fidelidad para SO(2), sin diferencia primaria establecida, junto con utilidad posterior dependiente de la tarea y ausencia de una ventaja universal.",
    )

    add_heading(doc, "13. Limitaciones", 1)
    add_bullet(
        doc,
        "Una sola trayectoria de entrenamiento por VAE y una sola trayectoria de clasificador por presupuesto: los intervalos no incluyen variacion entre semillas de entrenamiento.",
    )
    add_bullet(
        doc,
        "El test WSI tiene 23 laminas; LGSC y MC solo aportan dos WSI cada una.",
    )
    add_bullet(
        doc,
        "El test de tejido contiene 31.572 parches, pero su incertidumbre efectiva depende de 23 WSI y la necrosis aparece en solo cinco.",
    )
    add_bullet(
        doc,
        "Los intervalos de tejido condicionan la inferencia a los estratos de soporte observados (2/16/5 WSI). Los cinco presupuestos comparten parches de test y datos de entrenamiento anidados, por lo que sus resultados estan correlacionados.",
    )
    add_bullet(
        doc,
        "El bootstrap cubre remuestreo de WSI. No cubre seleccion de parches, seleccion de arquitectura ni nuevas semillas.",
    )
    add_bullet(
        doc,
        "Las desviaciones estandar por parche son dispersion descriptiva, no error estandar ni intervalo de confianza.",
    )
    add_bullet(
        doc,
        "SO(2) tiene aproximadamente 70,2% menos parametros; simetria y capacidad no estan aisladas como causas separadas.",
    )
    add_bullet(
        doc,
        "Las visualizaciones de rotacion y PCA explican comportamiento, pero no constituyen endpoints de rendimiento. El barrido de 25 parches caracteriza una poblacion fija de validacion y no incorpora variacion entre semillas ni test sellado.",
    )
    add_bullet(
        doc,
        "La evaluacion WSI sellada no guardo compuertas por parche. Cualquier mapa espacial de lectura o atribucion requiere una nueva inferencia instrumentada y una interpretacion post-hoc separada.",
    )

    add_heading(doc, "14. Conclusiones para la presentacion", 1)
    add_bullet(
        doc,
        "Los requerimientos de evaluacion del profesor quedaron cubiertos para ambos modelos.",
    )
    add_bullet(
        doc,
        "La reconstruccion sellada no demuestra una diferencia primaria; el modelo normal conserva una ligera ventaja descriptiva.",
    )
    add_bullet(
        doc,
        "SO(2) ofrece compactacion parametrica fuerte y senales favorables en tareas posteriores, especialmente con 500 etiquetas por clase y en diagnostico WSI, pero con incertidumbre y soporte limitado.",
    )
    add_bullet(
        doc,
        "En los 25 parches fijos, el barrido cada 1 grado respalda una trayectoria SO(2) localmente mas suave, pero no una ventaja general de uniformidad, planitud PCA, residuo equivarante ni rendimiento.",
    )
    add_bullet(
        doc,
        "La contribucion del experimento es mostrar dónde la arquitectura SO(2) presentó senales favorables, dónde no y dónde aparece una posible compensacion descriptiva con la fidelidad de reconstruccion.",
    )
    doc.add_page_break()
    add_heading(doc, "Anexo A. Cobertura de lo solicitado por el profesor", 1)
    coverage_table = add_table(
        doc,
        ["Solicitud", "Donde se presenta", "Estado"],
        [
            ["Graficas de resultados de la linea base", "Figuras 1-5", "Completo"],
            ["MAE, MSE, PSNR, SSIM", "Secciones 4 y 6", "Completo"],
            ["Desviacion estandar, n y boxplots", "Tablas y Figuras 2 y 4", "Completo"],
            ["25 originales y reconstrucciones", "Figura 3", "Completo"],
            ["Rotaciones 90/180/270", "Figura 6", "Completo"],
            ["Orbita latente 0-359 y suavidad visual", "Figura 7", "Completo"],
            ["Órbitas de los 25 parches cada 1 grado", "Figuras 8a y 8b", "Completo"],
            ["Visualizacion PCA espacial del latente", "Figura 9", "Completo"],
            [
                "Repetir evaluacion para SO(2)",
                "Todas las tablas y figuras comparativas",
                "Completo",
            ],
            [
                "Validar componentes SO(2)",
                "Seccion 2: no linealidades, normalizacion, upsampling, mu/logvar/z y runtime",
                "Completo",
            ],
            [
                "Barrido continuo 0-359 grados",
                "Seccion 8: resumen del artefacto fijo-25",
                "Completo; resumido",
            ],
            [
                "Utilidad posterior y desglose por clase",
                "Figuras 10-14; Secciones 10 y 11",
                "Completo",
            ],
            [
                "Atribucion espacial WSI",
                "Seccion 10: frontera y protocolo futuro",
                "Pendiente; requiere nueva inferencia",
            ],
        ],
        widths=[2.55, 3.15, 1.0],
        font_size=8.3,
    )

    add_heading(doc, "Anexo B. Definicion breve de metricas", 1)
    metrics_table = add_table(
        doc,
        ["Metrica", "Lectura"],
        [
            [
                "MAE",
                "Error absoluto medio en dominio normalizado [-1, 1]; menor es mejor.",
            ],
            [
                "MSE",
                "Error cuadratico medio en dominio normalizado [-1, 1]; menor es mejor.",
            ],
            [
                "PSNR",
                "Relacion senal-ruido pico en dB, imagen recortada a [0, 1]; mayor es mejor.",
            ],
            ["SSIM", "Similitud estructural en imagen [0, 1]; mayor es mejor."],
            [
                "F1 macro",
                "Promedio no ponderado del F1 de cada clase; valora por igual clases frecuentes y raras.",
            ],
            [
                "Exactitud balanceada",
                "Promedio del recall por clase; reduce el predominio de las clases frecuentes.",
            ],
            [
                "Entropia cruzada",
                "Perdida probabilistica; menor es mejor y penaliza predicciones confiadas incorrectas.",
            ],
            [
                "AULC",
                "Area trapezoidal normalizada del F1 macro sobre log10(etiquetas por clase), sin suavizado; equivale al rendimiento medio en el rango logaritmico preespecificado y no demuestra necesitar menos etiquetas.",
            ],
            [
                "Razon de linealidad local",
                "RMS ciclico de la segunda diferencia de mu dividido por el RMS ciclico de la primera diferencia a pasos de 1 grado; menor indica un recorrido localmente mas suave.",
            ],
            [
                "CV del paso angular",
                "Desviacion estandar dividida por la media de los pasos RMS ciclicos de mu entre angulos consecutivos; menor indica velocidad de recorrido mas uniforme.",
            ],
        ],
        widths=[1.5, 5.2],
        font_size=8.3,
    )
    for table in (coverage_table, metrics_table):
        for row in table.rows:
            for cell in row.cells:
                set_cell_margins(cell, top=45, bottom=45)

    add_heading(doc, "Trazabilidad de resultados", 2)
    add_body(
        doc,
        "Fuentes aceptadas: professor_metrics_v1; vae_test_reconstruction_scored_v1; ubc_ocean_mil_test_scored_v1; tissue_test_scored_v1; frozen_vae_rotation_orbits; Kaggle maximshtefan/eqvae-fixed25-dense-rotation-population/1. El informe no recalcula predicciones, no selecciona nuevos ejemplos y no modifica los artefactos experimentales.",
    )

    doc.save(output_docx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument("--output-docx", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    output = (
        args.output_docx
        or repo / "reports/professor/informe_final_experimento_eqvae.docx"
    )
    work_dir = args.work_dir or repo / ".agent_tmp/professor_report_build"
    build_report(repo, output.resolve(), work_dir.resolve())
    print(output.resolve())


if __name__ == "__main__":
    main()
