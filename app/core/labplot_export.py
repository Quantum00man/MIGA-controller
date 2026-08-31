"""Generate self-contained LabPlot 2.12.1 projects for Archive data."""

from __future__ import annotations

import base64
import lzma
import math
import struct
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4
from xml.etree import ElementTree as ET


LABPLOT_VERSION = "2.12.1"
LABPLOT_XML_VERSION = "16"


@dataclass
class Curve:
    name: str
    x: Sequence[float]
    y: Sequence[float]
    color: Tuple[int, int, int] = (13, 110, 253)
    line: bool = True
    symbols: bool = True


@dataclass
class Plot:
    title: str
    x_label: str
    y_label: str
    curves: List[Curve] = field(default_factory=list)


@dataclass
class Worksheet:
    name: str
    plots: List[Plot] = field(default_factory=list)


def _uuid() -> str:
    return "{" + str(uuid4()) + "}"


def _stamp() -> str:
    return datetime.now().strftime("%Y-%d-%m %H:%M:%S:%f")[:-3]


def _aspect(tag: str, name: str) -> ET.Element:
    return ET.Element(tag, {"creation_time": _stamp(), "name": name, "uuid": _uuid()})


def _comment(parent: ET.Element, text: str = "") -> None:
    ET.SubElement(parent, "comment").text = text


def _finite_pairs(x_values: Sequence[Any], y_values: Sequence[Any]) -> Tuple[List[float], List[float]]:
    x_out: List[float] = []
    y_out: List[float] = []
    for raw_x, raw_y in zip(x_values, y_values):
        try:
            x = float(raw_x)
            y = float(raw_y)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            x_out.append(x)
            y_out.append(y)
    return x_out, y_out


def _column(parent: ET.Element, name: str, values: Sequence[float], designation: int) -> None:
    column = _aspect("column", name)
    column.attrib.update({
        "rows": str(len(values)), "designation": str(designation), "mode": "0", "width": "120"
    })
    _comment(column)
    input_filter = ET.SubElement(column, "input_filter")
    simple_in = _aspect("simple_filter", "SimpleFilter")
    simple_in.attrib["filter_name"] = "String2DoubleFilter"
    _comment(simple_in)
    input_filter.append(simple_in)
    output_filter = ET.SubElement(column, "output_filter")
    simple_out = _aspect("simple_filter", "SimpleFilter")
    simple_out.attrib.update({"format": "g", "digits": "10", "filter_name": "Double2StringFilter"})
    _comment(simple_out)
    output_filter.append(simple_out)
    packed = b"".join(struct.pack("=d", float(value)) for value in values)
    output_filter.tail = base64.b64encode(packed).decode("ascii")
    parent.append(column)


def _background(parent: ET.Element, tag: str = "background", border: bool = False) -> None:
    attrs = {
        "type": "0", "colorStyle": "0", "imageStyle": "1", "brushStyle": "1",
        "firstColor_r": "255", "firstColor_g": "255", "firstColor_b": "255",
        "secondColor_r": "255", "secondColor_g": "255", "secondColor_b": "255",
        "fileName": "", "opacity": "1",
    }
    element = ET.SubElement(parent, tag, attrs)
    if border:
        ET.SubElement(element, "border", {
            "borderType": "15", "style": "1", "color_r": "40", "color_g": "40",
            "color_b": "40", "width": "1.5", "opacity": "1", "borderCornerRadius": "0",
        })


def _label(parent: ET.Element, name: str, text: str, point_size: str = "14") -> None:
    label = _aspect("textLabel", name)
    _comment(label)
    ET.SubElement(label, "geometry", {
        "x": "0", "y": "0", "horizontalPosition": "1", "verticalPosition": "0",
        "horizontalAlignment": "1", "verticalAlignment": "0", "rotationAngle": "0",
        "plotRangeIndex": "0", "visible": "1", "coordinateBinding": "0",
        "logicalPosX": "0", "logicalPosY": "0", "locked": "0",
    })
    ET.SubElement(label, "text").text = text
    ET.SubElement(label, "format", {
        "placeholder": "0", "mode": "0", "fontFamily": "Noto Sans", "fontSize": "-1",
        "fontPointSize": point_size, "fontWeight": "50", "fontItalic": "0",
        "fontColor_r": "30", "fontColor_g": "30", "fontColor_b": "30",
        "backgroundColor_r": "255", "backgroundColor_g": "255", "backgroundColor_b": "255",
    })
    ET.SubElement(label, "border", {
        "borderShape": "0", "style": "0", "color_r": "0", "color_g": "0", "color_b": "0",
        "width": "1", "opacity": "1",
    })
    parent.append(label)


def _axis(parent: ET.Element, name: str, orientation: int, position: int, title: str) -> None:
    axis = _aspect("axis", name)
    _comment(axis)
    ET.SubElement(axis, "general", {
        "rangeType": "0", "orientation": str(orientation), "position": str(position), "scale": "0",
        "offset": "0", "logicalPosition": "0", "start": "0", "end": "1",
        "majorTicksStartType": "1", "majorTickStartOffset": "0", "majorTickStartValue": "0",
        "scalingFactor": "1", "zeroOffset": "0", "showScaleOffset": "1",
        "titleOffsetX": "7", "titleOffsetY": "7", "plotRangeIndex": "0", "visible": "1",
    })
    _label(axis, f"{name} title", title, "11")
    ET.SubElement(axis, "line", {
        "style": "1", "color_r": "35", "color_g": "35", "color_b": "35", "width": "1.5",
        "opacity": "1", "arrowType": "0", "arrowPosition": "1", "arrowSize": "12",
    })
    ET.SubElement(axis, "majorTicks", {
        "direction": "1", "type": "0", "numberAuto": "1", "number": "6", "increment": "0",
        "majorTicksColumn": "", "length": "8", "style": "1", "color_r": "35", "color_g": "35",
        "color_b": "35", "width": "1.2", "opacity": "1",
    })
    ET.SubElement(axis, "minorTicks", {
        "direction": "1", "type": "0", "numberAuto": "1", "number": "1", "increment": "0",
        "minorTicksColumn": "", "length": "4", "style": "1", "color_r": "35", "color_g": "35",
        "color_b": "35", "width": "1", "opacity": "1",
    })
    ET.SubElement(axis, "labels", {
        "position": "2", "offset": "8", "rotation": "0", "textType": "0", "labelsTextColumn": "",
        "format": "0", "formatAuto": "1", "precision": "6", "autoPrecision": "1",
        "dateTimeFormat": "yyyy-MM-dd hh:mm:ss", "color_r": "35", "color_g": "35", "color_b": "35",
        "fontFamily": "Noto Sans", "fontSize": "-1", "fontPointSize": "10", "fontWeight": "50",
        "fontItalic": "0", "prefix": "", "suffix": "", "opacity": "1", "backgroundType": "0",
        "backgroundColor_r": "255", "backgroundColor_g": "255", "backgroundColor_b": "255",
    })
    ET.SubElement(axis, "majorGrid", {
        "style": "1", "color_r": "220", "color_g": "225", "color_b": "232", "width": "1",
        "opacity": "1",
    })
    ET.SubElement(axis, "minorGrid", {
        "style": "0", "color_r": "230", "color_g": "230", "color_b": "230", "width": "1",
        "opacity": "1",
    })
    parent.append(axis)


def _xy_curve(parent: ET.Element, curve: Curve, x_path: str, y_path: str) -> None:
    element = _aspect("xyCurve", curve.name)
    _comment(element)
    ET.SubElement(element, "general", {
        "xColumn": x_path, "yColumn": y_path, "plotRangeIndex": "0", "legendVisible": "1", "visible": "1",
    })
    r, g, b = curve.color
    ET.SubElement(element, "lines", {
        "type": "1" if curve.line else "0", "skipGaps": "0", "increasingXOnly": "0",
        "interpolationPointsCount": "1", "style": "1", "color_r": str(r), "color_g": str(g),
        "color_b": str(b), "width": "2.2", "opacity": "1",
    })
    ET.SubElement(element, "dropLines", {
        "type": "0", "style": "1", "color_r": str(r), "color_g": str(g), "color_b": str(b),
        "width": "1", "opacity": "1",
    })
    ET.SubElement(element, "symbols", {
        "symbolsStyle": "1" if curve.symbols else "0", "opacity": "1", "rotation": "0", "size": "8",
        "brush_style": "1", "brush_color_r": str(r), "brush_color_g": str(g), "brush_color_b": str(b),
        "style": "1", "color_r": str(r), "color_g": str(g), "color_b": str(b), "width": "1",
    })
    ET.SubElement(element, "values", {
        "type": "0", "valuesColumn": "", "position": "0", "distance": "8", "rotation": "0",
        "opacity": "1", "numericFormat": "g", "dateTimeFormat": "yyyy-MM-dd", "precision": "6",
        "prefix": "", "suffix": "", "color_r": "0", "color_g": "0", "color_b": "0",
        "fontFamily": "Noto Sans", "fontSize": "-1", "fontPointSize": "9", "fontWeight": "50",
        "fontItalic": "0",
    })
    ET.SubElement(element, "filling", {
        "position": "2", "type": "0", "colorStyle": "0", "imageStyle": "1", "brushStyle": "1",
        "firstColor_r": str(r), "firstColor_g": str(g), "firstColor_b": str(b),
        "secondColor_r": str(r), "secondColor_g": str(g), "secondColor_b": str(b),
        "fileName": "", "opacity": "0",
    })
    ET.SubElement(element, "errorBars", {
        "xErrorType": "0", "xErrorPlusColumn": "", "xErrorMinusColumn": "", "yErrorType": "0",
        "yErrorPlusColumn": "", "yErrorMinusColumn": "", "type": "0", "capSize": "8", "style": "1",
        "color_r": str(r), "color_g": str(g), "color_b": str(b), "width": "1", "opacity": "1",
    })
    ET.SubElement(element, "margins", {
        "rugEnabled": "0", "rugOrientation": "2", "rugLength": "8", "rugWidth": "0", "rugOffset": "0",
    })
    parent.append(element)


def _legend(parent: ET.Element) -> None:
    legend = _aspect("cartesianPlotLegend", "Legend")
    _comment(legend)
    ET.SubElement(legend, "general", {
        "usePlotColor": "1", "color_r": "35", "color_g": "35", "color_b": "35",
        "fontFamily": "Noto Sans", "fontSize": "-1", "fontPointSize": "9", "fontWeight": "50",
        "fontItalic": "0", "columnMajor": "1", "lineSymbolWidth": "36", "visible": "1",
    })
    ET.SubElement(legend, "geometry", {
        "x": "0", "y": "0", "horizontalPosition": "1", "verticalPosition": "1",
        "horizontalAlignment": "2", "verticalAlignment": "0", "rotationAngle": "0", "plotRangeIndex": "0",
        "visible": "1", "coordinateBinding": "0", "logicalPosX": "0", "logicalPosY": "0", "locked": "0",
    })
    _label(legend, "Legend title", "", "9")
    _background(legend)
    ET.SubElement(legend, "border", {
        "style": "1", "color_r": "180", "color_g": "185", "color_b": "190", "width": "1",
        "opacity": "1", "borderCornerRadius": "0",
    })
    ET.SubElement(legend, "layout", {
        "topMargin": "6", "bottomMargin": "6", "leftMargin": "6", "rightMargin": "6",
        "verticalSpacing": "4", "horizontalSpacing": "8", "columnCount": "1",
    })
    parent.append(legend)


def build_project(project_name: str, worksheets: Iterable[Worksheet], comment: str = "") -> bytes:
    prepared: List[Tuple[Worksheet, str, List[List[Tuple[Curve, str, str]]]]] = []
    for worksheet in worksheets:
        plots: List[List[Tuple[Curve, str, str]]] = []
        for plot in worksheet.plots:
            curves = []
            for curve in plot.curves:
                x, y = _finite_pairs(curve.x, curve.y)
                if x:
                    curves.append((Curve(curve.name, x, y, curve.color, curve.line, curve.symbols), "", ""))
            plots.append(curves)
        if any(plots):
            prepared.append((worksheet, f"Data - {worksheet.name}", plots))
    if not prepared:
        raise ValueError("No finite LabPlot data was selected")

    root = ET.Element("project", {
        "version": LABPLOT_VERSION, "xmlVersion": LABPLOT_XML_VERSION, "modificationTime": _stamp(),
        "author": "MIGA Controller", "saveCalculations": "1", "dockWidgetState": "",
        "saveDefaultDockWidgetState": "0", "thumbnail": "", "creation_time": _stamp(),
        "name": project_name, "uuid": _uuid(),
    })
    _comment(root, comment)

    for worksheet, sheet_name, plot_curves in prepared:
        wrapper = ET.SubElement(root, "child_aspect")
        spreadsheet = _aspect("spreadsheet", sheet_name)
        _comment(spreadsheet)
        ET.SubElement(spreadsheet, "general", {"showComments": "0", "showSparklines": "0"})
        ET.SubElement(spreadsheet, "linking", {"enabled": "0", "spreadsheet": ""})
        counter = 0
        for plot_index, curves in enumerate(plot_curves):
            for curve_index, item in enumerate(curves):
                curve, _, _ = item
                counter += 1
                x_name = f"P{plot_index + 1}_X{curve_index + 1}"
                y_name = f"P{plot_index + 1}_{curve.name}"
                _column(spreadsheet, x_name, curve.x, 1)
                _column(spreadsheet, y_name, curve.y, 2)
                path_root = f"{project_name}/{sheet_name}"
                curves[curve_index] = (curve, f"{path_root}/{x_name}", f"{path_root}/{y_name}")
        wrapper.append(spreadsheet)

        wrapper = ET.SubElement(root, "child_aspect")
        sheet = _aspect("worksheet", worksheet.name)
        _comment(sheet)
        plot_count = max(1, len(worksheet.plots))
        height = 1200 if plot_count == 1 else 1800
        ET.SubElement(sheet, "geometry", {
            "x": "0", "y": "0", "width": "1600", "height": str(height), "useViewSize": "0", "zoomFit": "0",
        })
        ET.SubElement(sheet, "layout", {
            "layout": "3", "topMargin": "25", "bottomMargin": "25", "leftMargin": "25", "rightMargin": "25",
            "verticalSpacing": "24", "horizontalSpacing": "12", "columnCount": "1", "rowCount": str(plot_count),
        })
        _background(sheet)
        ET.SubElement(sheet, "plotProperties", {
            "plotInteractive": "0", "cartesianPlotActionMode": "0", "cartesianPlotCursorMode": "1",
        })
        plot_height = (height - 50 - (plot_count - 1) * 24) / plot_count
        for index, plot in enumerate(worksheet.plots):
            curves = plot_curves[index]
            if not curves:
                continue
            element = _aspect("cartesianPlot", plot.title)
            _comment(element)
            ET.SubElement(element, "cursor", {
                "style": "1", "color_r": "220", "color_g": "53", "color_b": "69", "width": "1", "opacity": "1",
            })
            ET.SubElement(element, "geometry", {
                "x": "25", "y": str(25 + index * (plot_height + 24)), "width": "1550",
                "height": str(plot_height), "visible": "1",
            })
            x_values = [value for curve, _, _ in curves for value in curve.x]
            y_values = [value for curve, _, _ in curves for value in curve.y]
            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)
            if x_min == x_max:
                x_min -= 0.5
                x_max += 0.5
            if y_min == y_max:
                y_min -= 0.5
                y_max += 0.5
            x_ranges = ET.SubElement(element, "xRanges")
            ET.SubElement(x_ranges, "xRange", {
                "autoScale": "1", "start": f"{x_min:.16g}", "end": f"{x_max:.16g}", "scale": "0",
                "format": "0", "dateTimeFormat": "yyyy-MM-dd hh:mm:ss",
            })
            y_ranges = ET.SubElement(element, "yRanges")
            ET.SubElement(y_ranges, "yRange", {
                "autoScale": "1", "start": f"{y_min:.16g}", "end": f"{y_max:.16g}", "scale": "0",
                "format": "0", "dateTimeFormat": "yyyy-MM-dd hh:mm:ss",
            })
            systems = ET.SubElement(element, "coordinateSystems", {
                "defaultCoordinateSystem": "0", "horizontalPadding": "90", "verticalPadding": "35",
                "rightPadding": "30", "bottomPadding": "60", "symmetricPadding": "0", "rangeType": "0",
                "rangeFirstValues": "1000", "rangeLastValues": "1000", "niceExtend": "1",
            })
            ET.SubElement(systems, "coordinateSystem", {"name": "Default", "xIndex": "0", "yIndex": "0"})
            _background(element, "plotArea", border=True)
            _label(element, f"{plot.title} - Title", plot.title, "14")
            _axis(element, "x", 0, 1, plot.x_label)
            _axis(element, "y", 1, 2, plot.y_label)
            for curve, x_path, y_path in curves:
                _xy_curve(element, curve, x_path, y_path)
            if len(curves) > 1:
                _legend(element)
            sheet.append(element)
        wrapper.append(sheet)

    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    xml = xml.replace(b"?>", b"?>\n<!DOCTYPE LabPlotXML>", 1)
    return lzma.compress(xml, format=lzma.FORMAT_XZ, preset=6)
