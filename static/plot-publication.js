(function (global) {
    'use strict';

    if (!global.Plotly || global.MigaPlotStyle) return;

    const Plotly = global.Plotly;
    const STORAGE_KEY = 'migaPublicationColumn';
    const CSS_DPI = 96;
    const PNG_DPI = 600;
    const MM_PER_INCH = 25.4;
    const COLUMN_WIDTH_MM = Object.freeze({ single: 89, double: 183 });
    const SCREEN = Object.freeze({ base: 14, axisTitle: 16, plotTitle: 16, annotation: 13 });
    const PRINT = Object.freeze({ basePt: 8, axisTitlePt: 9, plotTitlePt: 10, annotationPt: 8 });
    const SINGLE_PRINT = Object.freeze({ basePt: 7.5, axisTitlePt: 8.5, plotTitlePt: 9.5, annotationPt: 7.5 });
    const COLORS = Object.freeze({
        up: '#D55E00',
        down: '#0072B2',
        total: '#009E73',
        accent: '#CC79A7',
        orange: '#E69F00',
        sky: '#56B4E9',
        ink: '#2B2B2B',
        grid: '#E3E5E8',
        white: '#FFFFFF'
    });
    const COLOR_MAP = Object.freeze({
        '#dc3545': COLORS.up,
        '#c62828': COLORS.up,
        '#0d6efd': COLORS.down,
        '#1565c0': COLORS.down,
        '#198754': COLORS.total,
        '#6f42c1': COLORS.accent,
        '#e67e22': COLORS.orange,
        '#ef6c00': COLORS.orange,
        '#fd7e14': COLORS.orange,
        '#20c997': COLORS.sky,
        '#212529': COLORS.ink,
        '#343a40': COLORS.ink,
        '#111827': COLORS.ink
    });

    const originalNewPlot = Plotly.newPlot.bind(Plotly);
    const originalReact = Plotly.react.bind(Plotly);
    const exportingPlots = new WeakSet();

    function clone(value) {
        if (value == null) return value;
        if (typeof global.structuredClone === 'function') {
            try {
                return global.structuredClone(value);
            } catch (_) {
                // Plotly input normally contains plain data. Fall back for older browsers
                // and for layouts containing values that structuredClone cannot copy.
            }
        }
        return JSON.parse(JSON.stringify(value));
    }

    function normalizeColumn(value) {
        return value === 'single' ? 'single' : 'double';
    }

    function getColumn() {
        try {
            return normalizeColumn(global.localStorage.getItem(STORAGE_KEY));
        } catch (_) {
            return 'double';
        }
    }

    function setColumn(value) {
        const normalized = normalizeColumn(value);
        try {
            global.localStorage.setItem(STORAGE_KEY, normalized);
        } catch (_) {
            // Export remains usable when storage is unavailable.
        }
        global.dispatchEvent(new CustomEvent('miga-publication-column-change', {
            detail: { column: normalized }
        }));
        return normalized;
    }

    function mapColor(value) {
        if (typeof value !== 'string') return value;
        return COLOR_MAP[value.toLowerCase()] || value;
    }

    function mapColorscale(colorscale) {
        if (!Array.isArray(colorscale)) return colorscale;
        return colorscale.map(stop => (
            Array.isArray(stop) && stop.length >= 2 ? [stop[0], mapColor(stop[1])] : stop
        ));
    }

    function titleObject(title, fontSize) {
        if (typeof title === 'string') {
            return { text: title, font: { family: 'Arial, Helvetica, sans-serif', size: fontSize, color: COLORS.ink } };
        }
        if (!title || typeof title !== 'object') return title;
        return {
            ...title,
            font: {
                ...(title.font || {}),
                family: 'Arial, Helvetica, sans-serif',
                size: fontSize,
                color: mapColor(title.font?.color || COLORS.ink)
            }
        };
    }

    function normalizeInlineTitleSize(title, fontSize) {
        if (!title || typeof title !== 'object' || typeof title.text !== 'string') return title;
        return {
            ...title,
            text: title.text.replace(/font-size\s*:\s*[\d.]+px/gi, `font-size:${fontSize}px`)
        };
    }

    function screenAxis(axis, orientation) {
        if (!axis || typeof axis !== 'object') return axis;
        const explicitGrid = axis.showgrid;
        return {
            ...axis,
            title: titleObject(axis.title, SCREEN.axisTitle),
            tickfont: {
                ...(axis.tickfont || {}),
                family: 'Arial, Helvetica, sans-serif',
                size: SCREEN.base,
                color: COLORS.ink
            },
            automargin: axis.automargin !== false,
            showline: axis.showline !== false,
            mirror: false,
            ticks: axis.ticks === '' ? '' : 'outside',
            ticklen: 5,
            tickwidth: 1,
            tickcolor: COLORS.ink,
            linecolor: COLORS.ink,
            linewidth: 1,
            showgrid: explicitGrid === false ? false : orientation === 'y',
            gridcolor: COLORS.grid,
            gridwidth: 0.8,
            zerolinecolor: COLORS.ink,
            zerolinewidth: 1
        };
    }

    function screenSceneAxis(axis) {
        if (!axis || typeof axis !== 'object') return axis;
        return {
            ...axis,
            title: titleObject(axis.title, SCREEN.axisTitle),
            tickfont: {
                ...(axis.tickfont || {}),
                family: 'Arial, Helvetica, sans-serif',
                size: SCREEN.base,
                color: COLORS.ink
            },
            backgroundcolor: COLORS.white,
            gridcolor: '#D8DADD',
            zerolinecolor: '#A7ABB0',
            linecolor: COLORS.ink,
            showbackground: true
        };
    }

    function screenLayout(input) {
        const layout = { ...(input || {}) };
        layout.font = {
            ...(layout.font || {}),
            family: 'Arial, Helvetica, sans-serif',
            size: SCREEN.base,
            color: COLORS.ink
        };
        layout.title = normalizeInlineTitleSize(titleObject(layout.title, SCREEN.plotTitle), SCREEN.annotation);
        if (layout.title?.text) {
            layout.title = {
                ...layout.title,
                x: 0.5,
                xanchor: 'center',
                y: 0.98,
                yanchor: 'top'
            };
            layout.margin = {
                ...(layout.margin || {}),
                t: Math.max(Number(layout.margin?.t) || 0, 52)
            };
        }
        layout.xaxis = screenAxis(layout.xaxis, 'x');
        layout.yaxis = screenAxis(layout.yaxis, 'y');
        Object.keys(layout).forEach(key => {
            if (/^xaxis\d+$/.test(key)) layout[key] = screenAxis(layout[key], 'x');
            if (/^yaxis\d+$/.test(key)) layout[key] = screenAxis(layout[key], 'y');
        });
        if (layout.legend) {
            layout.legend = {
                ...layout.legend,
                font: {
                    ...(layout.legend.font || {}),
                    family: 'Arial, Helvetica, sans-serif',
                    size: SCREEN.base,
                    color: COLORS.ink
                },
                bgcolor: 'rgba(255,255,255,0.9)',
                bordercolor: '#C8CCD0'
            };
        }
        if (layout.scene) {
            layout.scene = {
                ...layout.scene,
                xaxis: screenSceneAxis(layout.scene.xaxis),
                yaxis: screenSceneAxis(layout.scene.yaxis),
                zaxis: screenSceneAxis(layout.scene.zaxis)
            };
        }
        if (Array.isArray(layout.annotations)) {
            layout.annotations = layout.annotations.map(annotation => ({
                ...annotation,
                font: {
                    ...(annotation.font || {}),
                    family: 'Arial, Helvetica, sans-serif',
                    size: Math.max(Number(annotation.font?.size) || 0, SCREEN.annotation),
                    color: mapColor(annotation.font?.color || COLORS.ink)
                }
            }));
        }
        return layout;
    }

    function printTypography(column) {
        return column === 'single' ? SINGLE_PRINT : PRINT;
    }

    function clampSingleColumnMarkerSize(size) {
        if (Array.isArray(size)) {
            return size.map(value => Number.isFinite(Number(value)) ? Math.min(Number(value), 6) : value);
        }
        return Number.isFinite(Number(size)) ? Math.min(Number(size), 6) : size;
    }

    function themeTrace(trace, publication, column = 'double') {
        if (!trace || typeof trace !== 'object') return trace;
        const themed = { ...trace };
        const typography = printTypography(column);
        if (trace.line) {
            themed.line = { ...trace.line, color: mapColor(trace.line.color) };
            if (publication && Number.isFinite(Number(trace.line.width))) {
                themed.line.width = Math.min(1.6, Math.max(1, Number(trace.line.width) * 0.65));
            }
        }
        if (trace.marker) {
            themed.marker = {
                ...trace.marker,
                color: Array.isArray(trace.marker.color) ? trace.marker.color : mapColor(trace.marker.color),
                colorscale: mapColorscale(trace.marker.colorscale)
            };
            if (publication && column === 'single' && trace.marker.size != null) {
                themed.marker.size = clampSingleColumnMarkerSize(trace.marker.size);
            }
            if (trace.marker.line) {
                themed.marker.line = {
                    ...trace.marker.line,
                    color: mapColor(trace.marker.line.color)
                };
            }
            if (trace.marker.colorbar) {
                themed.marker.colorbar = publicationColorbar(trace.marker.colorbar, publication, typography);
            }
        }
        if (trace.error_y) {
            themed.error_y = {
                ...trace.error_y,
                color: mapColor(trace.error_y.color),
                thickness: publication ? 1 : (trace.error_y.thickness || 1.2)
            };
        }
        if (trace.error_x) {
            themed.error_x = {
                ...trace.error_x,
                color: mapColor(trace.error_x.color),
                thickness: publication ? 1 : (trace.error_x.thickness || 1.2)
            };
        }
        if (trace.textfont) {
            themed.textfont = {
                ...trace.textfont,
                family: 'Arial, Helvetica, sans-serif',
                size: publication ? pointsToPixels(typography.annotationPt) : Math.max(Number(trace.textfont.size) || 0, SCREEN.annotation),
                color: mapColor(trace.textfont.color || COLORS.ink)
            };
        }
        if (trace.colorbar) themed.colorbar = publicationColorbar(trace.colorbar, publication, typography);
        if (trace.colorscale) themed.colorscale = mapColorscale(trace.colorscale);
        return themed;
    }

    function publicationColorbar(colorbar, publication, typography = PRINT) {
        const size = publication ? pointsToPixels(typography.basePt) : SCREEN.base;
        const titleSize = publication ? pointsToPixels(typography.axisTitlePt) : SCREEN.axisTitle;
        return {
            ...colorbar,
            title: titleObject(colorbar.title, titleSize),
            tickfont: {
                ...(colorbar.tickfont || {}),
                family: 'Arial, Helvetica, sans-serif',
                size,
                color: COLORS.ink
            },
            outlinecolor: COLORS.ink,
            outlinewidth: 0.8
        };
    }

    function pointsToPixels(points) {
        return points * CSS_DPI / 72;
    }

    function publicationAxis(axis, orientation, typography, singleColumn) {
        if (!axis || typeof axis !== 'object') return axis;
        return {
            ...axis,
            title: titleObject(axis.title, pointsToPixels(typography.axisTitlePt)),
            tickfont: {
                ...(axis.tickfont || {}),
                family: 'Arial, Helvetica, sans-serif',
                size: pointsToPixels(typography.basePt),
                color: COLORS.ink
            },
            nticks: axis.nticks || (singleColumn ? 5 : undefined),
            automargin: true,
            showline: axis.showline !== false,
            mirror: false,
            ticks: axis.ticks === '' ? '' : 'outside',
            ticklen: 4,
            tickwidth: 0.8,
            tickcolor: COLORS.ink,
            linecolor: COLORS.ink,
            linewidth: 0.8,
            showgrid: axis.showgrid === false ? false : orientation === 'y',
            gridcolor: COLORS.grid,
            gridwidth: 0.6,
            zerolinecolor: COLORS.ink,
            zerolinewidth: 0.8
        };
    }

    function publicationSceneAxis(axis, typography) {
        if (!axis || typeof axis !== 'object') return axis;
        return {
            ...axis,
            title: titleObject(axis.title, pointsToPixels(typography.axisTitlePt)),
            tickfont: {
                ...(axis.tickfont || {}),
                family: 'Arial, Helvetica, sans-serif',
                size: pointsToPixels(typography.basePt),
                color: COLORS.ink
            },
            backgroundcolor: COLORS.white,
            gridcolor: '#D8DADD',
            zerolinecolor: '#A7ABB0',
            linecolor: COLORS.ink,
            showbackground: true
        };
    }

    function publicationLayout(input, width, height, column = 'double') {
        const layout = clone(input || {});
        const singleColumn = column === 'single';
        const typography = printTypography(column);
        layout.width = width;
        layout.height = height;
        layout.autosize = false;
        layout.paper_bgcolor = COLORS.white;
        layout.plot_bgcolor = COLORS.white;
        layout.font = {
            ...(layout.font || {}),
            family: 'Arial, Helvetica, sans-serif',
            size: pointsToPixels(typography.basePt),
            color: COLORS.ink
        };
        layout.title = normalizeInlineTitleSize(
            titleObject(layout.title, pointsToPixels(typography.plotTitlePt)),
            pointsToPixels(typography.annotationPt)
        );
        if (layout.title?.text) {
            layout.title = {
                ...layout.title,
                x: 0.5,
                xanchor: 'center',
                y: 0.98,
                yanchor: 'top'
            };
            layout.margin = {
                ...(layout.margin || {}),
                t: Math.max(Number(layout.margin?.t) || 0, singleColumn ? 82 : 46)
            };
        }
        if (singleColumn) {
            layout.margin = {
                ...(layout.margin || {}),
                l: 64,
                r: 18,
                b: 48,
                t: Math.max(Number(layout.margin?.t) || 0, layout.title?.text ? 82 : 56),
                pad: 2
            };
        }
        layout.xaxis = publicationAxis(layout.xaxis, 'x', typography, singleColumn);
        layout.yaxis = publicationAxis(layout.yaxis, 'y', typography, singleColumn);
        Object.keys(layout).forEach(key => {
            if (/^xaxis\d+$/.test(key)) layout[key] = publicationAxis(layout[key], 'x', typography, singleColumn);
            if (/^yaxis\d+$/.test(key)) layout[key] = publicationAxis(layout[key], 'y', typography, singleColumn);
        });
        if ((singleColumn && layout.showlegend !== false) || layout.legend) {
            layout.legend = {
                ...(layout.legend || {}),
                font: {
                    ...(layout.legend?.font || {}),
                    family: 'Arial, Helvetica, sans-serif',
                    size: pointsToPixels(typography.basePt),
                    color: COLORS.ink
                },
                bgcolor: singleColumn ? 'rgba(255,255,255,0)' : 'rgba(255,255,255,0.92)',
                bordercolor: singleColumn ? 'rgba(255,255,255,0)' : '#B8BCC1',
                borderwidth: singleColumn ? 0 : 0.7,
                ...(singleColumn ? {
                    orientation: 'h',
                    x: 0,
                    xanchor: 'left',
                    y: 1.02,
                    yanchor: 'bottom',
                    traceorder: 'normal',
                    tracegroupgap: 4
                } : {})
            };
        }
        if (layout.scene) {
            layout.scene = {
                ...layout.scene,
                xaxis: publicationSceneAxis(layout.scene.xaxis, typography),
                yaxis: publicationSceneAxis(layout.scene.yaxis, typography),
                zaxis: publicationSceneAxis(layout.scene.zaxis, typography)
            };
        }
        if (Array.isArray(layout.annotations)) {
            layout.annotations = layout.annotations.map(annotation => ({
                ...annotation,
                font: {
                    ...(annotation.font || {}),
                    family: 'Arial, Helvetica, sans-serif',
                    size: pointsToPixels(typography.annotationPt),
                    color: mapColor(annotation.font?.color || COLORS.ink)
                }
            }));
        }
        delete layout.datarevision;
        delete layout.uirevision;
        return layout;
    }

    function exportDimensions(graphDiv, column) {
        const width = Math.round(COLUMN_WIDTH_MM[column] * CSS_DPI / MM_PER_INCH);
        const fullWidth = Number(graphDiv?._fullLayout?.width) || graphDiv?.clientWidth || 700;
        const fullHeight = Number(graphDiv?._fullLayout?.height) || graphDiv?.clientHeight || 450;
        const sourceRatio = fullHeight / Math.max(fullWidth, 1);
        const ratio = column === 'single'
            ? Math.min(1.15, Math.max(0.9, sourceRatio))
            : Math.min(1.35, Math.max(0.55, sourceRatio));
        return { width, height: Math.round(width * ratio) };
    }

    function safeFilename(graphDiv, column) {
        const title = typeof graphDiv?.layout?.title === 'string'
            ? graphDiv.layout.title
            : graphDiv?.layout?.title?.text;
        const raw = String(title || graphDiv?.id || 'miga_plot')
            .replace(/<[^>]*>/g, ' ')
            .replace(/[^a-zA-Z0-9._-]+/g, '_')
            .replace(/^_+|_+$/g, '')
            .slice(0, 80) || 'miga_plot';
        return `${raw}_${column === 'single' ? '89mm' : '183mm'}`;
    }

    function downloadDataUrl(url, filename) {
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.style.display = 'none';
        document.body.appendChild(link);
        link.click();
        link.remove();
    }

    async function exportPublicationPlot(graphDiv, options = {}) {
        if (!graphDiv || exportingPlots.has(graphDiv)) return;
        exportingPlots.add(graphDiv);
        graphDiv.dataset.publicationExport = 'exporting';
        graphDiv.setAttribute('aria-busy', 'true');
        const column = normalizeColumn(options.column || getColumn());
        const dimensions = exportDimensions(graphDiv, column);
        const filename = options.filename || safeFilename(graphDiv, column);
        const holder = document.createElement('div');
        holder.setAttribute('aria-hidden', 'true');
        holder.style.cssText = `position:fixed;left:-10000px;top:0;width:${dimensions.width}px;height:${dimensions.height}px;background:#fff;`;
        document.body.appendChild(holder);
        try {
            const data = clone(Array.from(graphDiv.data || [])).map(trace => themeTrace(trace, true, column));
            const layout = publicationLayout(graphDiv.layout || {}, dimensions.width, dimensions.height, column);
            await originalNewPlot(holder, data, layout, {
                staticPlot: true,
                displayModeBar: false,
                displaylogo: false,
                responsive: false
            });
            const [svgUrl, pngUrl] = await Promise.all([
                Plotly.toImage(holder, {
                    format: 'svg', width: dimensions.width, height: dimensions.height, scale: 1
                }),
                Plotly.toImage(holder, {
                    format: 'png', width: dimensions.width, height: dimensions.height, scale: PNG_DPI / CSS_DPI
                })
            ]);
            downloadDataUrl(svgUrl, `${filename}.svg`);
            downloadDataUrl(pngUrl, `${filename}_600ppi.png`);
            graphDiv.dataset.publicationExport = 'complete';
        } catch (error) {
            graphDiv.dataset.publicationExport = 'error';
            global.console.error('Publication plot export failed', error);
            global.alert(`Publication plot export failed: ${error?.message || error}`);
        } finally {
            try {
                Plotly.purge(holder);
            } catch (_) {
                // The temporary element may already have been detached after a failed render.
            }
            holder.remove();
            exportingPlots.delete(graphDiv);
            graphDiv.removeAttribute('aria-busy');
        }
    }

    const publicationCameraButton = {
        name: 'Download publication SVG + PNG (600 ppi)',
        icon: Plotly.Icons.camera,
        click: graphDiv => exportPublicationPlot(graphDiv)
    };

    function plotConfig(input) {
        const config = { ...(input || {}) };
        const removed = new Set([...(config.modeBarButtonsToRemove || []), 'toImage']);
        const additions = (config.modeBarButtonsToAdd || []).filter(button => button?.name !== publicationCameraButton.name);
        additions.push(publicationCameraButton);
        return {
            ...config,
            responsive: config.responsive !== false,
            displaylogo: false,
            modeBarButtonsToRemove: [...removed],
            modeBarButtonsToAdd: additions
        };
    }

    function themedData(data) {
        return Array.isArray(data) ? data.map(trace => themeTrace(trace, false)) : data;
    }

    Plotly.newPlot = function (graphDiv, data, layout, config) {
        return originalNewPlot(graphDiv, themedData(data), screenLayout(layout), plotConfig(config));
    };

    Plotly.react = function (graphDiv, data, layout, config) {
        return originalReact(graphDiv, themedData(data), screenLayout(layout), plotConfig(config));
    };

    global.MigaPlotStyle = Object.freeze({
        colors: COLORS,
        getColumn,
        setColumn,
        screenLayout,
        plotConfig,
        exportPublicationPlot,
        exportDimensions,
        publicationLayout
    });
})(window);
