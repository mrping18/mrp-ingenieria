from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import io
import statsmodels.api as sm

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Simulating a session/in-memory store for Vercel simplicity (Stateless is better, but for demo we hold in dict)
# In a real Vercel app with multiple users, we should use a proper database or client-side storage
app_data = {
    'df_cond': None,
    'df_def': None,
    'df_merged': None,
    'vars_polvo': [],
    'model_coefs': None,
    'bounds': None
}

@app.route('/clear', methods=['POST'])
def clear_data():
    app_data['df_cond'] = None
    app_data['df_def'] = None
    app_data['df_merged'] = None
    app_data['vars_polvo'] = []
    app_data['model_coefs'] = None
    app_data['bounds'] = None
    return jsonify({'success': 'Memoria borrada'})


@app.route('/')
def index():
    files_loaded = 1 if app_data['df_merged'] is not None else 0
    return render_template('index.html', files_loaded=files_loaded)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        try:
            # We assume a format similar to what we worked on
            xls = pd.ExcelFile(file)
            
            if 'SEGUIMIENTO' in xls.sheet_names and 'DEFECTOS' in xls.sheet_names:
                df_cond = pd.read_excel(file, sheet_name='SEGUIMIENTO')
                df_def = pd.read_excel(file, sheet_name='DEFECTOS')
                
                # --- Basic Cleaning mimicking previous script ---
                df_cond['Mes_lower'] = df_cond['Mes'].astype(str).str.lower().str.strip()
                df_def['Mes_lower'] = df_def['Mes'].astype(str).str.lower().str.strip()
                
                group_cond = df_cond.groupby(['Mes_lower', 'Día']).mean(numeric_only=True).reset_index()
                
                for c in df_def.columns:
                    if c not in ['Año', 'Proceso', 'Día', 'Mes', 'Hora', 'Mes_lower']:
                        df_def[c] = pd.to_numeric(df_def[c], errors='coerce').fillna(0)
                        
                group_def = df_def.groupby(['Mes_lower', 'Día']).sum(numeric_only=True).reset_index()
                df_merged = pd.merge(group_cond, group_def, on=['Mes_lower', 'Día'], how='inner')
                
                defects_cols = [c for c in group_def.columns if c not in ['Año', 'Proceso', 'Día', 'Mes_lower']]
                defects_cols = [c for c in defects_cols if c in df_merged.columns]
                df_merged['Total_Defectos_Polvo'] = df_merged[defects_cols].sum(axis=1)
                
                vars_polvo = ['Humedad (%)', 'Índice de Hausner', 'Fluidez', 'Tiempo de añejamiento (h)', 
                              'Malla Nº 30', 'Malla Nº 40', 'Malla Nº 50', 'Malla Nº 60', 'Malla Nº 80', 
                              'Malla Nº 120', 'Malla Nº 230', 'Fondo', 'Diferencia de humedad']
                vars_polvo = [v for v in vars_polvo if v in df_merged.columns]
                
                app_data['df_merged'] = df_merged
                app_data['vars_polvo'] = vars_polvo
                
                return jsonify({'success': 'Data loaded and merged successfully.', 'rows': len(df_merged)})
            else:
                return jsonify({'error': 'Missing SEGUIMIENTO or DEFECTOS sheet'})
        except Exception as e:
            return jsonify({'error': str(e)})

    files_loaded = 1 if app_data['df_merged'] is not None else 0
    return render_template('upload.html', files_loaded=files_loaded)

@app.route('/analysis')
def analysis():
    if app_data['df_merged'] is None:
        return render_template('error.html', message="Primero debe cargar los datos.")
    
    df = app_data['df_merged']
    vars_to_analyze = app_data['vars_polvo'] + ['Total_Defectos_Polvo']
    
    # 1. Descriptive Stats (Translated and Explanatory)
    stats_df = df[vars_to_analyze].describe().round(2)
    index_map = {
        'count': 'Cantidad de Registros',
        'mean': 'Promedio Histórico',
        'std': 'A qué tanto varían los datos (Desviación)',
        'min': 'El valor más bajo registrado',
        '25%': 'El 25% de los días fue menor a esto',
        '50%': 'El valor central (Mediana)',
        '75%': 'El 75% de los días fue menor a esto',
        'max': 'El valor más alto registrado'
    }
    stats_df = stats_df.rename(index=index_map)
    stats = stats_df.to_html(classes='table table-striped table-hover align-middle text-start', justify='left')
    
    # 2. Correlation Plot (Heatmap)
    # Fill NaN correlations with 0 just in case to avoid rendering issues
    corr = df[vars_to_analyze].corr().round(2).fillna(0)
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', 
                         title="Heatmap de Correlaciones")
    graphJSON_corr = fig_corr.to_json(engine='json')
    
    # 3. Target Correlation Bar Chart
    corr_target = corr[['Total_Defectos_Polvo']].drop('Total_Defectos_Polvo').sort_values(by='Total_Defectos_Polvo', ascending=False)
    fig_bar = px.bar(corr_target, x='Total_Defectos_Polvo', y=corr_target.index, orientation='h',
                     title="Impacto de Variables en Defectos (Correlación)",
                     color='Total_Defectos_Polvo', color_continuous_scale='RdBu_r')
    graphJSON_bar = fig_bar.to_json(engine='json')
    
    # 4. Scatter Plot for highest correlated variable
    if not corr_target.empty:
        top_var = corr_target.abs().sort_values(by='Total_Defectos_Polvo', ascending=False).index[0]
        # Dropna specifically for these to avoid ols failure plotting
        df_clean = df[[top_var, 'Total_Defectos_Polvo']].dropna()
        fig_scatter = px.scatter(df_clean, x=top_var, y='Total_Defectos_Polvo', trendline="ols",
                                 title=f"Dispersión Detallada: {top_var} vs Defectos")
        graphJSON_scatter = fig_scatter.to_json(engine='json')
    else:
        graphJSON_scatter = "{}"

    # 5. Histogram of Defects
    fig_hist = px.histogram(df.dropna(subset=['Total_Defectos_Polvo']), x='Total_Defectos_Polvo', nbins=20, 
                            title='Distribución Histórica de Defectos Totales',
                            color_discrete_sequence=['#8C68CB'])
    graphJSON_hist = fig_hist.to_json(engine='json')

    return render_template('analysis.html', stats=stats, 
                           graphJSON_corr=graphJSON_corr, graphJSON_bar=graphJSON_bar,
                           graphJSON_scatter=graphJSON_scatter, graphJSON_hist=graphJSON_hist)

@app.route('/test_graph')
def test_graph():
    if app_data['df_merged'] is None: return "Sube el excel primero."
    df = app_data['df_merged']
    fig_hist = px.histogram(df.dropna(subset=['Total_Defectos_Polvo']), x='Total_Defectos_Polvo', nbins=20)
    return fig_hist.to_html(full_html=True, include_plotlyjs=True)

@app.route('/optimization', methods=['GET', 'POST'])

def optimization():
    if app_data['df_merged'] is None:
        return render_template('error.html', message="Primero debe cargar los datos.")
    
    df = app_data['df_merged']
    vars_polvo = app_data['vars_polvo']
    
    if request.method == 'POST':
        # Fit model
        X = df[vars_polvo].fillna(df[vars_polvo].mean())
        y = df['Total_Defectos_Polvo'].fillna(0)
        
        X_std = X.std()
        cols_to_keep = X_std[X_std > 0].index
        X = X[cols_to_keep]
        vars_used = list(cols_to_keep)
        
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        app_data['model_coefs'] = model.params
        
        bounds_dict = {}
        bounds_tuple = []
        for v in vars_used:
            # Check if user overriden bounds
            min_val = float(request.form.get(f'{v}_min', df[v].min()))
            max_val = float(request.form.get(f'{v}_max', df[v].max()))
            bounds_dict[v] = {'min': min_val, 'max': max_val, 'mean': df[v].mean()}
            bounds_tuple.append((min_val, max_val))
            
        app_data['bounds'] = bounds_dict
        
        # Optimize using scipy minimize (SLSQP / L-BFGS-B equivalent)
        coefs = model.params
        def objective(x):
            return coefs['const'] + sum(c * val for c, val in zip(coefs[vars_used], x))
            
        x0 = [bounds_dict[v]['mean'] for v in vars_used]
        res = minimize(objective, x0, bounds=bounds_tuple, method='L-BFGS-B')
        
        optimal_values = {v: round(val, 3) for v, val in zip(vars_used, res.x)}
        expected_defects = round(res.fun, 3)
        current_defects = round(objective(x0), 3)
        
        # Create comparison chart
        fig_comp = go.Figure(data=[
            go.Bar(name='Media Histórica', x=vars_used, y=[bounds_dict[v]['mean'] for v in vars_used], marker_color='#D9E1F2'),
            go.Bar(name='Óptimo Sugerido', x=vars_used, y=list(optimal_values.values()), marker_color='#4472C4')
        ])
        fig_comp.update_layout(title="Comparación: Valores Actuales vs Recomendación Óptima")
        graphJSON_comp = fig_comp.to_json(engine='json')
        
        return render_template('optimization_results.html', 
                               optimal_values=optimal_values,
                               defect_reduction=defect_reduction,
                               target_defects=target_defects,
                               graphJSON_comp=graphJSON_comp,
                               params=vars_used,
                               bounds=bounds_dict)
                               
    # Request GET
    bounds = {v: {'min': df[v].min(), 'max': df[v].max()} for v in vars_polvo}
    return render_template('optimization.html', vars=vars_polvo, bounds=bounds)

@app.route('/export_solver')
def export_solver():
    if app_data['model_coefs'] is None:
        return "Debe ejecutar la optimización primero.", 400
        
    df_merged = app_data['df_merged']
    vars_polvo = app_data['vars_polvo']
    coefs = app_data['model_coefs']
    bounds = app_data['bounds']
    
    # Exclude constant
    vars_polvo_used = [v for v in bounds.keys()]
    
    output = io.BytesIO()
    import xlsxwriter
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    workbook = writer.book
    
    # --- PREMIUM FORMATS ---
    title_format = workbook.add_format({'bold': True, 'font_size': 18, 'bg_color': '#203764', 'font_color': 'white', 'valign': 'vcenter'})
    subtitle_format = workbook.add_format({'bold': True, 'font_size': 14, 'bg_color': '#4472C4', 'font_color': 'white', 'valign': 'vcenter'})
    header_format = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1, 'text_wrap': True, 'align': 'center', 'valign': 'vcenter'})
    cell_format = workbook.add_format({'border': 1, 'align': 'center', 'valign': 'vcenter'})
    text_format = workbook.add_format({'text_wrap': True, 'valign': 'vcenter'})
    instruction_format = workbook.add_format({'text_wrap': True, 'valign': 'vcenter', 'font_size': 11, 'italic': True, 'font_color': '#595959'})
    highlight_format = workbook.add_format({'bold': True, 'bg_color': '#FFD966', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
    result_format = workbook.add_format({'bold': True, 'font_size': 14, 'bg_color': '#C6EFCE', 'font_color': '#006100', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
    corr_pos_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
    corr_neg_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
    
    # --- SHEET 1: INTRODUCCIÓN Y EXPLICACIÓN ---
    worksheet0 = workbook.add_worksheet('0. INTRODUCCIÓN')
    worksheet0.set_column('A:A', 5)
    worksheet0.set_column('B:I', 15)
    
    worksheet0.merge_range('B2:H3', ' Bienvenidos al Optimizador de Polvo Atomizado', title_format)
    
    intro_text = [
        "Este archivo te ayudará a reducir los defectos de la losa vinculados al polvo atomizado mediante matemáticas.",
        "",
        "¿CÓMO FUNCIONA?",
        "1. En la pestaña '1. DATOS_CONSOLIDADOS' el archivo ya ha unido tu historial de condiciones de polvo con los defectos de la prensa por cada día.",
        "2. En la pestaña '2. CORRELACIONES' te explicamos qué variables son las más culpables de generar defectos según tu propia historia.",
        "3. En la pestaña '3. OPTIMIZACION_SOLVER' encontrarás el simulador. Allí podrás jugar con los valores y Excel calculará automáticamente los defectos esperados.",
        "",
        "¿QUÉ ES UNA CORRELACIÓN? (Explicación para la pestaña 2)",
        "Imagina que mides dos cosas todos los días (ej. la Diferencia de Humedad y la cantidad total de Defectos). La correlación mide si esas dos cosas se mueven juntas:",
        "- Correlación Positiva (+): Significan que van de la mano. Si la Diferencia de Humedad sube, los Defectos también suben. ¡Esto es malo!",
        "- Correlación Negativa (-): Significa que van al revés. Si el Tiempo de Añejamiento sube, los Defectos bajan. ¡Esto es bueno!",
        "- Correlación Cero (0): Significa que no tienen nada que ver la una con la otra.",
        "",
        "Sigue las instrucciones en la hoja 3 para usar Solver y encontrar el punto ideal donde los defectos se hacen cero."
    ]
    
    row = 5
    for line in intro_text:
        format_to_use = subtitle_format if ("¿" in line or line.isupper() and len(line)>0) else text_format
        if format_to_use == subtitle_format:
            worksheet0.merge_range(f'B{row}:H{row}', " " + line, format_to_use)
        else:
            worksheet0.merge_range(f'B{row}:H{row}', line, format_to_use)
        if line == "":
            worksheet0.set_row(row-1, 10) 
        else:
            worksheet0.set_row(row-1, 20)
        row += 1
        
    # --- SHEET 2: DATOS CONSOLIDADOS ---
    df_merged_to_export = df_merged[['Mes_lower', 'Día'] + vars_polvo + ['Total_Defectos_Polvo']].copy()
    df_merged_to_export.to_excel(writer, sheet_name='1. DATOS_CONSOLIDADOS', index=False, startrow=2)
    worksheet1 = writer.sheets['1. DATOS_CONSOLIDADOS']
    
    worksheet1.merge_range('A1:O2', ' Base de Datos Consolidada (Historia del Proceso)', title_format)
    for col_num, value in enumerate(df_merged_to_export.columns.values):
        worksheet1.write(2, col_num, value, header_format)
        worksheet1.set_column(col_num, col_num, 15)
    worksheet1.set_column('O:O', 20)
    
    # --- SHEET 3: ANALISIS DESCRIPTIVO ---
    worksheet2 = workbook.add_worksheet('2. CORRELACIONES')
    worksheet2.merge_range('B2:E3', ' Relación Variable vs Defectos (Correlaciones)', title_format)
    worksheet2.set_column('A:A', 5)
    worksheet2.set_column('B:B', 30)
    worksheet2.set_column('C:C', 20)
    worksheet2.set_column('D:D', 25)
    worksheet2.set_column('E:E', 25)
    
    corr = df_merged[vars_polvo + ['Total_Defectos_Polvo']].corr()
    corr_target = corr[['Total_Defectos_Polvo']].drop('Total_Defectos_Polvo').sort_values(by='Total_Defectos_Polvo', ascending=False)
    
    worksheet2.write('B5', 'Variable del Polvo', header_format)
    worksheet2.write('C5', 'Fuerza de Relación', header_format)
    worksheet2.write('D5', '¿Qué Significa?', header_format)
    
    row = 6
    for index, value in corr_target['Total_Defectos_Polvo'].items():
        worksheet2.write(row-1, 1, index, cell_format)
        if value > 0.3:
            worksheet2.write(row-1, 2, value, corr_pos_format)
            worksheet2.write(row-1, 3, "↗️ Crítico. Si sube, aumentan mucho los defectos.", corr_pos_format)
        elif value > 0.1:
            worksheet2.write(row-1, 2, value, cell_format)
            worksheet2.write(row-1, 3, "↗️ Si sube, aumentan un poco los defectos.", cell_format)
        elif value < -0.2:
            worksheet2.write(row-1, 2, value, corr_neg_format)
            worksheet2.write(row-1, 3, "↘️ Excelente. Si sube, BAJAN los defectos.", corr_neg_format)
        elif value < -0.1:
            worksheet2.write(row-1, 2, value, cell_format)
            worksheet2.write(row-1, 3, "↘️ Si sube, bajan un poco los defectos.", cell_format)
        else:
            worksheet2.write(row-1, 2, value, cell_format)
            worksheet2.write(row-1, 3, "--- Casi no afecta los defectos.", cell_format)
        worksheet2.set_row(row-1, 30)
        row += 1
        
    chart = workbook.add_chart({'type': 'bar'})
    chart.add_series({
        'categories': ['2. CORRELACIONES', 5, 1, row-2, 1],
        'values':     ['2. CORRELACIONES', 5, 2, row-2, 2],
        'fill':       {'color': '#4472C4'},
    })
    chart.set_title ({'name': 'Impacto de Variables en Defectos'})
    chart.set_x_axis({'name': 'Correlación (Positiva=Malo, Negativa=Bueno)'})
    chart.set_y_axis({'name': 'Variable', 'reverse': True})
    chart.set_legend({'none': True})
    worksheet2.insert_chart('F5', chart, {'x_scale': 1.5, 'y_scale': 1.5})
    
    # --- SHEET 4: MODELO Y OPTIMIZACION ---
    worksheet3 = workbook.add_worksheet('3. OPTIMIZACION_SOLVER')
    worksheet3.set_column('A:A', 5)
    worksheet3.set_column('B:B', 30)
    worksheet3.set_column('C:E', 18)
    worksheet3.set_column('G:J', 25)
    
    worksheet3.merge_range('B2:E3', ' Simulador y Optimizador', title_format)
    worksheet3.merge_range('G2:I2', ' PASOS PARA OBTENER LOS CERO DEFECTOS CON SOLVER', subtitle_format)
    worksheet3.write('G4', '1. Modifica la celda amarilla de arriba para ver como cambian los defectos.', instruction_format)
    worksheet3.write('G6', '2. O usa SOLVER (pestaña Datos -> Solver) para que Excel lo calcule perfecto:', instruction_format)
    worksheet3.write('G7', '   - Establecer objetivo: Selecciona la celda verde brillante (C20).', instruction_format)
    worksheet3.write('G8', '   - Para: Selecciona "Min" (Minimizar defectos).', instruction_format)
    worksheet3.write('G9', '   - Cambiando celdas de variables: Selecciona las celdas amarillas (C6:C18).', instruction_format)
    worksheet3.write('G10','   - Agrega las restricciones: C6:C18 >= D6:D18  y  C6:C18 <= E6:E18.', instruction_format)
    worksheet3.write('G11','   - Clic en "Resolver".', instruction_format)
    
    worksheet3.write('B5', 'Variable', header_format)
    worksheet3.write('C5', 'INGRESA VALOR AQUI', highlight_format)
    worksheet3.write('D5', 'Límite Min. Histórico', header_format)
    worksheet3.write('E5', 'Límite Max. Histórico', header_format)
    worksheet3.write('L5', 'Coeficientes Regresión', header_format) 
    
    row = 6
    for v in vars_polvo_used:
        worksheet3.write(row-1, 1, v, header_format)
        worksheet3.write(row-1, 2, bounds[v]['mean'], cell_format) 
        worksheet3.write(row-1, 3, bounds[v]['min'], cell_format)
        worksheet3.write(row-1, 4, bounds[v]['max'], cell_format)
        worksheet3.write(row-1, 11, coefs[v]) 
        worksheet3.set_row(row-1, 20)
        row += 1
        
    worksheet3.write(row, 1, 'Constante del Modelo', cell_format)
    worksheet3.write(row, 11, coefs['const'])
    worksheet3.write(row+2, 1, 'DEFECTOS PROYECTADOS:', header_format)
    formula = f"=MAX(0, SUMPRODUCT(C6:C{5+len(vars_polvo_used)}, L6:L{5+len(vars_polvo_used)}) + L{row+1})"
    worksheet3.write_formula(row+2, 2, formula, result_format)
    worksheet3.set_row(row+2, 35)
    worksheet3.set_column('L:L', None, None, {'hidden': True})
    
    writer.close()
    output.seek(0)
    
    from flask import send_file
    return send_file(output, as_attachment=True, download_name="Modelo_Optimizacion_Polvo.xlsx", 
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Ensure app object is named 'app' for Vercel Serverless


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)
