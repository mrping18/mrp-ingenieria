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
from statsmodels.stats.outliers_influence import variance_inflation_factor


def remove_high_vif(X, threshold=10.0):
    """Iteratively remove variables with VIF > threshold to fix multicollinearity.
    Returns cleaned X and list of dropped columns."""
    dropped = []
    while True:
        X_with_const = sm.add_constant(X)
        vifs = []
        for i in range(1, X_with_const.shape[1]):  # skip const
            try:
                vif_val = variance_inflation_factor(X_with_const.values, i)
            except:
                vif_val = 0
            vifs.append(vif_val)
        
        max_vif = max(vifs) if vifs else 0
        if max_vif <= threshold or len(X.columns) <= 2:
            break
        
        worst_idx = vifs.index(max_vif)
        worst_col = X.columns[worst_idx]
        dropped.append((worst_col, round(max_vif, 1)))
        X = X.drop(columns=[worst_col])
    
    return X, dropped


app = Flask(__name__, template_folder='../templates', static_folder='../static')

# In-memory store (for single-user / demo use)
app_data = {
    'df_cond': None,
    'df_def': None,
    'df_merged': None,
    'vars_polvo': [],
    'defectos_cols': [],
    'model_coefs': None,
    'model_rsquared': None,
    'model_rsquared_adj': None,
    'model_pvalues': None,
    'model_nobs': None,
    'model_vif': None,
    'model_dropped_vars': None,
    'bounds': None,
    'standards': None,
    'target_defecto': 'Total_Defectos_Polvo'
}

@app.route('/clear', methods=['POST'])
def clear_data():
    for key in app_data:
        app_data[key] = None if key not in ['vars_polvo', 'defectos_cols'] else []
    app_data['target_defecto'] = 'Total_Defectos_Polvo'
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
            xls = pd.ExcelFile(file)
            
            # --- Validation ---
            missing_sheets = []
            if 'SEGUIMIENTO' not in xls.sheet_names:
                missing_sheets.append('SEGUIMIENTO')
            if 'DEFECTOS' not in xls.sheet_names:
                missing_sheets.append('DEFECTOS')
            if missing_sheets:
                return jsonify({'error': f'Faltan las hojas requeridas: {", ".join(missing_sheets)}. Tu archivo tiene: {", ".join(xls.sheet_names)}'})
            
            df_cond = pd.read_excel(file, sheet_name='SEGUIMIENTO')
            df_def = pd.read_excel(file, sheet_name='DEFECTOS')
            
            # Validate key columns exist
            required_cond_cols = ['Mes', 'Día']
            missing_cond = [c for c in required_cond_cols if c not in df_cond.columns]
            if missing_cond:
                return jsonify({'error': f'La hoja SEGUIMIENTO no tiene las columnas: {", ".join(missing_cond)}'})
            
            required_def_cols = ['Mes', 'Día']
            missing_def = [c for c in required_def_cols if c not in df_def.columns]
            if missing_def:
                return jsonify({'error': f'La hoja DEFECTOS no tiene las columnas: {", ".join(missing_def)}'})
            
            # --- Read Standards if available ---
            standards = None
            for sname in ['ESTÁNDARES ACTUALES', 'ESTANDARES ACTUALES', 'ESTÁNDARES']:
                if sname in xls.sheet_names:
                    try:
                        df_std = pd.read_excel(file, sheet_name=sname)
                        if 'Variable' in df_std.columns and 'Rango' in df_std.columns:
                            standards = {}
                            for _, row_s in df_std.iterrows():
                                var_name = str(row_s['Variable']).strip()
                                rango = str(row_s['Rango']).strip()
                                standards[var_name] = rango
                    except:
                        pass
                    break
            
            # --- Cleaning ---
            df_cond['Mes_lower'] = df_cond['Mes'].astype(str).str.lower().str.strip()
            df_def['Mes_lower'] = df_def['Mes'].astype(str).str.lower().str.strip()
            
            group_cond = df_cond.groupby(['Mes_lower', 'Día']).mean(numeric_only=True).reset_index()
            
            for c in df_def.columns:
                if c not in ['Año', 'Proceso', 'Día', 'Mes', 'Hora', 'Mes_lower']:
                    df_def[c] = pd.to_numeric(df_def[c], errors='coerce').fillna(0)
                    
            group_def = df_def.groupby(['Mes_lower', 'Día']).sum(numeric_only=True).reset_index()
            df_merged = pd.merge(group_cond, group_def, on=['Mes_lower', 'Día'], how='inner')
            
            if len(df_merged) == 0:
                return jsonify({'error': 'No se encontraron coincidencias entre SEGUIMIENTO y DEFECTOS. Verifica que las columnas Mes y Día coincidan.'})
            
            defects_cols = [c for c in group_def.columns if c not in ['Año', 'Proceso', 'Día', 'Mes_lower']]
            defects_cols = [c for c in defects_cols if c in df_merged.columns]
            df_merged['Total_Defectos_Polvo'] = df_merged[defects_cols].sum(axis=1)
            
            vars_polvo = ['Humedad (%)', 'Índice de Hausner', 'Fluidez', 'Tiempo de añejamiento (h)', 
                          'Malla Nº 30', 'Malla Nº 40', 'Malla Nº 50', 'Malla Nº 60', 'Malla Nº 80', 
                          'Malla Nº 120', 'Malla Nº 230', 'Fondo', 'Diferencia de humedad']
            vars_polvo = [v for v in vars_polvo if v in df_merged.columns]
            
            # --- Pre-fit model with VIF check for multicollinearity ---
            X_pre = df_merged[vars_polvo].fillna(df_merged[vars_polvo].mean())
            y_pre = df_merged['Total_Defectos_Polvo'].fillna(0)
            X_std = X_pre.std()
            cols_keep = X_std[X_std > 0].index
            X_pre = X_pre[cols_keep]
            
            # Remove multicollinear variables (VIF > 10)
            X_clean, dropped_vars = remove_high_vif(X_pre, threshold=10.0)
            
            X_pre_c = sm.add_constant(X_clean)
            model_pre = sm.OLS(y_pre, X_pre_c).fit()
            
            # Calculate final VIFs for display
            final_vifs = {}
            for i, col in enumerate(X_clean.columns):
                try:
                    final_vifs[col] = round(variance_inflation_factor(X_pre_c.values, i+1), 1)
                except:
                    final_vifs[col] = 0
            
            app_data['df_merged'] = df_merged
            app_data['vars_polvo'] = vars_polvo
            app_data['defectos_cols'] = defects_cols
            app_data['standards'] = standards
            app_data['model_rsquared'] = round(model_pre.rsquared, 4)
            app_data['model_rsquared_adj'] = round(model_pre.rsquared_adj, 4)
            app_data['model_pvalues'] = model_pre.pvalues.round(4).to_dict()
            app_data['model_nobs'] = int(model_pre.nobs)
            app_data['model_vif'] = final_vifs
            app_data['model_dropped_vars'] = dropped_vars
            
            std_msg = ' Se detectaron estándares de la empresa.' if standards else ''
            return jsonify({'success': f'Datos cargados correctamente. {len(df_merged)} registros consolidados.{std_msg}', 'rows': len(df_merged)})
        except Exception as e:
            return jsonify({'error': f'Error procesando archivo: {str(e)}'})

    files_loaded = 1 if app_data['df_merged'] is not None else 0
    return render_template('upload.html', files_loaded=files_loaded)

@app.route('/analysis')
def analysis():
    if app_data['df_merged'] is None:
        return render_template('error.html', message="Primero debe cargar los datos.")
    
    df = app_data['df_merged']
    vars_polvo = app_data['vars_polvo']
    defectos_cols = app_data.get('defectos_cols', [])
    vars_to_analyze = vars_polvo + ['Total_Defectos_Polvo']
    
    # --- Data Summary ---
    meses_unicos = df['Mes_lower'].nunique()
    meses_lista = ', '.join(df['Mes_lower'].unique().tolist()[:6])
    n_registros = len(df)
    n_variables = len(vars_polvo)
    defectos_promedio = round(df['Total_Defectos_Polvo'].mean(), 2)
    
    data_summary = {
        'n_registros': n_registros,
        'n_meses': meses_unicos,
        'meses_lista': meses_lista,
        'n_variables': n_variables,
        'defectos_promedio': defectos_promedio
    }
    
    # --- Model Info ---
    model_info = {
        'rsquared': app_data.get('model_rsquared', 'N/A'),
        'rsquared_pct': round(app_data.get('model_rsquared', 0) * 100, 1),
        'rsquared_adj': app_data.get('model_rsquared_adj', 'N/A'),
        'rsquared_adj_pct': round(app_data.get('model_rsquared_adj', 0) * 100, 1),
        'nobs': app_data.get('model_nobs', 'N/A'),
        'pvalues': app_data.get('model_pvalues', {}),
        'vif': app_data.get('model_vif', {}),
        'dropped_vars': app_data.get('model_dropped_vars', [])
    }
    # Determine significant variables (p < 0.05)
    sig_vars = [v for v, p in model_info['pvalues'].items() if p < 0.05 and v != 'const']
    model_info['significant_vars'] = sig_vars
    
    # --- Standards Comparison ---
    standards = app_data.get('standards', None)
    standards_comparison = None
    if standards:
        standards_comparison = []
        for v in vars_polvo:
            actual_mean = round(df[v].mean(), 2)
            actual_min = round(df[v].min(), 2)
            actual_max = round(df[v].max(), 2)
            # Try to find matching standard
            std_range = None
            for std_name, std_val in standards.items():
                if std_name.lower().replace('(g/s)', '').strip() in v.lower().replace('(%)', '').strip() or v.lower().replace('(%)', '').strip() in std_name.lower().replace('(g/s)', '').strip():
                    std_range = std_val
                    break
            standards_comparison.append({
                'variable': v,
                'mean': actual_mean,
                'min': actual_min,
                'max': actual_max,
                'standard': std_range or '—'
            })
    
    # 1. Descriptive Stats
    stats_df = df[vars_to_analyze].describe().round(2)
    index_map = {
        'count': 'Cantidad de Registros',
        'mean': 'Promedio Histórico',
        'std': 'Desviación Estándar',
        'min': 'Mínimo Registrado',
        '25%': 'Percentil 25%',
        '50%': 'Mediana',
        '75%': 'Percentil 75%',
        'max': 'Máximo Registrado'
    }
    stats_df = stats_df.rename(index=index_map)
    stats = stats_df.to_html(classes='table table-striped table-hover align-middle text-start', justify='left')
    
    # 2. Correlation Plot (Heatmap)
    if defectos_cols:
        corr_matrix = df[vars_polvo + defectos_cols].corr().loc[vars_polvo, defectos_cols].round(2).fillna(0)
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', 
                             title="Mapa de Calor: Qué variable impacta qué defecto individual",
                             labels=dict(x="Tipo de Defecto", y="Variable del Polvo", color="Correlación"))
    else:
        corr = df[vars_to_analyze].corr().round(2).fillna(0)
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', 
                             title="Heatmap de Correlaciones")
    graphJSON_corr = fig_corr.to_json(engine='json')
    
    # 3. Target Correlation Bar Chart (Total Defects)
    corr_target = df[vars_polvo + ['Total_Defectos_Polvo']].corr()[['Total_Defectos_Polvo']].drop('Total_Defectos_Polvo').sort_values(by='Total_Defectos_Polvo', ascending=False)
    fig_bar = px.bar(corr_target, x='Total_Defectos_Polvo', y=corr_target.index, orientation='h',
                     title="Impacto de Variables en el Total de Defectos",
                     color='Total_Defectos_Polvo', color_continuous_scale='RdBu_r')
    graphJSON_bar = fig_bar.to_json(engine='json')
    
    # 4. Scatter Plot for highest correlated variable
    if not corr_target.empty:
        top_var = corr_target.abs().sort_values(by='Total_Defectos_Polvo', ascending=False).index[0]
        df_clean = df[[top_var, 'Total_Defectos_Polvo']].dropna()
        fig_scatter = px.scatter(df_clean, x=top_var, y='Total_Defectos_Polvo', trendline="ols",
                                 title=f"Dispersión: {top_var} vs Total Defectos")
        graphJSON_scatter = fig_scatter.to_json(engine='json')
    else:
        graphJSON_scatter = "{}"

    # 5. Histogram of Defects
    fig_hist = px.histogram(df.dropna(subset=['Total_Defectos_Polvo']), x='Total_Defectos_Polvo', nbins=20, 
                            title='Distribución Histórica de Defectos Totales',
                            color_discrete_sequence=['#8C68CB'])
    graphJSON_hist = fig_hist.to_json(engine='json')
    
    # 6. Temporal Trend Chart
    mes_order = {'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
                 'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12}
    df_trend = df[['Mes_lower', 'Día', 'Total_Defectos_Polvo']].copy()
    df_trend['mes_num'] = df_trend['Mes_lower'].map(mes_order).fillna(0)
    df_trend = df_trend.sort_values(['mes_num', 'Día'])
    df_trend['Fecha_Aprox'] = df_trend['Día'].astype(str) + '-' + df_trend['Mes_lower'].str.capitalize()
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=df_trend['Fecha_Aprox'], y=df_trend['Total_Defectos_Polvo'],
                                    mode='lines+markers', name='Defectos Totales',
                                    line=dict(color='#8C68CB', width=2),
                                    marker=dict(size=6)))
    # Add average line
    avg_def = df_trend['Total_Defectos_Polvo'].mean()
    fig_trend.add_hline(y=avg_def, line_dash="dash", line_color="#FF6B6B",
                        annotation_text=f"Promedio: {avg_def:.1f}", annotation_position="top right")
    fig_trend.update_layout(title="Tendencia Temporal de Defectos", 
                            xaxis_title="Día-Mes", yaxis_title="Total Defectos",
                            xaxis=dict(tickangle=-45))
    graphJSON_trend = fig_trend.to_json(engine='json')
    
    return render_template('analysis.html', stats=stats, 
                           graphJSON_corr=graphJSON_corr, graphJSON_bar=graphJSON_bar,
                           graphJSON_scatter=graphJSON_scatter, graphJSON_hist=graphJSON_hist,
                           graphJSON_trend=graphJSON_trend,
                           data_summary=data_summary, model_info=model_info,
                           standards_comparison=standards_comparison)

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
    defectos_cols = app_data.get('defectos_cols', [])
    
    if request.method == 'POST':
        target_defecto = request.form.get('target_defecto', 'Total_Defectos_Polvo')
        app_data['target_defecto'] = target_defecto
        
        # Fit model with VIF check
        X = df[vars_polvo].fillna(df[vars_polvo].mean())
        y = df[target_defecto].fillna(0)
        
        X_std = X.std()
        cols_to_keep = X_std[X_std > 0].index
        X = X[cols_to_keep]
        
        # Remove multicollinear variables
        X, dropped_vars = remove_high_vif(X, threshold=10.0)
        vars_used = list(X.columns)
        
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        app_data['model_coefs'] = model.params
        app_data['model_rsquared'] = round(model.rsquared, 4)
        app_data['model_rsquared_adj'] = round(model.rsquared_adj, 4)
        app_data['model_pvalues'] = model.pvalues.round(4).to_dict()
        app_data['model_nobs'] = int(model.nobs)
        app_data['model_dropped_vars'] = dropped_vars
        
        bounds_dict = {}
        bounds_tuple = []
        for v in vars_used:
            min_val = float(request.form.get(f'{v}_min', df[v].min()))
            max_val = float(request.form.get(f'{v}_max', df[v].max()))
            bounds_dict[v] = {'min': min_val, 'max': max_val, 'mean': df[v].mean()}
            bounds_tuple.append((min_val, max_val))
            
        app_data['bounds'] = bounds_dict
        
        coefs = model.params
        def objective(x):
            return coefs['const'] + sum(c * val for c, val in zip(coefs[vars_used], x))
            
        x0 = [bounds_dict[v]['mean'] for v in vars_used]
        res = minimize(objective, x0, bounds=bounds_tuple, method='L-BFGS-B')
        
        optimal_values = {v: round(val, 3) for v, val in zip(vars_used, res.x)}
        expected_defects = max(0, round(res.fun, 3))
        current_defects = max(0, round(objective(x0), 3))
        
        target_defects = expected_defects
        defect_reduction = max(0, round(current_defects - target_defects, 2))
        
        # Create comparison chart
        fig_comp = go.Figure(data=[
            go.Bar(name='Media Histórica', x=vars_used, y=[bounds_dict[v]['mean'] for v in vars_used], marker_color='#D9E1F2'),
            go.Bar(name='Óptimo Sugerido', x=vars_used, y=list(optimal_values.values()), marker_color='#4472C4')
        ])
        fig_comp.update_layout(title="Comparación: Valores Actuales vs Recomendación Óptima")
        graphJSON_comp = fig_comp.to_json(engine='json')
        
        # Significant variables for display
        sig_vars = {v: round(model.pvalues[v], 4) for v in vars_used if model.pvalues[v] < 0.05}
        
        return render_template('optimization_results.html', 
                               optimal_values=optimal_values,
                               current_defects=current_defects,
                               expected_defects=expected_defects,
                               defect_reduction=defect_reduction,
                               target_defecto=target_defecto,
                               graphJSON_comp=graphJSON_comp,
                               params=vars_used,
                               bounds=bounds_dict,
                               rsquared=round(model.rsquared * 100, 1),
                               nobs=int(model.nobs),
                               sig_vars=sig_vars)
                               
    # Request GET
    bounds = {v: {'min': df[v].min(), 'max': df[v].max()} for v in vars_polvo}
    defectos_cols = app_data.get('defectos_cols', [])
    return render_template('optimization.html', vars=vars_polvo, bounds=bounds, defectos_cols=defectos_cols)

@app.route('/export_solver')
def export_solver():
    if app_data['model_coefs'] is None:
        return "Debe ejecutar la optimización primero.", 400
        
    df_merged = app_data['df_merged']
    vars_polvo = app_data['vars_polvo']
    coefs = app_data['model_coefs']
    bounds = app_data['bounds']
    target_defecto = app_data.get('target_defecto', 'Total_Defectos_Polvo')
    
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
        "3. En la pestaña '3. DEFECTOS_Y_CAUSAS' encontrarás una guía técnica de qué condiciones del polvo disparan qué defectos.",
        "4. En la pestaña '4. RESUMEN_EJECUTIVO' encontrarás los hallazgos clave y las recomendaciones automáticas.",
        "5. En la pestaña '5. OPTIMIZACION_SOLVER' encontrarás el simulador. Allí podrás jugar con los valores y Excel calculará automáticamente los defectos esperados.",
        "",
        "¿QUÉ ES UNA CORRELACIÓN? (Explicación para la pestaña 2)",
        "Imagina que mides dos cosas todos los días (ej. la Diferencia de Humedad y la cantidad total de Defectos). La correlación mide si esas dos cosas se mueven juntas:",
        "- Correlación Positiva (+): Significan que van de la mano. Si la Diferencia de Humedad sube, los Defectos también suben. ¡Esto es malo!",
        "- Correlación Negativa (-): Significa que van al revés. Si el Tiempo de Añejamiento sube, los Defectos bajan. ¡Esto es bueno!",
        "- Correlación Cero (0): Significa que no tienen nada que ver la una con la otra.",
        "",
        "Sigue las instrucciones en la hoja 5 para usar Solver y encontrar el punto ideal donde los defectos se hacen cero."
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
        
    # --- SHEET 1: DATOS CONSOLIDADOS ---
    df_merged_to_export = df_merged[['Mes_lower', 'Día'] + vars_polvo + ['Total_Defectos_Polvo']].copy()
    df_merged_to_export.to_excel(writer, sheet_name='1. DATOS_CONSOLIDADOS', index=False, startrow=2)
    worksheet1 = writer.sheets['1. DATOS_CONSOLIDADOS']
    
    n_cols = len(df_merged_to_export.columns)
    last_col_letter = chr(ord('A') + min(n_cols - 1, 25))
    worksheet1.merge_range(f'A1:{last_col_letter}2', ' Base de Datos Consolidada (Historia del Proceso)', title_format)
    for col_num, value in enumerate(df_merged_to_export.columns.values):
        worksheet1.write(2, col_num, value, header_format)
        worksheet1.set_column(col_num, col_num, 15)
    
    # Add summary formulas at bottom
    data_last_row = 2 + len(df_merged_to_export)  # 0-indexed last data row
    summary_start = data_last_row + 2  # skip a row
    formula_format = workbook.add_format({'bold': True, 'bg_color': '#E2EFDA', 'border': 1, 'align': 'center', 'valign': 'vcenter', 'num_format': '0.00'})
    label_format = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1, 'valign': 'vcenter'})
    
    summary_labels = ['PROMEDIO', 'DESV. ESTÁNDAR', 'MÍNIMO', 'MÁXIMO']
    summary_funcs = ['AVERAGE', 'STDEV', 'MIN', 'MAX']
    
    for i, (label, func) in enumerate(zip(summary_labels, summary_funcs)):
        r = summary_start + i
        worksheet1.write(r, 0, label, label_format)
        worksheet1.write(r, 1, '', label_format)
        for col_num in range(2, n_cols):  # skip Mes_lower and Día
            col_letter = chr(ord('A') + col_num)
            formula = f'={func}({col_letter}4:{col_letter}{data_last_row + 1})'
            worksheet1.write_formula(r, col_num, formula, formula_format)
    
    worksheet1.set_column(n_cols - 1, n_cols - 1, 20)  # wider for Total_Defectos
    
    # --- SHEET 2: CORRELACIONES ---
    worksheet2 = workbook.add_worksheet('2. CORRELACIONES')
    n_def_cols = len(defectos_cols) if defectos_cols else 0
    last_merge_col = max(4, n_def_cols + 3)
    last_merge_letter = chr(ord('A') + min(last_merge_col, 25))
    worksheet2.merge_range(f'B2:{last_merge_letter}3', ' Relación Variable vs Tipos de Defectos', title_format)
    worksheet2.set_column('A:A', 5)
    worksheet2.set_column('B:B', 30)
    
    defectos_cols = app_data.get('defectos_cols', [])
    if defectos_cols:
        worksheet2.write('B5', 'Variable del Polvo', header_format)
        for col_idx, defecto_name in enumerate(defectos_cols):
            worksheet2.write(4, 2 + col_idx, defecto_name, header_format)
            worksheet2.set_column(2 + col_idx, 2 + col_idx, 15)
            
        corr_matrix = df_merged[vars_polvo + defectos_cols].corr().loc[vars_polvo, defectos_cols].round(2).fillna(0)
        
        row = 6
        for var_name in vars_polvo:
            worksheet2.write(row-1, 1, var_name, cell_format)
            for col_idx, defecto_name in enumerate(defectos_cols):
                val = corr_matrix.loc[var_name, defecto_name]
                if val > 0.3:
                    fmt = corr_pos_format
                elif val < -0.2:
                    fmt = corr_neg_format
                else:
                    fmt = cell_format
                worksheet2.write(row-1, 2 + col_idx, val, fmt)
            worksheet2.set_row(row-1, 25)
            row += 1
            
        worksheet2.merge_range(f'B{row+2}:E{row+2}', ' Guía de Colores:', subtitle_format)
        worksheet2.write(row+3, 1, 'Rojo', corr_pos_format)
        worksheet2.write(row+3, 2, 'Peligro. Si la variable sube, el defecto aumenta fuertemente.')
        worksheet2.write(row+4, 1, 'Verde', corr_neg_format)
        worksheet2.write(row+4, 2, 'Excelente. Si la variable sube, el defecto disminuye.')
        
        # --- Interpretation column ---
        interp_col = 2 + n_def_cols + 1
        worksheet2.set_column(interp_col, interp_col, 40)
        worksheet2.write(4, interp_col, 'Interpretación Automática', header_format)
        
        corr_vs_total = df_merged[vars_polvo + ['Total_Defectos_Polvo']].corr()['Total_Defectos_Polvo'].drop('Total_Defectos_Polvo')
        row_interp = 6
        for var_name in vars_polvo:
            val = corr_vs_total.get(var_name, 0)
            if val > 0.3:
                interp = f'⚠️ Si {var_name} sube, los defectos AUMENTAN (+{val:.2f})'
            elif val < -0.2:
                interp = f'✅ Si {var_name} sube, los defectos BAJAN ({val:.2f})'
            else:
                interp = f'➖ {var_name} tiene poco efecto directo ({val:.2f})'
            worksheet2.write(row_interp-1, interp_col, interp, text_format)
            row_interp += 1

        # Add Bar Chart for Total Defects Correlation if available
        corr_total = df_merged[vars_polvo + ['Total_Defectos_Polvo']].corr()[['Total_Defectos_Polvo']].drop('Total_Defectos_Polvo').sort_values(by='Total_Defectos_Polvo', ascending=False)
        chart_row = row + 7
        worksheet2.write(chart_row, 1, 'Impacto en DEFECTOS TOTALES (Resumen)', subtitle_format)
        worksheet2.write(chart_row+1, 1, 'Variable', header_format)
        worksheet2.write(chart_row+1, 2, 'Correlación', header_format)
        
        r = chart_row + 2
        for var_name, val in corr_total['Total_Defectos_Polvo'].items():
            worksheet2.write(r, 1, var_name, cell_format)
            worksheet2.write(r, 2, val, cell_format)
            r += 1
            
        chart = workbook.add_chart({'type': 'bar'})
        chart.add_series({
            'name':       'Impacto (Correlación)',
            'categories': ['2. CORRELACIONES', chart_row+2, 1, r-1, 1],
            'values':     ['2. CORRELACIONES', chart_row+2, 2, r-1, 2],
            'fill':       {'color': '#4472C4'},
        })
        chart.set_title({'name': '¿Qué tanto afecta cada variable al TOTAL de defectos?'})
        chart.set_legend({'none': True})
        worksheet2.insert_chart(chart_row, 4, chart, {'x_scale': 1.2, 'y_scale': 1.2})

    else:
        worksheet2.write('B5', 'No hay datos de defectos individuales disponibles.', text_format)
    
    # --- SHEET 3: GUÍA DE DEFECTOS Y CAUSAS ---
    worksheet_def = workbook.add_worksheet('3. DEFECTOS_Y_CAUSAS')
    worksheet_def.set_column('A:A', 5)
    worksheet_def.set_column('B:B', 25)
    worksheet_def.set_column('C:C', 35)
    worksheet_def.set_column('D:D', 50)
    
    worksheet_def.merge_range('B2:D3', ' Guía Técnica: Defectos vs Condiciones de Polvo', title_format)
    worksheet_def.write('B5', 'Defecto Común', header_format)
    worksheet_def.write('C5', 'Apariencia Física', header_format)
    worksheet_def.write('D5', 'Causas Posibles en el Polvo Atomizado', header_format)
    
    defects_data = [
        ["Desgarre", "Grietas finas en la superficie o bordes al salir de prensa.", "Humedad muy baja (<7.0%) o excesiva diferencia de humedad en el lecho del polvo."],
        ["Pegado", "Material adherido a los moldes (punzones).", "Humedad alta (>7.8%) o temperatura del polvo muy alta (falta enfriamiento)."],
        ["Laminado / Aire", "Separación en capas horizontales (grieta de corazón).", "Exceso de finos (Fondo >15%), polvo muy compresible (Hausner >1.18) o falta de fluidez."],
        ["Falta de Llenado", "Bordes incompletos o esquinas débiles.", "Baja fluidez del polvo (polvo 'pegajoso'), mallas gruesas obstruyendo rejillas o exceso de humedad."],
        ["Chiteado", "Pequeños desprendimientos superficiales.", "Inconsistencia en la granulometría o presencia de aglomerados duros o húmedos."],
        ["Inestabilidad Dimensional", "Variación de tamaño final tras cocción.", "Mucha variación en la humedad día a día o granulometría inconsistente."]
    ]
    
    row_d = 6
    for item in defects_data:
        worksheet_def.write(row_d-1, 1, item[0], cell_format)
        worksheet_def.write(row_d-1, 2, item[1], text_format)
        worksheet_def.write(row_d-1, 3, item[2], text_format)
        worksheet_def.set_row(row_d-1, 40)
        row_d += 1
        
    worksheet_def.merge_range(f'B{row_d+1}:D{row_d+1}', ' Nota: Esta guía es referencial basada en estándares de la industria cerámica.', instruction_format)

    # --- SHEET 4: RESUMEN EJECUTIVO ---
    ws_exec = workbook.add_worksheet('4. RESUMEN_EJECUTIVO')
    ws_exec.set_column('A:A', 5)
    ws_exec.set_column('B:B', 35)
    ws_exec.set_column('C:C', 25)
    ws_exec.set_column('D:D', 40)
    
    ws_exec.merge_range('B2:D3', ' Resumen Ejecutivo del Análisis', title_format)
    
    # General stats
    ws_exec.write('B5', 'Métrica', header_format)
    ws_exec.write('C5', 'Valor', header_format)
    ws_exec.write('D5', 'Interpretación', header_format)
    
    exec_data = [
        ['Registros analizados', str(len(df_merged)), f'Datos de {df_merged["Mes_lower"].nunique()} meses'],
        ['Defectos promedio/día', f'{df_merged["Total_Defectos_Polvo"].mean():.2f}', 'Sumatoria de todos los defectos por día'],
        ['Defectos máximo', f'{df_merged["Total_Defectos_Polvo"].max():.2f}', 'El peor día registrado'],
        ['R² del Modelo', f'{app_data.get("model_rsquared", 0)*100:.1f}%', 'Qué % de defectos explica el polvo'],
    ]
    
    r_exec = 6
    for item in exec_data:
        ws_exec.write(r_exec-1, 1, item[0], cell_format)
        ws_exec.write(r_exec-1, 2, item[1], highlight_format)
        ws_exec.write(r_exec-1, 3, item[2], text_format)
        ws_exec.set_row(r_exec-1, 25)
        r_exec += 1
    
    # Top 3 most dangerous variables
    r_exec += 1
    ws_exec.merge_range(f'B{r_exec}:D{r_exec}', ' Top Variables Más Peligrosas (mayor correlación positiva)', subtitle_format)
    r_exec += 1
    ws_exec.write(r_exec-1, 1, 'Variable', header_format)
    ws_exec.write(r_exec-1, 2, 'Correlación', header_format)
    ws_exec.write(r_exec-1, 3, 'Acción Recomendada', header_format)
    r_exec += 1
    
    corr_exec = df_merged[vars_polvo + ['Total_Defectos_Polvo']].corr()['Total_Defectos_Polvo'].drop('Total_Defectos_Polvo').sort_values(ascending=False)
    for var_name, val in corr_exec.head(3).items():
        action = 'REDUCIR esta variable' if val > 0 else 'AUMENTAR esta variable'
        ws_exec.write(r_exec-1, 1, var_name, corr_pos_format if val > 0.2 else cell_format)
        ws_exec.write(r_exec-1, 2, round(val, 3), cell_format)
        ws_exec.write(r_exec-1, 3, action, text_format)
        ws_exec.set_row(r_exec-1, 25)
        r_exec += 1
    
    # Top 3 most beneficial variables
    r_exec += 1
    ws_exec.merge_range(f'B{r_exec}:D{r_exec}', ' Top Variables Más Benéficas (mayor correlación negativa)', subtitle_format)
    r_exec += 1
    ws_exec.write(r_exec-1, 1, 'Variable', header_format)
    ws_exec.write(r_exec-1, 2, 'Correlación', header_format)
    ws_exec.write(r_exec-1, 3, 'Acción Recomendada', header_format)
    r_exec += 1
    
    for var_name, val in corr_exec.tail(3).items():
        action = 'AUMENTAR esta variable' if val < 0 else 'Monitorear'
        ws_exec.write(r_exec-1, 1, var_name, corr_neg_format if val < -0.2 else cell_format)
        ws_exec.write(r_exec-1, 2, round(val, 3), cell_format)
        ws_exec.write(r_exec-1, 3, action, text_format)
        ws_exec.set_row(r_exec-1, 25)
        r_exec += 1
    
    # Optimal conditions summary
    r_exec += 1
    ws_exec.merge_range(f'B{r_exec}:D{r_exec}', ' Condiciones Óptimas Sugeridas', subtitle_format)
    r_exec += 1
    ws_exec.write(r_exec-1, 1, 'Variable', header_format)
    ws_exec.write(r_exec-1, 2, 'Media Histórica', header_format)
    ws_exec.write(r_exec-1, 3, 'Óptimo Sugerido por el Modelo', header_format)
    r_exec += 1
    
    for v in vars_polvo_used:
        ws_exec.write(r_exec-1, 1, v, cell_format)
        ws_exec.write(r_exec-1, 2, round(bounds[v]['mean'], 3), cell_format)
        # Calculate where optimizer would push this
        if coefs.get(v, 0) > 0:
            ws_exec.write(r_exec-1, 3, round(bounds[v]['min'], 3), corr_neg_format)
        else:
            ws_exec.write(r_exec-1, 3, round(bounds[v]['max'], 3), corr_neg_format)
        ws_exec.set_row(r_exec-1, 22)
        r_exec += 1

    # --- SHEET 5: OPTIMIZACION_SOLVER ---
    worksheet3 = workbook.add_worksheet('5. OPTIMIZACION_SOLVER')
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
    worksheet3.write(row+2, 1, f'PROYECTADO ({target_defecto.upper()}):', header_format)
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
