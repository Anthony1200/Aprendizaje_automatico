import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf

# ---------------------------------------------------------
# 1. Configuración de la Página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predicción Temporal LSTM - Deserción Escolar",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------------
# 2. Cargar Red Neuronal LSTM y Artefactos
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('modelo_lstm_estudiantes.keras')
    artefactos = joblib.load('artefactos_lstm.pkl')
    return model, artefactos

try:
    model, artefactos = load_assets()
    scaler = artefactos['scaler']
    le = artefactos['label_encoder']
    cols_estaticas = artefactos['cols_estaticas']
    cols_sem1 = artefactos['cols_sem1']
    cols_sem2 = artefactos['cols_sem2']
    columnas_orden = artefactos['columnas_orden_scaler']
except Exception as e:
    st.error(f"⚠️ Error al cargar los archivos del modelo LSTM: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. Encabezado de la Aplicación
# ---------------------------------------------------------
st.title("🧠 Sistema Predictivo Temporal con Red Neuronal LSTM")
st.markdown("""
Esta aplicación utiliza una **Red Neuronal Recurrente (LSTM)** entrenada sobre secuencias semestrales 
($t_1$: 1er Semestre $\rightarrow$ $t_2$: 2do Semestre) para proyectar la trayectoria académica y el riesgo de deserción.
""")

st.divider()

# ---------------------------------------------------------
# 4. Formulario de Entrada en Barra Lateral
# ---------------------------------------------------------
st.sidebar.header("📌 Perfil Temporal del Estudiante")

with st.sidebar.form("lstm_student_form"):
    st.subheader("⏱️ Paso Temporal 1 (1er Semestre)")
    c1_enrolled = st.number_input("Materias matriculadas (1er Sem.)", 0, 25, 6)
    c1_approved = st.slider("Materias aprobadas (1er Sem.)", 0, 20, 5)
    c1_grade = st.slider("Nota promedio (1er Sem., 0-20)", 0.0, 20.0, 12.5)
    
    st.subheader("⏱️ Paso Temporal 2 (2do Semestre)")
    c2_enrolled = st.number_input("Materias matriculadas (2do Sem.)", 0, 25, 6)
    c2_approved = st.slider("Materias aprobadas (2do Sem.)", 0, 20, 5)
    c2_grade = st.slider("Nota promedio (2do Sem., 0-20)", 0.0, 20.0, 12.5)
    
    st.subheader("📋 Contexto Socioeconómico")
    age = st.number_input("Edad al matricularse", min_value=15, max_value=80, value=20)
    tuition_fees = st.selectbox("¿Cuotas de matrícula al día?", options=[1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
    scholarship = st.selectbox("¿Tiene Beca?", options=[1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
    debtor = st.selectbox("¿Es deudor de la institución?", options=[1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
    gender = st.selectbox("Género", options=[1, 0], format_func=lambda x: "Femenino" if x == 1 else "Masculino")
    
    submit_btn = st.form_submit_button("🔮 Evaluar Secuencia con LSTM")

# ---------------------------------------------------------
# 5. Procesamiento de Secuencia 3D y Predicción
# ---------------------------------------------------------
if submit_btn:
    # Construir fila base en el mismo orden exacto en que se entrenó el StandardScaler
    input_dict = {col: [0.0] for col in columnas_orden}
    
    # Asignar variables estáticas
    input_dict['Age at enrollment'] = [float(age)]
    input_dict['Tuition fees up to date'] = [float(tuition_fees)]
    input_dict['Scholarship holder'] = [float(scholarship)]
    input_dict['Debtor'] = [float(debtor)]
    input_dict['Gender'] = [float(gender)]
    
    # Asignar variables temporales Semestre 1
    input_dict['Curricular units 1st sem (enrolled)'] = [float(c1_enrolled)]
    input_dict['Curricular units 1st sem (evaluations)'] = [float(c1_enrolled)]
    input_dict['Curricular units 1st sem (approved)'] = [float(c1_approved)]
    input_dict['Curricular units 1st sem (grade)'] = [float(c1_grade)]
    
    # Asignar variables temporales Semestre 2
    input_dict['Curricular units 2nd sem (enrolled)'] = [float(c2_enrolled)]
    input_dict['Curricular units 2nd sem (evaluations)'] = [float(c2_enrolled)]
    input_dict['Curricular units 2nd sem (approved)'] = [float(c2_approved)]
    input_dict['Curricular units 2nd sem (grade)'] = [float(c2_grade)]
    
    df_raw = pd.DataFrame(input_dict)[columnas_orden]
    
    # Escalar con el StandardScaler entrenado
    df_scaled = pd.DataFrame(scaler.transform(df_raw), columns=columnas_orden)
    
    # Reconstruir el tensor 3D: (1 muestra, 2 pasos de tiempo, 30 características)
    paso_t1 = np.hstack([df_scaled[cols_estaticas].values, df_scaled[cols_sem1].values])
    paso_t2 = np.hstack([df_scaled[cols_estaticas].values, df_scaled[cols_sem2].values])
    X_seq_input = np.stack([paso_t1, paso_t2], axis=1)
    
    # Predicción con la Red Neuronal LSTM
    probabilities = model.predict(X_seq_input, verbose=0)[0]
    pred_idx = np.argmax(probabilities)
    pred_class = le.inverse_transform([pred_idx])[0]
    
    prob_df = pd.DataFrame({
        'Estado': le.classes_,
        'Probabilidad (%)': np.round(probabilities * 100, 2)
    })
    
    # Proyección binaria ajustada (Graduación vs Deserción)
    p_drop = prob_df.loc[prob_df['Estado'] == 'Dropout', 'Probabilidad (%)'].values[0]
    p_grad = prob_df.loc[prob_df['Estado'] == 'Graduate', 'Probabilidad (%)'].values[0]
    total_def = p_drop + p_grad
    
    p_grad_adj = np.round((p_grad / total_def) * 100, 1) if total_def > 0 else 50.0
    p_drop_adj = np.round((p_drop / total_def) * 100, 1) if total_def > 0 else 50.0
    
    df_binario = pd.DataFrame({
        'Proyección Final': ['Graduación Presunta', 'Riesgo de Deserción'],
        'Probabilidad (%)': [p_grad_adj, p_drop_adj]
    }).sort_values(by='Probabilidad (%)', ascending=False).reset_index(drop=True)
    
    top_prob = df_binario['Probabilidad (%)'].iloc[0]

    # ---------------------------------------------------------
    # 6. Visualización de Resultados y Diagnóstico
    # ---------------------------------------------------------
    col_res1, col_res2 = st.columns([1, 1])
    
    with col_res1:
        st.subheader("🎯 Diagnóstico de la Red LSTM")
        if df_binario['Proyección Final'].iloc[0] == 'Riesgo de Deserción':
            st.error(f"⚠️ **Riesgo Alto de Deserción Académica** ({top_prob}% proyección ajustada)")
        else:
            st.success(f"🎓 **Alta Probabilidad de Graduación** ({top_prob}% proyección ajustada)")
            
        st.caption(f"Salida directa Softmax LSTM — Dropout: {p_drop}% | Enrolled: {prob_df.loc[prob_df['Estado'] == 'Enrolled', 'Probabilidad (%)'].values[0]}% | Graduate: {p_grad}%")
        st.write("---")
        st.write("### 🔝 Proyección de Estado Final:")
        st.dataframe(df_binario, use_container_width=True, hide_index=True)
        
        st.markdown("### 💡 Factores Detectados en la Trayectoria:")
        motivos = []
        if c2_grade < c1_grade:
            motivos.append(f"📉 **Tendencia temporal descendente:** La nota cayó de {c1_grade} a {c2_grade} entre semestres.")
        elif c2_grade > c1_grade:
            motivos.append(f"📈 **Evolución académica positiva:** La nota subió de {c1_grade} a {c2_grade} en el 2do semestre.")
        if tuition_fees == 0 or debtor == 1:
            motivos.append("🚨 **Alerta financiera:** Registra deudas o cuotas de matrícula pendientes.")
        else:
            motivos.append("✅ **Estabilidad financiera:** Cuotas al día y sin registros de deuda activa.")
        if c2_approved < 3 or c2_grade < 10.0:
            motivos.append("🚨 **Bajo rendimiento reciente:** Pocas materias aprobadas o nota reprobatoria en el 2do semestre.")
            
        for m in motivos:
            st.markdown(m)

    with col_res2:
        st.subheader("📊 Distribución de Probabilidades")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.barplot(
            data=df_binario,
            x='Probabilidad (%)',
            y='Proyección Final',
            palette=['#00c0f2' if x == 'Graduación Presunta' else '#ff4b4b' for x in df_binario['Proyección Final']],
            ax=ax
        )
        ax.set_xlim(0, 105)
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f'{width:.1f}%', (width + 3, p.get_y() + p.get_height() / 2),
                        ha='center', va='center', fontsize=9, fontweight='bold')
        sns.despine()
        st.pyplot(fig)

    st.divider()

    # ---------------------------------------------------------
    # 7. Tablero Comparativo Temporal
    # ---------------------------------------------------------
    st.subheader("📈 Evolución Temporal del Estudiante (Semestre 1 vs Semestre 2)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Nota 1er Semestre", f"{c1_grade} pts")
    kpi2.metric("Nota 2do Semestre", f"{c2_grade} pts", delta=f"{round(c2_grade - c1_grade, 2)} pts")
    kpi3.metric("Aprobadas 1er Sem.", f"{c1_approved}")
    kpi4.metric("Aprobadas 2do Sem.", f"{c2_approved}", delta=f"{c2_approved - c1_approved} materias")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        df_evol = pd.DataFrame({
            'Semestre': ['1er Semestre', '1er Semestre', '2do Semestre', '2do Semestre'],
            'Métrica': ['Nota Promedio', 'Materias Aprobadas', 'Nota Promedio', 'Materias Aprobadas'],
            'Valor': [c1_grade, c1_approved, c2_grade, c2_approved]
        })
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df_evol, x='Semestre', y='Valor', hue='Métrica', palette=['#1f77b4', '#ff7f0e'], ax=ax_bar)
        ax_bar.set_title("Comparativa Semestral")
        sns.despine()
        st.pyplot(fig_bar)

    with col_g2:
        labels = ['Prom. 1er Sem', 'Aprob. 1er Sem', 'Prom. 2do Sem', 'Aprob. 2do Sem', 'Pago al día']
        values = [
            (c1_grade / 20.0) * 100,
            min((c1_approved / 10.0) * 100, 100),
            (c2_grade / 20.0) * 100,
            min((c2_approved / 10.0) * 100, 100),
            100 if tuition_fees == 1 else 0
        ]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        fig_radar, ax_radar = plt.subplots(figsize=(6, 4), subplot_kw=dict(polar=True))
        ax_radar.plot(angles, values, color='#2ca02c', linewidth=2)
        ax_radar.fill(angles, values, color='#2ca02c', alpha=0.25)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(labels, fontsize=8)
        ax_radar.set_title("Balance Académico y Financiero", y=1.08)
        st.pyplot(fig_radar)
else:
    st.info("👈 Configura las métricas del 1er y 2do semestre en el panel izquierdo para ejecutar la predicción temporal con la red LSTM.")