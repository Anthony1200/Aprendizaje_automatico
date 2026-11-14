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
# 4. Panel Lateral Reactivo (Validación Aprobadas <= Matriculadas)
# ---------------------------------------------------------
st.sidebar.header("📌 Perfil Temporal del Estudiante")

st.sidebar.subheader("⏱️ Paso Temporal 1 (1er Semestre)")
c1_enrolled = st.sidebar.number_input("Materias matriculadas (1er Sem.)", min_value=1, max_value=25, value=6)
c1_approved = st.sidebar.slider("Materias aprobadas (1er Sem.)", min_value=0, max_value=int(c1_enrolled), value=min(5, int(c1_enrolled)))
c1_grade = st.sidebar.slider("Nota promedio (1er Sem., 0-20)", min_value=0.0, max_value=20.0, value=12.5, step=0.1)

st.sidebar.subheader("⏱️ Paso Temporal 2 (2do Semestre)")
c2_enrolled = st.sidebar.number_input("Materias matriculadas (2do Sem.)", min_value=1, max_value=25, value=6)
c2_approved = st.sidebar.slider("Materias aprobadas (2do Sem.)", min_value=0, max_value=int(c2_enrolled), value=min(5, int(c2_enrolled)))
c2_grade = st.sidebar.slider("Nota promedio (2do Sem., 0-20)", min_value=0.0, max_value=20.0, value=12.5, step=0.1)

st.sidebar.subheader("📋 Contexto Socioeconómico")
age = st.sidebar.number_input("Edad al matricularse", min_value=15, max_value=80, value=20)
tuition_fees = st.sidebar.selectbox("¿Cuotas de matrícula al día?", options=[1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
scholarship = st.sidebar.selectbox("¿Tiene Beca?", options=[1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
debtor = st.sidebar.selectbox("¿Es deudor de la institución?", options=[1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
gender = st.sidebar.selectbox("Género", options=[1, 0], format_func=lambda x: "Femenino" if x == 1 else "Masculino")

submit_btn = st.sidebar.button("🔮 Evaluar Secuencia con LSTM", use_container_width=True, type="primary")

# ---------------------------------------------------------
# 5. Procesamiento de Secuencia 3D y Predicción
# ---------------------------------------------------------
if submit_btn:
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
    df_scaled = pd.DataFrame(scaler.transform(df_raw), columns=columnas_orden)
    
    # Reconstruir tensor 3D: (1 muestra, 2 pasos de tiempo, 30 características)
    paso_t1 = np.hstack([df_scaled[cols_estaticas].values, df_scaled[cols_sem1].values])
    paso_t2 = np.hstack([df_scaled[cols_estaticas].values, df_scaled[cols_sem2].values])
    X_seq_input = np.stack([paso_t1, paso_t2], axis=1)
    
    # Predicción con la Red Neuronal LSTM
    probabilities = model.predict(X_seq_input, verbose=0)[0]
    
    prob_df = pd.DataFrame({
        'Estado': le.classes_,
        'Probabilidad': probabilities * 100
    })
    
    # Proyección binaria ajustada (Graduación vs Deserción) con 1 decimal exacto
    p_drop = float(prob_df.loc[prob_df['Estado'] == 'Dropout', 'Probabilidad'].values[0])
    p_grad = float(prob_df.loc[prob_df['Estado'] == 'Graduate', 'Probabilidad'].values[0])
    total_def = p_drop + p_grad
    
    p_grad_adj = round((p_grad / total_def) * 100, 1) if total_def > 0 else 50.0
    p_drop_adj = round(100.0 - p_grad_adj, 1)
    
    df_binario = pd.DataFrame({
        'Proyección Final': ['Graduación Presunta', 'Riesgo de Deserción'],
        'Probabilidad (%)': [p_grad_adj, p_drop_adj]
    }).sort_values(by='Probabilidad (%)', ascending=False).reset_index(drop=True)
    
    top_estado = df_binario['Proyección Final'].iloc[0]
    top_prob = df_binario['Probabilidad (%)'].iloc[0]

    # ---------------------------------------------------------
    # 6. Visualización de Resultados y Diagnóstico
    # ---------------------------------------------------------
    col_res1, col_res2 = st.columns([1.1, 0.9])
    
    with col_res1:
        st.subheader("🎯 Diagnóstico de la Red LSTM")
        if top_estado == 'Riesgo de Deserción':
            st.error(f"⚠️ **Riesgo Alto de Deserción Académica** ({top_prob:.1f}% proyección ajustada)")
        else:
            st.success(f"🎓 **Alta Probabilidad de Graduación** ({top_prob:.1f}% proyección ajustada)")
            
        st.write("### 🔝 Proyección de Estado Final:")
        df_tabla = df_binario.copy()
        df_tabla['Probabilidad (%)'] = df_tabla['Probabilidad (%)'].map(lambda x: f"{x:.1f}%")
        st.dataframe(df_tabla, use_container_width=True, hide_index=True)
        
        st.markdown("### 💡 Factores Detectados en la Trayectoria:")
        tasa_aprob_1 = (c1_approved / c1_enrolled) * 100
        tasa_aprob_2 = (c2_approved / c2_enrolled) * 100
        
        motivos = []
        if c2_grade < c1_grade:
            motivos.append(f"📉 **Caída de calificación:** El promedio descendió de **{c1_grade:.1f}** a **{c2_grade:.1f}** entre semestres.")
        elif c2_grade > c1_grade:
            motivos.append(f"📈 **Mejora de calificación:** El promedio subió de **{c1_grade:.1f}** a **{c2_grade:.1f}** en el 2do semestre.")
        else:
            motivos.append(f"➖ **Estabilidad en notas:** El estudiante mantiene un promedio constante de **{c2_grade:.1f}**.")
            
        if tasa_aprob_2 < 50:
            motivos.append(f"🚨 **Baja tasa de aprobación reciente:** Solo aprobó el **{tasa_aprob_2:.1f}%** ({c2_approved}/{c2_enrolled}) de las materias en el 2do semestre.")
        else:
            motivos.append(f"✅ **Buena efectividad académica:** Aprobó el **{tasa_aprob_2:.1f}%** ({c2_approved}/{c2_enrolled}) de su carga académica reciente.")
            
        if tuition_fees == 0 or debtor == 1:
            motivos.append("🚨 **Alerta financiera:** Registra deudas o cuotas de matrícula pendientes.")
        else:
            motivos.append("✅ **Estabilidad financiera:** Cuotas al día y sin registros de deuda activa.")
            
        for m in motivos:
            st.markdown(m)

    with col_res2:
        st.subheader("📊 Distribución de Proyección")
        # Gráfico de Dona (Donut Chart)
        fig_donut, ax_donut = plt.subplots(figsize=(5.5, 4))
        colores = ['#00c0f2' if est == 'Graduación Presunta' else '#ff4b4b' for est in df_binario['Proyección Final']]
        
        wedges, texts, autotexts = ax_donut.pie(
            df_binario['Probabilidad (%)'],
            labels=df_binario['Proyección Final'],
            autopct='%1.1f%%',
            startangle=90,
            colors=colores,
            pctdistance=0.78,
            wedgeprops=dict(width=0.42, edgecolor='white', linewidth=2),
            textprops=dict(fontsize=10, fontweight='bold')
        )
        for autotext in autotexts:
            autotext.set_color('white')
            
        # Texto central en la dona
        ax_donut.text(0, 0, f"{top_prob:.1f}%\n{top_estado.split()[0]}", ha='center', va='center', fontsize=12, fontweight='bold', color='#333333')
        ax_donut.axis('equal')
        st.pyplot(fig_donut)

    st.divider()

    # ---------------------------------------------------------
    # 7. Tablero Analítico Temporal (Nuevas Gráficas)
    # ---------------------------------------------------------
    st.subheader("📈 Analítica Temporal de Carga y Rendimiento Académico")
    
    tasa_1 = round((c1_approved / c1_enrolled) * 100, 1)
    tasa_2 = round((c2_approved / c2_enrolled) * 100, 1)
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Tasa Aprobación 1er Sem.", f"{tasa_1:.1f}%", f"{c1_approved}/{c1_enrolled} materias")
    kpi2.metric("Tasa Aprobación 2do Sem.", f"{tasa_2:.1f}%", delta=f"{round(tasa_2 - tasa_1, 1)}%")
    kpi3.metric("Nota Promedio 1er Sem.", f"{c1_grade:.1f} pts")
    kpi4.metric("Nota Promedio 2do Sem.", f"{c2_grade:.1f} pts", delta=f"{round(c2_grade - c1_grade, 1)} pts")

    st.write("")
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("**📚 Carga Académica: Materias Aprobadas vs. Reprobadas**")
        semestres = ['1er Semestre ($t_1$)', '2do Semestre ($t_2$)']
        aprobadas = np.array([c1_approved, c2_approved])
        no_aprobadas = np.array([c1_enrolled - c1_approved, c2_enrolled - c2_approved])
        
        fig_stack, ax_stack = plt.subplots(figsize=(6, 4))
        ax_stack.bar(semestres, aprobadas, label='Aprobadas', color='#2ca02c', width=0.45)
        ax_stack.bar(semestres, no_aprobadas, bottom=aprobadas, label='No Aprobadas / Pendientes', color='#d62728', alpha=0.8, width=0.45)
        
        for i in range(len(semestres)):
            if aprobadas[i] > 0:
                ax_stack.text(i, aprobadas[i] / 2, f"{aprobadas[i]}", ha='center', va='center', color='white', fontweight='bold')
            if no_aprobadas[i] > 0:
                ax_stack.text(i, aprobadas[i] + (no_aprobadas[i] / 2), f"{no_aprobadas[i]}", ha='center', va='center', color='white', fontweight='bold')
                
        ax_stack.set_ylabel("Cantidad de Materias")
        ax_stack.set_ylim(0, max(c1_enrolled, c2_enrolled) + 3)
        ax_stack.legend(loc='upper right')
        sns.despine()
        st.pyplot(fig_stack)

    with col_g2:
        st.markdown("**📉 Trayectoria Temporal: Efectividad (%) vs. Nota Normalizada (%)**")
        nota_pct_1 = (c1_grade / 20.0) * 100
        nota_pct_2 = (c2_grade / 20.0) * 100
        
        fig_line, ax_line = plt.subplots(figsize=(6, 4))
        ax_line.plot(semestres, [tasa_1, tasa_2], marker='o', markersize=8, linewidth=2.5, color='#1f77b4', label='Tasa de Aprobación (%)')
        ax_line.fill_between(semestres, [tasa_1, tasa_2], color='#1f77b4', alpha=0.12)
        
        ax_line.plot(semestres, [nota_pct_1, nota_pct_2], marker='s', markersize=8, linewidth=2.5, linestyle='--', color='#ff7f0e', label='Nota Promedio (Escala 0-100%)')
        
        for i, txt in enumerate([tasa_1, tasa_2]):
            ax_line.annotate(f"{txt:.1f}%", (semestres[i], [tasa_1, tasa_2][i] + 3), ha='center', fontweight='bold', color='#1f77b4')
        for i, txt in enumerate([nota_pct_1, nota_pct_2]):
            ax_line.annotate(f"{txt:.1f}%", (semestres[i], [nota_pct_1, nota_pct_2][i] - 6), ha='center', fontweight='bold', color='#ff7f0e')
            
        ax_line.set_ylim(0, 112)
        ax_line.set_ylabel("Porcentaje (%)")
        ax_line.grid(True, linestyle=':', alpha=0.5)
        ax_line.legend(loc='lower right')
        sns.despine()
        st.pyplot(fig_line)
else:
    st.info("👈 Configura las materias matriculadas, aprobadas y notas del 1er y 2do semestre en el panel izquierdo para ejecutar la predicción temporal con la red LSTM.")
