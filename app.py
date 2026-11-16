"""Simulador local de control PID para una celda de flotación de minerales.

Ejecute: streamlit run app.py
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


st.set_page_config(page_title="Celda de flotación · PID", page_icon="⚙️", layout="wide")

# Modelo linealizado didáctico: G(s) = -0.8 / (120s + 1)
TS, TAU, K_PROCESO = 1.0, 120.0, -0.8
PV_BASE, CV_BASE, QIN_BASE, H_MAX, HISTORIAL = 60.0, 45.0, 100.0, 1.30, 180


def iniciar_simulacion(forzar: bool = False) -> None:
    """Crea el estado de proceso que persiste entre actualizaciones de Streamlit."""
    if "sim" in st.session_state and not forzar:
        return
    st.session_state.sim = {"t": 0.0, "pv": PV_BASE, "cv": CV_BASE, "integral": 0.0,
                            "error_anterior": 0.0, "pulso_alimentacion": 0.0,
                            "historial": deque(maxlen=HISTORIAL)}
    for segundo in range(60):
        st.session_state.sim["historial"].append({"t": float(segundo - 59), "pv": PV_BASE,
            "sp": float(st.session_state.get("sp", PV_BASE)), "cv": CV_BASE,
            "ma": 4.0 + 16.0 * PV_BASE / 100.0, "p": 0.0, "i": 0.0, "d": 0.0,
            "q_in": QIN_BASE})


def reiniciar() -> None:
    iniciar_simulacion(forzar=True)


def aplicar_pulso() -> None:
    st.session_state.sim["pulso_alimentacion"] += 15.0


def calcular_simc() -> tuple[float, float, float]:
    """Ajuste SIMC/IMC para FOPDT, expresado como PID paralelo."""
    lam, retraso = float(st.session_state.lambda_simc), 5.0
    kc = TAU / (abs(K_PROCESO) * (lam + retraso))
    ti = min(TAU, 4.0 * (lam + retraso))
    return kc, kc / ti, 0.0  # Para un proceso de primer orden, SIMC recomienda PI.


def aplicar_simc() -> None:
    kp, ki, kd = calcular_simc()
    st.session_state.kp, st.session_state.ki, st.session_state.kd = round(kp, 3), round(ki, 4), round(kd, 3)


def avanzar_un_ciclo() -> dict[str, float]:
    """Ejecuta PID discreto, anti-windup y el modelo de primer orden de la celda."""
    sim = st.session_state.sim
    sp, q_in = float(st.session_state.sp), float(st.session_state.q_in)
    kp, ki, kd = float(st.session_state.kp), float(st.session_state.ki), float(st.session_state.kd)
    pv, automatico = float(sim["pv"]), bool(st.session_state.automatico)
    error = sp - pv
    p = kp * error
    d = kd * (error - float(sim["error_anterior"])) / TS
    integral_candidata = float(sim["integral"]) + ki * error * TS

    if automatico:
        # Acción inversa: nivel bajo cierra descarga; nivel alto la abre.
        cv_sin_limite = CV_BASE - (p + integral_candidata + d)
        if 0.0 < cv_sin_limite < 100.0 or (cv_sin_limite <= 0.0 and error < 0.0) or (cv_sin_limite >= 100.0 and error > 0.0):
            sim["integral"] = integral_candidata
        i = float(sim["integral"])
        cv = float(np.clip(CV_BASE - (p + i + d), 0.0, 100.0))
    else:
        cv = float(st.session_state.cv_manual)
        sim["integral"] = CV_BASE - cv - p - d  # Transferencia suave a automático.
        i = float(sim["integral"])

    pulso = float(sim["pulso_alimentacion"])
    sim["pulso_alimentacion"] = pulso * 0.965
    perturbacion = 0.32 * (q_in - QIN_BASE + pulso)
    pv_objetivo = PV_BASE + K_PROCESO * (cv - CV_BASE) + perturbacion
    pv_nuevo = float(np.clip(pv + (pv_objetivo - pv) * TS / TAU, 0.0, 100.0))
    ma = 4.0 + 16.0 * pv_nuevo / 100.0
    sim["t"], sim["pv"], sim["cv"], sim["error_anterior"] = sim["t"] + TS, pv_nuevo, cv, error
    fila = {"t": float(sim["t"]), "pv": pv_nuevo, "sp": sp, "cv": cv, "ma": ma,
            "p": p, "i": i, "d": d, "q_in": q_in + pulso}
    sim["historial"].append(fila)
    return fila


def figura_celda(pv: float, cv: float, q_in: float) -> go.Figure:
    """Diagrama de proceso de celda con nivel y válvula actualizados en vivo."""
    altura = H_MAX * pv / 100.0
    x_onda = np.linspace(0.8, 4.2, 80)
    figura = go.Figure()
    figura.add_shape(type="rect", x0=0.7, x1=4.3, y0=0, y1=H_MAX, line={"color": "#9ec9df", "width": 3}, fillcolor="#0f2136")
    figura.add_shape(type="rect", x0=0.72, x1=4.28, y0=0.02, y1=altura, line={"width": 0}, fillcolor="rgba(40,137,210,0.62)")
    figura.add_trace(go.Scatter(x=x_onda, y=altura + 0.018 * np.sin(np.linspace(0, 5 * np.pi, 80)), mode="lines", line={"color": "#d8f5ff", "width": 3}, hovertemplate="Nivel: %{y:.2f} m<extra></extra>"))
    rng = np.random.default_rng(8)
    figura.add_trace(go.Scatter(x=rng.uniform(0.95, 4.0, 40), y=rng.uniform(0.08, max(0.10, altura - 0.05), 40), mode="markers", marker={"size": rng.uniform(4, 12, 40), "color": "#b8efff", "opacity": 0.65}, hoverinfo="skip"))
    figura.add_shape(type="line", x0=2.5, x1=2.5, y0=H_MAX + 0.33, y1=0.23, line={"color": "#ffbd55", "width": 8})
    figura.add_shape(type="circle", x0=1.92, x1=3.08, y0=0.16, y1=0.45, line={"color": "#ffbd55", "width": 3}, fillcolor="#7a4917")
    figura.add_shape(type="line", x0=-0.2, x1=0.7, y0=H_MAX * 0.76, y1=H_MAX * 0.76, line={"color": "#60e6b9", "width": 7})
    figura.add_shape(type="line", x0=4.3, x1=5.1, y0=0.30, y1=0.30, line={"color": "#ffb24e", "width": 7})
    figura.add_annotation(x=-0.1, y=H_MAX * 0.90, text=f"Q<sub>in</sub> = {q_in:.0f} m³/h", showarrow=False, font={"color": "#bfffea", "size": 15})
    figura.add_annotation(x=5.2, y=0.30, text=f"LV-101<br>{cv:.1f}%", showarrow=False, font={"color": "#ffd391", "size": 15})
    figura.add_annotation(x=2.5, y=H_MAX + 0.43, text="Agitador", showarrow=False, font={"color": "#ffd391"})
    figura.add_annotation(x=2.5, y=altura + 0.12, text=f"PV = {pv:.1f}% = {altura:.2f} m", showarrow=False, font={"color": "#f5fbff", "size": 15})
    figura.update_layout(height=390, margin={"l": 5, "r": 25, "t": 30, "b": 5}, paper_bgcolor="#101b2d", plot_bgcolor="#101b2d", showlegend=False, xaxis={"range": [-0.6, 5.9], "visible": False}, yaxis={"range": [0, H_MAX + 0.6], "visible": False})
    return figura


def figura_tendencias(datos: pd.DataFrame) -> go.Figure:
    figura = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Nivel de pulpa: PV vs SP", "Salida del controlador: válvula LV-101", "Transmisor LT-101: señal de corriente"))
    figura.add_trace(go.Scatter(x=datos.t, y=datos.sp, name="SP", line={"color": "#a9b8c8", "dash": "dash", "width": 2}), row=1, col=1)
    figura.add_trace(go.Scatter(x=datos.t, y=datos.pv, name="PV", line={"color": "#5ee3a1", "width": 3}), row=1, col=1)
    figura.add_trace(go.Scatter(x=datos.t, y=datos.cv, name="Válvula", line={"color": "#ffb24e", "width": 3}), row=2, col=1)
    figura.add_trace(go.Scatter(x=datos.t, y=datos.ma, name="LT-101", line={"color": "#47d9eb", "width": 3}), row=3, col=1)
    figura.update_yaxes(title_text="%", range=[0, 100], row=1, col=1)
    figura.update_yaxes(title_text="%", range=[0, 100], row=2, col=1)
    figura.update_yaxes(title_text="mA", range=[4, 20], row=3, col=1)
    figura.update_xaxes(title_text="Tiempo simulado (s)", row=3, col=1)
    figura.update_layout(template="plotly_dark", height=670, margin={"l": 45, "r": 20, "t": 70, "b": 45}, paper_bgcolor="#101b2d", plot_bgcolor="#101b2d", legend={"orientation": "h", "y": 1.05})
    return figura


def figura_ladder(fila: dict[str, float], automatico: bool) -> go.Figure:
    rungs = [("AI → escalado de LT-101", "PV = (AI − 4) / 16 × 100", f"{fila['ma']:.2f} mA"),
             ("Automático", "SUB · e = SP − PV", f"{st.session_state.sp - fila['pv']:+.2f} %"),
             ("Automático", "MUL · P = Kp × e", f"{fila['p']:+.2f} %"),
             ("Auto + anti-windup", "ADD · I = Iₖ₋₁ + Ki × Ts × e", f"{fila['i']:+.2f} %"),
             ("Automático", "MUL · D = Kd × Δe / Ts", f"{fila['d']:+.2f} %"),
             ("Automático", "SUB · CV = 45 − (P + I + D)", f"{fila['cv']:.2f} %"),
             ("AO → válvula LV-101", "SCP · 0..100 % → 4..20 mA", f"{4 + 16 * fila['cv'] / 100:.2f} mA")]
    figura, total = go.Figure(), len(rungs)
    figura.add_shape(type="line", x0=0.35, x1=0.35, y0=0.25, y1=total + 0.55, line={"color": "#c7e7ec", "width": 3})
    figura.add_shape(type="line", x0=9.65, x1=9.65, y0=0.25, y1=total + 0.55, line={"color": "#c7e7ec", "width": 3})
    for indice, (contacto, operacion, valor) in enumerate(rungs):
        y, color = total - indice, ("#66e7b3" if automatico or indice in (0, 6) else "#68768b")
        figura.add_shape(type="line", x0=0.35, x1=2.0, y0=y, y1=y, line={"color": color, "width": 2})
        figura.add_shape(type="rect", x0=2.0, x1=7.55, y0=y - 0.32, y1=y + 0.32, line={"color": "#5683a9", "width": 2}, fillcolor="#16314e")
        figura.add_shape(type="line", x0=7.55, x1=9.65, y0=y, y1=y, line={"color": color, "width": 2})
        figura.add_annotation(x=1.16, y=y, text=f"—| |—<br>{contacto}", showarrow=False, font={"color": color, "size": 11})
        figura.add_annotation(x=4.78, y=y, text=operacion, showarrow=False, font={"color": "#e7f3ff", "size": 13})
        figura.add_annotation(x=8.58, y=y, text=f"<b>{valor}</b>", showarrow=False, font={"color": "#b9f4ff", "size": 13})
        figura.add_annotation(x=0.12, y=y, text=str(indice + 1), showarrow=False, font={"color": "#93a9b7", "size": 11})
    figura.update_layout(template="plotly_dark", height=540, margin={"l": 0, "r": 0, "t": 15, "b": 8}, paper_bgcolor="#101b2d", plot_bgcolor="#101b2d", xaxis={"visible": False, "range": [0, 10]}, yaxis={"visible": False, "range": [0, total + 0.8]})
    return figura


def diagrama_control(fila: dict[str, float]) -> str:
    return f'''digraph PID {{
      graph [rankdir=LR, bgcolor="#101b2d", pad="0.25", nodesep="0.45", ranksep="0.7"];
      node [shape=box, style="rounded,filled", fillcolor="#183450", color="#75a6cf", fontcolor="#ecf6ff", fontname="Arial", fontsize=12, margin="0.20,0.12"];
      edge [color="#8cbfd4", fontcolor="#cdeaf7", fontname="Arial", fontsize=10, penwidth=1.8];
      SP [label="Consigna\\nSP = {st.session_state.sp:.1f} %", fillcolor="#174d47"];
      SUM [shape=circle, label="Σ\\ne", fillcolor="#26334d", width="0.65", height="0.65"];
      PID [label="CONTROLADOR PID\\nP={fila['p']:+.2f} | I={fila['i']:+.2f} | D={fila['d']:+.2f}", fillcolor="#26395e"];
      VALVULA [label="LV-101\\nCV = {fila['cv']:.1f} %", fillcolor="#5a3e1c"];
      CELDA [label="CELDA DE FLOTACIÓN\\nG(s) = −0.8 / (120s + 1)", fillcolor="#16405a"];
      PV [label="Nivel de pulpa\\nPV = {fila['pv']:.1f} %", fillcolor="#174d47"];
      SENSOR [label="LT-101\\n{fila['ma']:.2f} mA", fillcolor="#26395e"];
      SP -> SUM [label=" + "]; SUM -> PID [label=" e = SP − PV "]; PID -> VALVULA [label=" u "];
      VALVULA -> CELDA [label=" Qout "]; CELDA -> PV [label=" h "]; PV -> SENSOR [label=" medición "];
      SENSOR -> SUM [label=" − ", constraint=false]; }}'''


def mostrar_panel_vivo() -> None:
    fila = avanzar_un_ciclo() if st.session_state.en_marcha else st.session_state.sim["historial"][-1]
    datos, nivel_m = pd.DataFrame(st.session_state.sim["historial"]), H_MAX * fila["pv"] / 100.0
    st.subheader("Proceso de flotación y lazo de control")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Consigna SP", f"{fila['sp']:.1f} %")
    c2.metric("Nivel PV", f"{fila['pv']:.1f} %", f"{fila['pv'] - fila['sp']:+.1f} %")
    c3.metric("Nivel físico", f"{nivel_m:.2f} m")
    c4.metric("Válvula LV-101", f"{fila['cv']:.1f} %")
    c5.metric("LT-101", f"{fila['ma']:.2f} mA")
    proceso, bloques = st.columns((1, 1.25), gap="large")
    with proceso:
        st.plotly_chart(figura_celda(fila["pv"], fila["cv"], fila["q_in"]), width="stretch", config={"displayModeBar": False})
    with bloques:
        st.graphviz_chart(diagrama_control(fila), width="stretch")
    st.plotly_chart(figura_tendencias(datos), width="stretch", config={"displayModeBar": False})
    st.subheader("Lógica del ciclo de control · Ladder didáctico en vivo")
    st.caption("Cada renglón muestra el cálculo que está ejecutando el simulador en este instante. Es una representación de referencia: para PLC real debe adaptarse y validarse en el fabricante correspondiente.")
    st.plotly_chart(figura_ladder(fila, st.session_state.automatico), width="stretch", config={"displayModeBar": False})


def configurar_estado() -> None:
    for clave, valor in {"kp": 2.0, "ki": 0.020, "kd": 0.0, "sp": 60.0, "q_in": 100.0,
                         "cv_manual": 45.0, "automatico": True, "en_marcha": True,
                         "lambda_simc": 30.0}.items():
        st.session_state.setdefault(clave, valor)
    iniciar_simulacion()


configurar_estado()
st.title("Simulador PID · Celda de flotación de minerales")
st.caption("Aplicación local en Python. Ajusta parámetros sin reiniciar y observa proceso, gráficas, esquema PID y Ladder cambiar en tiempo real.")

with st.sidebar:
    st.header("Panel de operación")
    st.toggle("Ejecutar simulación", key="en_marcha")
    st.toggle("Modo automático PID", key="automatico")
    st.slider("Consigna de nivel SP (%)", 20.0, 90.0, step=0.5, key="sp")
    st.slider("Caudal de alimentación Qᵢₙ (m³/h)", 60.0, 145.0, step=1.0, key="q_in")
    st.slider("Válvula manual (%)", 0.0, 100.0, step=0.5, key="cv_manual", disabled=st.session_state.automatico)
    b1, b2 = st.columns(2)
    b1.button("Pulso +15", use_container_width=True, on_click=aplicar_pulso, help="Perturbación temporal de caudal de entrada.")
    b2.button("Reiniciar", use_container_width=True, on_click=reiniciar)
    st.divider(); st.header("Ganancias PID")
    st.slider("Kp · proporcional", 0.0, 8.0, step=0.05, key="kp")
    st.slider("Ki · integral (1/s)", 0.0, 0.20, step=0.001, key="ki")
    st.slider("Kd · derivativa (s)", 0.0, 8.0, step=0.05, key="kd")
    st.caption("Acción inversa: si PV sube sobre SP, el controlador abre la descarga para bajar el nivel.")

with st.expander("Modelo matemático y PID que ejecuta el simulador", expanded=True):
    a, b = st.columns(2)
    with a:
        st.markdown("**Balance de pulpa**"); st.latex(r"A\frac{dh}{dt}=Q_{in}-Q_{out}")
        st.markdown("**Modelo linealizado para la simulación**"); st.latex(r"G(s)=\frac{-0.8}{120s+1}")
        st.caption("El signo negativo significa que, al abrir la válvula de descarga, el nivel disminuye.")
    with b:
        st.markdown("**PID discreto de acción inversa**"); st.latex(r"e=SP-PV\qquad u=\mathrm{sat}[45-(P+I+D)]")
        st.latex(r"P=K_pe\quad I_k=I_{k-1}+K_iT_se\quad D=K_d\frac{e_k-e_{k-1}}{T_s}")
        st.caption("Incluye saturación 0–100 %, anti-windup y transferencia suave de manual a automático.")

with st.expander("Sintonización SIMC (IMC) para el modelo identificado", expanded=True):
    st.markdown("Se usa un retardo identificado de 5 s. Para este proceso de primer orden, SIMC recomienda un controlador PI; por eso Kd se propone inicialmente en cero.")
    col_lam, col_resultado, col_aplicar = st.columns((1, 2, 1))
    with col_lam: st.slider("Respuesta λ (s)", 10.0, 120.0, step=5.0, key="lambda_simc")
    kp_simc, ki_simc, kd_simc = calcular_simc()
    with col_resultado: st.code(f"Kc = τ / [|K|(λ + θ)] = {kp_simc:.3f}\nTi = min(τ, 4(λ + θ))\nKp = {kp_simc:.3f}    Ki = {ki_simc:.4f} s⁻¹    Kd = {kd_simc:.3f} s")
    with col_aplicar: st.button("Calcular y aplicar", type="primary", use_container_width=True, on_click=aplicar_simc)

if hasattr(st, "fragment"):
    @st.fragment(run_every=0.6)
    def actualizacion_viva() -> None:
        mostrar_panel_vivo()
    actualizacion_viva()
else:
    st.warning("Actualiza Streamlit a la versión indicada en requirements.txt para obtener actualización automática cada 0.6 s.")
    mostrar_panel_vivo()
