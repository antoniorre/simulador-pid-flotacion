# Simulador PID · Celda de Flotación

Aplicación web autónoma en español para estudiar el control PID del nivel de pulpa de una celda de flotación de minerales.

## Publicación en Render

Este repositorio está configurado como un sitio estático: no necesita servidor Python ni Streamlit para funcionar.

1. Cree o conecte un repositorio de GitHub y suba esta carpeta.
2. En Render, elija **New > Blueprint** y seleccione el repositorio.
3. Render detectará `render.yaml`, publicará el archivo `index.html` de la raíz y entregará una URL segura `https://...onrender.com`.

También puede crear manualmente un **Static Site** con estos datos:

- Build Command: `true`
- Publish Directory: `.`

## Ejecución local

Abra `index.html` en un navegador. Es un archivo autónomo; no usa dependencias externas.

## Incluye

- Ajuste en tiempo real de Kp, Ki, Kd, consigna y caudal de alimentación.
- Gráficas de nivel PV/SP, apertura de válvula y corriente de transmisor 4–20 mA.
- Modelo matemático de primer orden, explicación de sintonía y lógica Ladder PLC referencial.
