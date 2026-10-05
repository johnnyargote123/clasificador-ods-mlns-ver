"""
Clasificador de textos por ODS (Microproyecto 2, MLNS).
Pipeline: preprocesamiento en español -> TF-IDF -> TruncatedSVD (1000 componentes) -> LinearSVC.

Ejecutar:  python -m streamlit run app.py
"""
import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# El modelo guardado referencia preprocesamiento.preprocess_text: este import debe existir
# y el archivo debe estar junto a app.py con ese mismo nombre.
from preprocesamiento import preprocess_text  # noqa: F401

NOMBRES_ODS = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educación de calidad",
    5: "Igualdad de género",
    6: "Agua limpia y saneamiento",
    7: "Energía asequible y no contaminante",
    8: "Trabajo decente y crecimiento económico",
    9: "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables",
    13: "Acción por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas",
    17: "Alianzas para lograr los objetivos",
}

# Textos escritos fuera del dataset (sección 8.4 del notebook)
EJEMPLOS = {
    "Energía": "Una cooperativa de energía en un municipio rural instaló paneles solares en las viviendas que no tenían acceso a la red eléctrica, y ahora las familias pueden usar electrodomésticos y estudiar de noche con luz eléctrica.",
    "Género": "La empresa publicó su primer informe de brecha salarial y se comprometió a igualar el sueldo entre hombres y mujeres en el mismo cargo, además de extender la licencia de paternidad a cuatro semanas.",
    "Ciudades": "La alcaldía inauguró una red de ciclorrutas que conecta los barrios periféricos con el centro de la ciudad, buscando reducir la congestión vehicular y bajar los niveles de contaminación del aire.",
    "Agua": "El acueducto veredal terminó la construcción de un nuevo tanque de almacenamiento que garantiza agua potable todos los días de la semana a las familias de la zona, que antes solo tenían servicio dos días.",
}

st.set_page_config(page_title="Clasificador de textos por ODS", page_icon="🌍", layout="wide")


# ---------- Modelo (se carga una sola vez) ----------
@st.cache_resource(show_spinner="Cargando el modelo…")
def cargar_modelo():
    modelo = joblib.load("modelo_ods.pkl")
    tfidf = modelo.named_steps["tfidf"]
    svd = modelo.named_steps["svd"]
    clf = modelo.named_steps["clasificador"]
    # Todo el pipeline es lineal: margen = tfidf(x) · (coef · componentes_svd)ᵀ + intercepto.
    # Por eso se puede calcular cuánto aporta cada término del vocabulario a cada ODS.
    pesos_por_termino = (clf.coef_ @ svd.components_).astype(np.float32)  # (n_clases, n_terminos)
    vocabulario = tfidf.get_feature_names_out()
    return modelo, tfidf, pesos_por_termino, vocabulario


def contribuciones(texto, clase_idx, tfidf, pesos, vocabulario, n=10):
    """Aporte de cada término del texto al margen de un ODS: valor TF-IDF × peso del término."""
    x = tfidf.transform([texto])
    idx_terminos = x.indices
    aporte = x.data * pesos[clase_idx, idx_terminos]
    df = pd.DataFrame({"Raíz": vocabulario[idx_terminos], "Aporte": aporte})
    return df.sort_values("Aporte", ascending=False), df.sort_values("Aporte").head(5)


modelo, tfidf, pesos, vocabulario = cargar_modelo()
clases = modelo.classes_

# ---------- Barra lateral ----------
with st.sidebar:
    st.header("Sobre el modelo")
    st.markdown(
        "- **Pipeline:** preprocesamiento en español → TF-IDF → SVD (1.000 componentes) → LinearSVC\n"
        "- **F1 macro:** 0,86 en validación cruzada y 0,87 en test\n"
        "- **Datos:** 9.655 textos de informes sobre los ODS\n"
        "- **Partición sin fuga:** los textos parafraseados se mantienen juntos en train o en test"
    )
    st.caption("El ODS 17 no aparece en los datos de entrenamiento, así que el modelo no puede predecirlo.")

# ---------- Entrada ----------
st.title("🌍 Clasificador de textos por ODS")
st.write("Escribe o pega un texto y el modelo predice a cuál Objetivo de Desarrollo Sostenible pertenece.")

st.caption("Prueba con un ejemplo escrito fuera del dataset:")
botones = st.columns(len(EJEMPLOS))
for col, (nombre, texto_ejemplo) in zip(botones, EJEMPLOS.items()):
    if col.button(nombre, width="stretch"):
        st.session_state.texto = texto_ejemplo

texto = st.text_area("Texto a clasificar", key="texto", height=160)

if not st.button("Clasificar", type="primary"):
    st.stop()
if not texto.strip():
    st.warning("Escribe un texto primero.")
    st.stop()

# ---------- Predicción ----------
margenes = modelo.decision_function([texto])[0]
orden = np.argsort(margenes)[::-1]
pred, segundo = clases[orden[0]], clases[orden[1]]
diferencia = margenes[orden[0]] - margenes[orden[1]]

st.success(f"**ODS {pred} — {NOMBRES_ODS[pred]}**")
if diferencia < 0.3:
    st.info(f"Predicción con poco margen: el segundo candidato es el ODS {segundo} "
            f"({NOMBRES_ODS[segundo]}), a solo {diferencia:.2f}. Es común cuando el texto toca dos temas.")

izq, der = st.columns(2)

with izq:
    st.subheader("Candidatos")
    top = pd.DataFrame({
        "ODS": [f"{clases[i]} · {NOMBRES_ODS[clases[i]]}" for i in orden[:5]],
        "Margen": margenes[orden[:5]],
        "Elegido": [i == orden[0] for i in orden[:5]],
    })
    st.altair_chart(
        alt.Chart(top).mark_bar().encode(
            x=alt.X("Margen:Q"),
            y=alt.Y("ODS:N", sort=None, title=None),
            color=alt.Color("Elegido:N", legend=None,
                            scale=alt.Scale(domain=[True, False], range=["#2e7d32", "#9e9e9e"])),
            tooltip=["ODS", alt.Tooltip("Margen:Q", format=".3f")],
        ).properties(height=220),
        width="stretch",
    )
    st.caption("El margen de `decision_function` no es una probabilidad: indica de qué lado de la frontera "
               "de cada ODS queda el texto y qué tan separado está el primer candidato del segundo.")

with der:
    st.subheader("Raíces que más pesaron")
    a_favor, en_contra = contribuciones(texto, orden[0], tfidf, pesos, vocabulario)
    a_favor = a_favor[a_favor["Aporte"] > 0].head(10)
    if a_favor.empty:
        st.write("Ninguna palabra del texto empujó con fuerza hacia este ODS.")
    else:
        st.altair_chart(
            alt.Chart(a_favor).mark_bar(color="#2e7d32").encode(
                x=alt.X("Aporte:Q"), y=alt.Y("Raíz:N", sort="-x", title=None),
                tooltip=["Raíz", alt.Tooltip("Aporte:Q", format=".3f")],
            ).properties(height=220),
            width="stretch",
        )
    st.caption("Aporte de cada raíz al margen del ODS elegido (valor TF-IDF × peso del término en el modelo). "
               "Las palabras aparecen truncadas por el stemming.")

with st.expander("Ver el texto como lo procesa el modelo"):
    st.code(preprocess_text(texto), language=None)
    if not en_contra.empty and en_contra["Aporte"].iloc[0] < 0:
        st.write("Raíces que restaron a este ODS:",
                 ", ".join(en_contra[en_contra["Aporte"] < 0]["Raíz"].tolist()))
