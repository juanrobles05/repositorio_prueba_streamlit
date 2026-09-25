import streamlit as st
import tiktoken
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import pytesseract
from PIL import Image

# Configuración de página
st.set_page_config(page_title="NLP & Groq Explorer", layout="wide")

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def get_colored_tokens_html(tokens):
    """Genera HTML para mostrar tokens con colores de fondo."""
    colors = ["#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", "#E6B3FF", "#FFB3E6", "#E2F0CB", "#FFB7B2"]
    html = '<div style="line-height: 2.5;">'
    for i, token in enumerate(tokens):
        color = colors[i % len(colors)]
        safe_token = str(token).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html += f'<span style="background-color: {color}; padding: 4px 6px; margin: 2px; border-radius: 4px; color: black; font-family: monospace; border: 1px solid #ccc;">{safe_token}</span>'
    html += '</div>'
    return html

# ==========================================
# BARRA LATERAL: API KEY
# ==========================================
st.sidebar.title("🔑 Configuración")
groq_api_key = st.sidebar.text_input("Ingresa tu Groq API Key", type="password")

if not groq_api_key:
    st.sidebar.warning("⚠️ Debes ingresar tu API Key de Groq para usar las secciones generativas.")

# ==========================================
# TABS PRINCIPALES
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧩 Tokenización", 
    "🛍️ Bag of Words", 
    "📐 Similitud Coseno", 
    "🧠 Generación (Groq)",
    "📸 OCR & Expansión"
])

# ------------------------------------------
# TAB 1: TOKENIZACIÓN
# ------------------------------------------
with tab1:
    st.header("Esquemas de Tokenización")
    text_input = st.text_area("Ingresa el texto a tokenizar:", "El procesamiento de lenguaje natural es fascinante.")
    scheme = st.radio("Selecciona el esquema:", ["Palabras (Word-level)", "Caracteres (Char-level)", "Sub-palabras (BPE - Estilo GPT)"])
    
    if text_input:
        tokens, token_ids = [], []
        if scheme == "Palabras (Word-level)":
            tokens = text_input.split()
            vocab = {word: i for i, word in enumerate(set(tokens))}
            token_ids = [vocab[w] for w in tokens]
        elif scheme == "Caracteres (Char-level)":
            tokens = list(text_input)
            token_ids = [ord(c) for c in tokens] 
        elif scheme == "Sub-palabras (BPE - Estilo GPT)":
            enc = tiktoken.get_encoding("cl100k_base")
            token_ids = enc.encode(text_input)
            tokens = [enc.decode([t]) for t in token_ids]

        st.subheader("Tokens Obtenidos (Coloreados)")
        st.markdown(get_colored_tokens_html(tokens), unsafe_allow_html=True)
        st.subheader("IDs de los Tokens")
        st.write(token_ids)

# ------------------------------------------
# TAB 2: BAG OF WORDS
# ------------------------------------------
with tab2:
    st.header("Bag of Words (BoW)")
    bow_text = st.text_area("Ingresa varias frases (separadas por salto de línea):", 
                            "El gato come pescado\nEl perro come carne\nEl gato y el perro")
    if bow_text:
        corpus = [frase.strip() for frase in bow_text.split('\n') if frase.strip()]
        if corpus:
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(corpus)
            df_bow = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out(), index=[f"Frase {i+1}" for i in range(len(corpus))])
            st.dataframe(df_bow, use_container_width=True)
        else:
            st.warning("Ingresa al menos una frase válida.")

# ------------------------------------------
# TAB 3: SIMILITUD DEL COSENO
# ------------------------------------------
with tab3:
    st.header("Similitud con Distancia de Coseno")
    col1, col2 = st.columns(2)
    with col1:
        frase1 = st.text_input("Frase 1:", "La inteligencia artificial está revolucionando el mundo")
    with col2:
        frase2 = st.text_input("Frase 2:", "El aprendizaje automático transforma nuestra sociedad actual")
        
    if frase1 and frase2:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([frase1, frase2])
        similitud = cosine_similarity(X[0], X[1])[0][0]
        
        st.metric(label="Puntuación de Similitud (0 a 1)", value=round(similitud, 4))
        st.progress(float(similitud))

# ------------------------------------------
# TAB 4: ESQUEMA GENERATIVO (GROQ)
# ------------------------------------------
with tab4:
    st.header("Generación de Respuestas con Groq")
    if groq_api_key:
        groq_models = ["mixtral-8x7b-32768", "gemma2-9b-it", "gemma-7b-it"]
        col_params1, col_params2 = st.columns(2)
        
        with col_params1:
            selected_model = st.selectbox("Catálogo de Modelos (Sin Llama):", groq_models, key="t4_model")
            temperature = st.slider("Temperatura:", 0.0, 2.0, 0.7, 0.1, key="t4_temp")
            
        with col_params2:
            max_tokens = st.slider("Tokens Máximos:", 50, 4000, 1024, 50, key="t4_tokens")
            top_p = st.slider("Top P:", 0.0, 1.0, 1.0, 0.05, key="t4_top_p")
            
        user_prompt = st.text_area("Escribe tu prompt:", "¿Cuáles son las ventajas de usar arquitecturas Mixtured of Experts (MoE)?")
        
        if st.button("🚀 Generar Respuesta", type="primary", key="btn_gen"):
            try:
                client = Groq(api_key=groq_api_key)
                with st.spinner("Generando..."):
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": user_prompt}],
                        model=selected_model, temperature=temperature, max_tokens=max_tokens, top_p=top_p
                    )
                    st.markdown("### Respuesta:")
                    st.write(chat_completion.choices[0].message.content)
            except Exception as e:
                st.error(f"Error: {e}")

# ------------------------------------------
# TAB 5: OCR Y EXPANSIÓN
# ------------------------------------------
with tab5:
    st.header("Lector de Imágenes y Expansión Asistida")
    st.write("Sube una imagen con texto. Extraeremos el contenido y usaremos la IA para explicarlo o ampliarlo.")
    
    uploaded_image = st.file_uploader("Sube una imagen (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_image is not None:
        # Mostrar la imagen cargada
        img = Image.open(uploaded_image)
        st.image(img, caption="Imagen original", width=400)
        
        # Extraer texto con Tesseract
        with st.spinner("Procesando imagen (OCR)..."):
            try:
                extracted_text = pytesseract.image_to_string(img)
            except Exception as e:
                st.error(f"Error en OCR: {e}. Asegúrate de tener Tesseract instalado en tu sistema.")
                extracted_text = ""
        
        if extracted_text.strip():
            st.success("Texto extraído con éxito.")
            # Permitir que el usuario edite el texto por si el OCR tuvo algún fallo
            edited_text = st.text_area("Texto extraído (puedes corregirlo):", extracted_text, height=150)
            
            st.markdown("---")
            st.subheader("Configuración de Expansión")
            
            if not groq_api_key:
                st.warning("Ingresa tu API Key en el menú lateral para usar esta función.")
            else:
                col_ocr1, col_ocr2 = st.columns(2)
                with col_ocr1:
                    ocr_model = st.selectbox("Modelo:", ["mixtral-8x7b-32768", "gemma2-9b-it"], key="ocr_model")
                with col_ocr2:
                    instruction = st.selectbox(
                        "¿Qué quieres hacer con este texto?",
                        [
                            "Ampliar y explicar en detalle los conceptos mencionados.",
                            "Resumir las ideas principales.",
                            "Corregir redacción y ortografía.",
                            "Traducir al inglés."
                        ]
                    )
                
                if st.button("✨ Procesar texto extraído", type="primary"):
                    final_prompt = f"Instrucción: {instruction}\n\nTexto original extraído:\n{edited_text}"
                    
                    try:
                        client = Groq(api_key=groq_api_key)
                        with st.spinner("Analizando y generando respuesta..."):
                            completion = client.chat.completions.create(
                                messages=[{"role": "user", "content": final_prompt}],
                                model=ocr_model,
                                temperature=0.7,
                                max_tokens=2048
                            )
                            st.markdown("### Resultado de la IA:")
                            st.info(completion.choices[0].message.content)
                    except Exception as e:
                        st.error(f"Error al conectar con Groq: {e}")
        else:
            st.warning("No se detectó texto en la imagen. Intenta con otra o recorta la imagen para enfocar las letras.")
