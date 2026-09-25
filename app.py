import streamlit as st
import tiktoken
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

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
        # Limpiar tokens para HTML
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
    st.sidebar.warning("⚠️ Debes ingresar tu API Key de Groq para usar la sección generativa.")

st.sidebar.markdown("---")
st.sidebar.info("Esta app utiliza Mixtral/Gemma en Groq y Tiktoken (GPT) para tokenización sub-palabra.")

# ==========================================
# TABS PRINCIPALES
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🧩 Tokenización", 
    "🛍️ Bag of Words", 
    "📐 Similitud Coseno", 
    "🧠 Generación (Groq)"
])

# ------------------------------------------
# TAB 1: TOKENIZACIÓN
# ------------------------------------------
with tab1:
    st.header("Esquemas de Tokenización")
    
    text_input = st.text_area("Ingresa el texto a tokenizar:", "El procesamiento de lenguaje natural es fascinante.")
    scheme = st.radio("Selecciona el esquema:", ["Palabras (Word-level)", "Caracteres (Char-level)", "Sub-palabras (BPE - Estilo GPT)"])
    
    if text_input:
        tokens = []
        token_ids = []
        
        if scheme == "Palabras (Word-level)":
            tokens = text_input.split()
            # Diccionario improvisado para IDs
            vocab = {word: i for i, word in enumerate(set(tokens))}
            token_ids = [vocab[w] for w in tokens]
            
        elif scheme == "Caracteres (Char-level)":
            tokens = list(text_input)
            token_ids = [ord(c) for c in tokens] # Usamos código ASCII/Unicode como ID
            
        elif scheme == "Sub-palabras (BPE - Estilo GPT)":
            # Usamos tiktoken (codificación cl100k_base usada por GPT-3.5/GPT-4)
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
    st.write("Representación matricial de la frecuencia de las palabras.")
    
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
        # Entrenamos el vectorizador con ambas frases
        X = vectorizer.fit_transform([frase1, frase2])
        
        # Calculamos similitud
        similitud = cosine_similarity(X[0], X[1])[0][0]
        
        st.metric(label="Puntuación de Similitud (0 a 1)", value=round(similitud, 4))
        st.progress(float(similitud))
        
        if similitud > 0.8:
            st.success("¡Las frases son muy similares!")
        elif similitud > 0.3:
            st.info("Tienen cierta similitud en su vocabulario.")
        else:
            st.warning("Son frases bastante diferentes según su vocabulario exacto.")

# ------------------------------------------
# TAB 4: ESQUEMA GENERATIVO (GROQ)
# ------------------------------------------
with tab4:
    st.header("Generación de Respuestas con Groq")
    
    if not groq_api_key:
        st.error("Por favor, ingresa tu API Key en la barra lateral para continuar.")
    else:
        # Modelos disponibles en Groq (Excluyendo Llama explícitamente)
        groq_models = [
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "gemma-7b-it"
        ]
        
        col_params1, col_params2 = st.columns(2)
        
        with col_params1:
            selected_model = st.selectbox("Catálogo de Modelos (Sin Llama):", groq_models)
            temperature = st.slider("Temperatura (Creatividad):", 0.0, 2.0, 0.7, 0.1)
            
        with col_params2:
            max_tokens = st.slider("Tokens Máximos:", 50, 4000, 1024, 50)
            top_p = st.slider("Top P (Diversidad de núcleo):", 0.0, 1.0, 1.0, 0.05)
            
        user_prompt = st.text_area("Escribe tu prompt:", "¿Cuáles son las ventajas de usar arquitecturas Mixtured of Experts (MoE)?")
        
        if st.button("🚀 Generar Respuesta", type="primary"):
            try:
                client = Groq(api_key=groq_api_key)
                
                with st.spinner(f"Generando respuesta usando {selected_model}..."):
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": user_prompt,
                            }
                        ],
                        model=selected_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p
                    )
                    
                    st.success("¡Completado!")
                    st.markdown("### Respuesta:")
                    st.write(chat_completion.choices[0].message.content)
                    
                    # Detalles de uso
                    with st.expander("Ver detalles de inferencia"):
                        st.json(chat_completion.usage.model_dump())
                        
            except Exception as e:
                st.error(f"Error al conectar con Groq: {e}")
