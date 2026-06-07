import streamlit as st
import PyPDF2
import spacy
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from collections import Counter
import re
import subprocess
import sys
import os
import gc
import pandas as pd

# --- VISUAL DESIGN SETTINGS ---
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Scientific Lit-Dashboard", layout="wide")
st.title("🔬 Scientific Literary Analysis Dashboard")
st.markdown("Onderzoekscyclus en Stijlanalyse voor de 3de graad Moderne Talen.")

# --- CACHING MODELS & DATA ---
@st.cache_resource(show_spinner="Laden van AI-modellen (eenmalig)...")
def load_models():
    nltk.download('punkt', quiet=True)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        target_dir = "/tmp/spacy_models"
        os.makedirs(target_dir, exist_ok=True)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl", 
            "--target", target_dir, "--no-deps", "--quiet"
        ])
        if target_dir not in sys.path: 
            sys.path.insert(0, target_dir)
        nlp = spacy.load("en_core_web_sm")
    
    if "sentencizer" not in nlp.pipe_names: 
        nlp.add_pipe("sentencizer")
        
    nlp.max_length = 2000000 
    sia = SentimentIntensityAnalyzer()
    return nlp, sia

nlp, sia = load_models()

@st.cache_data
def extract_all_pages(file_buffer):
    reader = PyPDF2.PdfReader(file_buffer)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return pages

def get_chunks(text, chunk_size=50000):
    words = text.split()
    current_chunk = []
    current_length = 0
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        if current_length >= chunk_size:
            yield " ".join(current_chunk)
            current_chunk = []
            current_length = 0
    if current_chunk:
        yield " ".join(current_chunk)

def count_syllables(word):
    """Simpele heuristiek om lettergrepen te tellen voor de Flesch-Kincaid score."""
    word = word.lower()
    if len(word) <= 3:
        return 1
    word = re.sub(r'(?:[^laeiouy]es|ed|[^laeiouy]e)$', '', word)
    word = re.sub(r'^y', '', word)
    matches = re.findall(r'[aeiouy]{1,2}', word)
    return max(1, len(matches))

# --- SIDEBAR UPLOAD & SETTINGS ---
st.sidebar.header("1. Upload Bronmateriaal")
uploaded_file = st.sidebar.file_uploader("Upload Engelstalige roman (PDF)", type="pdf")

if uploaded_file is not None:
    with st.spinner("PDF structuur analyseren..."):
        all_pages = extract_all_pages(uploaded_file)
        total_pages = len(all_pages)

    st.sidebar.markdown("---")
    st.sidebar.header("2. Afbakening Onderzoek")
    st.sidebar.info("Beperk de data om 'overfitting' van je analyse te voorkomen en verwerkingstijd te sparen.")
    
    max_limit = min(total_pages, 200)
    selected_page_count = st.sidebar.slider("Aantal te analyseren pagina's", min_value=10, max_value=max_limit, value=max_limit)
    read_direction = st.sidebar.radio("Selecteer brondeel:", ["Begin van het boek", "Einde van het boek"])

    if read_direction == "Begin van het boek":
        target_pages = all_pages[:selected_page_count]
    else:
        target_pages = all_pages[-selected_page_count:]

    text_data = re.sub(r'\s+', ' ', " ".join(target_pages))
    chunks = list(get_chunks(text_data))
    total_chunks = len(chunks)

    # --- TABS VOOR DE ONDERZOEKSCYCLUS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 1. Didactisch Kader", 
        "🧠 2. Thematische Lemmatisering", 
        "⚙️ 3. Linguïstische Stijl & Register", 
        "🎭 4. Figuratieve Taalreflectie"
    ])

    # ==========================================
    # TAB 1: DIDACTISCH KADER (LEERPLANDOELEN)
    # ==========================================
    with tab1:
        st.header("Onderzoeksdoelen & Verantwoording")
        st.markdown("""
        Dit dashboard is ontworpen ter ondersteuning van de **Onderzoekscyclus Literatuur** in de 3de graad Moderne Talen. 
        De output van de verschillende modules helpt je bij het behalen van de volgende specifieke leerplandoelen (Engels):
        """)
        
        st.info("**WD3_01.01.01: Doorlopen van de onderzoekscyclus**\n\nJe gebruikt dit dashboard om kritisch data (literaire bronnen) te verzamelen, te ordenen via parameters (pagina-selecties), en de verwerkte visuele data te interpreteren om een onderzoeksvraag te beantwoorden.")
        st.success("**WD3_02.20.01: Analyseren van poëticale en narratieve structuren**\n\nMet behulp van technologische ondersteuning onderzoek je de stilistische keuzes van een auteur uit de wereldliteratuur (zoals woordkeuze, syntax en actiegerichtheid).")
        st.warning("**WD3_02.07.04: Verklaren van automatische analyses**\n\nIn de tab *Linguïstische Stijl* en *Lemmatisering* maak je gebruik van Natural Language Processing (NLP). Je leert begrijpen hoe de computer tekst opbreekt (parsing), woorden terugbrengt naar hun stamvorm (lemmatisering) en woordsoorten toekent (POS-tagging).")
        st.error("**WD3_02.07.03: Beperkingen van AI evalueren**\n\nIn de tab *Figuratieve Taalreflectie* controleer je handmatig de door de AI berekende sentiment-scores. Je leert hoe AI vaak 'struikelt' over ironie, sarcasme, of cultureel impliciet taalgebruik.")

    # ==========================================
    # TAB 2: THEMATISCHE LEMMATISERING
    # ==========================================
    with tab2:
        st.header("Thematische Lemmatisering")
        st.markdown("**Doel:** Achterhaal de kernthema's door alle inhoudswoorden (zelfstandige naamwoorden) terug te brengen naar hun woordenboekvorm (lemma).")

        if st.button("Genereer Thematische Lemma's"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            lemma_counts = Counter()
            
            for i, doc in enumerate(nlp.pipe(chunks, disable=["ner", "textcat", "custom"])):
                for token in doc:
                    # Filter: moet een noun zijn, geen stopwoord, geen interpunctie, alleen alfabetisch
                    if token.pos_ == "NOUN" and not token.is_stop and token.is_alpha and len(token.text) > 2:
                        lemma_counts[token.lemma_.lower()] += 1
                        
                progress_bar.progress((i + 1) / total_chunks)
                status_text.text(f"Lemmatiseren: deel {i + 1} van {total_chunks}...")
                gc.collect()

            status_text.text("Genereren van Top 20...")
            
            if lemma_counts:
                top_lemmas = lemma_counts.most_common(20)
                lemmas, counts = zip(*top_lemmas)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(x=list(counts), y=list(lemmas), palette="viridis", ax=ax)
                ax.set_title("Top 20 Meest Voorkomende Zelfstandige Naamwoorden (Lemma's)", fontsize=14, fontweight='bold')
                ax.set_xlabel("Frequentie")
                ax.set_ylabel("Lemma (Woordenboekvorm)")
                sns.despine()
                st.pyplot(fig)
                plt.close(fig)
                
                st.markdown("### 📝 Reflectievraag voor je onderzoek:")
                st.write("> *Zijn er lemma's in de bovenstaande lijst die wijzen op een overkoepelend motief of thema in het gekozen romanfragment? Hoe sturen deze woorden de perceptie van de lezer?*")
            
            progress_bar.empty()
            status_text.empty()

    # ==========================================
    # TAB 3: LINGUÏSTISCHE STIJL & REGISTER
    # ==========================================
    with tab3:
        st.header("Linguïstische Stijlanalyse (POS-tagging & Syntaxis)")
        st.markdown("**Doel:** Bereken het taalkundig register en de schrijfstijl van de auteur met behulp van AI Part-of-Speech (POS) tagging en complexiteitsalgoritmes.")

        if st.button("Voer Stijl- en Registeranalyse uit"):
            progress_bar = st.progress(0)
            
            total_sentences = 0
            total_words = 0
            total_syllables = 0
            
            adj_count = 0
            verb_count = 0
            
            for i, doc in enumerate(nlp.pipe(chunks, disable=["ner", "textcat", "custom"])):
                for sent in doc.sents:
                    total_sentences += 1
                    for token in sent:
                        if token.is_alpha:
                            total_words += 1
                            total_syllables += count_syllables(token.text)
                            
                            if token.pos_ == "ADJ":
                                adj_count += 1
                            elif token.pos_ == "VERB":
                                verb_count += 1
                progress_bar.progress((i + 1) / total_chunks)
                gc.collect()

            # 1. Syntactische Complexiteit & Leesbaarheid (Flesch-Kincaid)
            if total_sentences > 0 and total_words > 0:
                avg_sentence_length = total_words / total_sentences
                
                # Flesch Reading Ease Formula
                flesch_score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
                
                if flesch_score > 80: register = "Zeer Informeel / Kinderlijk (Conversational)"
                elif flesch_score > 60: register = "Standaard / Toegankelijk (Average Reader)"
                elif flesch_score > 30: register = "Complex / Formeel (College Level)"
                else: register = "Zeer Academisch / Archaïsch (Difficult)"

                col1, col2 = st.columns(2)
                col1.metric("Gemiddelde Zinslengte (Words/Sent)", f"{avg_sentence_length:.1f}")
                col2.metric("Flesch Readability Score", f"{flesch_score:.1f}")
                
                st.info(f"**Register Analyse:** Op basis van de syntactische complexiteit wordt het register van dit tekstdeel beoordeeld als: **{register}**.")

            # 2. POS Gauge: Action vs Descriptive
            st.markdown("---")
            if (adj_count + verb_count) > 0:
                verb_ratio = verb_count / (adj_count + verb_count)
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = verb_ratio * 100,
                    title = {'text': "Auteur Stijl: Descriptief vs. Actiegericht", 'font': {'size': 18}},
                    number = {'suffix': "% Werkwoorden"},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1},
                        'bar': {'color': "#1e3d59"},
                        'steps': [
                            {'range': [0, 40], 'color': "#a8dadc"},     # Descriptive
                            {'range': [40, 60], 'color': "#f1faee"},    # Balanced
                            {'range': [60, 100], 'color': "#e63946"}    # Action
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': verb_ratio * 100
                        }
                    }
                ))
                
                fig.add_annotation(x=0.1, y=0, text="Descriptive (Poetic)", showarrow=False, font=dict(size=14, color="#1d3557"))
                fig.add_annotation(x=0.9, y=0, text="Action-oriented (Dynamic)", showarrow=False, font=dict(size=14, color="#e63946"))
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 📝 Reflectievraag voor je onderzoek:")
                st.write("> *Een score onder de 40% wijst op veel bijvoeglijke naamwoorden (descriptief/poëtisch). Boven de 60% wijst op veel werkwoorden (actie/plot-gedreven). Komt dit overeen met jouw leeservaring van de auteur?*")

            progress_bar.empty()

    # ==========================================
    # TAB 4: FIGURATIEVE TAALREFLECTIE
    # ==========================================
    with tab4:
        st.header("Figuratieve Taal & AI Sentiment Evaluatie")
        st.markdown("**Doel:** Evalueer de beperkingen van kunstmatige intelligentie. AI beoordeelt tekst vaak uiterst letterlijk. Zoek naar ironie, sarcasme, of metaforen in de uitschieters.")

        if st.button("Isoleer Extremen voor Handmatige Controle"):
            with st.spinner("VADER Sentiment Analyzer leest zinnen..."):
                sentences = re.split(r'(?<=[.!?]) +', text_data)
                scored_sentences = []
                
                for s in sentences:
                    s_clean = s.strip().replace('\n', ' ')
                    # We kijken enkel naar volwaardige zinnen (meer dan 8 woorden)
                    if len(s_clean.split()) > 8:
                        score = sia.polarity_scores(s_clean)['compound']
                        scored_sentences.append({'text': s_clean, 'score': score})
                
                # Sorteer op score
                scored_sentences.sort(key=lambda x: x['score'])
                
                top_5_negative = scored_sentences[:5]
                top_5_positive = scored_sentences[-5:]
                top_5_positive.reverse() # Hoogste positieve bovenaan

                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.error("### 🔴 AI's Top 5 Meest Negatieve Fragmenten")
                    for i, item in enumerate(top_5_negative):
                        st.markdown(f"**{i+1}. Score: {item['score']:.2f}**\n\n> *\"{item['text']}\"*")
                
                with col2:
                    st.success("### 🟢 AI's Top 5 Meest Positieve Fragmenten")
                    for i, item in enumerate(top_5_positive):
                        st.markdown(f"**{i+1}. Score: {item['score']:.2f}**\n\n> *\"{item['text']}\"*")

                st.markdown("---")
                st.markdown("### 📝 Reflectie-opdrachten (Handmatige Controle):")
                st.markdown("""
                1. **Valse Positieven:** Is een van de 'Positieve' fragmenten eigenlijk cynisch, sarcastisch of duister als je de bredere context van de roman kent? Leg uit waarom de AI zich vergiste.
                2. **Metaforiek:** Bevatten de fragmenten metaforen (bijv. "Een storm in zijn hart") die door de AI letterlijk (als een gevaarlijke weersomstandigheid) werden vertaald naar een negatieve score?
                3. **Culturele Implicaties:** Mist het algoritme bepaalde culturele of historische nuances?
                """)

else:
    st.info("Upload een PDF in het zijpaneel om het onderzoek te starten.")
