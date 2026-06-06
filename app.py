import streamlit as st
import PyPDF2
import spacy
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import itertools
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
st.set_page_config(page_title="Literary Dashboard", layout="wide")
st.title("📚 The Ultimate Literary Analysis Dashboard")
st.markdown("Upload a novel (PDF), select a range of pages, and let AI uncover its hidden structures.")

# --- CACHING MODELS & DATA ---
@st.cache_resource(show_spinner="Configuring AI... (first time only, please wait)")
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

# --- ENTITY CLEANER ---
def clean_entity_name(text):
    """Wasstraat voor personagenamen om ruis uit de AI te filteren."""
    # 1. Hak af vanaf de bezits-s ("Gatsby's foot" -> "Gatsby")
    text = re.split(r"['’]s\b", text)[0]
    
    # 2. Verwijder veelvoorkomende titels (Aunt, Mr, Dr, etc.)
    titles = [r'\bMr\.?\s', r'\bMrs\.?\s', r'\bMs\.?\s', r'\bMiss\s', 
              r'\bDr\.?\s', r'\bAunt\s', r'\bUncle\s', r'\bProfessor\s', r'\bCaptain\s']
    for t in titles:
        text = re.sub(t, "", text, flags=re.IGNORECASE)
        
    # 3. Houd alleen woorden over die met een hoofdletter beginnen ("suppose Tom" -> "Tom")
    words = [w for w in text.split() if w.istitle()]
    
    return " ".join(words).strip()

# --- SIDEBAR UPLOAD & SETTINGS ---
st.sidebar.header("1. Upload Book")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Reading PDF structure..."):
        all_pages = extract_all_pages(uploaded_file)
        total_pages = len(all_pages)

    st.sidebar.markdown("---")
    st.sidebar.header("2. Analysis Limits")
    
    max_limit = min(total_pages, 200)
    selected_page_count = st.sidebar.slider("Number of pages to analyze", min_value=10, max_value=max_limit, value=max_limit)
    read_direction = st.sidebar.radio("Extract text from:", ["Beginning of book", "End of book"])

    if read_direction == "Beginning of book":
        target_pages = all_pages[:selected_page_count]
    else:
        target_pages = all_pages[-selected_page_count:]

    text_data = re.sub(r'\s+', ' ', " ".join(target_pages))
    chunks = list(get_chunks(text_data))
    total_chunks = len(chunks)
    
    # Reset Social Data if user changes slider
    current_hash = hash(text_data)
    if st.session_state.get('text_hash') != current_hash:
        if 'social_data' in st.session_state:
            del st.session_state['social_data']
        st.session_state['text_hash'] = current_hash

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Style Scanner", 
        "2. Emotion Arc", 
        "3. Social Network", 
        "4. Gender Bias", 
        "5. Color Palette"
    ])

    # ==========================================
    # TAB 1: STYLE SCANNER
    # ==========================================
    with tab1:
        st.header("Comparative Style Scanner")
        if st.button("Run Style Analysis"):
            progress_bar = st.progress(0)
            sent_lengths = []
            
            for i, doc in enumerate(nlp.pipe(chunks, disable=["ner", "lemmatizer", "textcat", "custom"])):
                for s in doc.sents:
                    length = len([t for t in s if not t.is_punct])
                    if length > 2: sent_lengths.append(length)
                progress_bar.progress((i + 1) / total_chunks)
                gc.collect()

            if sent_lengths:
                avg_len = sum(sent_lengths) / len(sent_lengths)
                window = max(30, len(sent_lengths) // 100) 
                smoothed = [sum(sent_lengths[i:i+window])/window for i in range(len(sent_lengths)-window+1)]
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(smoothed, color='#2b8cbe', linewidth=2)
                ax.fill_between(range(len(smoothed)), smoothed, color='#2b8cbe', alpha=0.2)
                ax.set_title("Sentence Length Over Time", fontsize=14, fontweight='bold')
                sns.despine()
                st.pyplot(fig)
                plt.close(fig)
            progress_bar.empty()

    # ==========================================
    # TAB 2: VONNEGUT ARC
    # ==========================================
    with tab2:
        st.header("The Vonnegut Emotion Arc")
        if st.button("Run Emotion Arc"):
            sentences = re.split(r'(?<=[.!?]) +', text_data)
            scores = [sia.polarity_scores(s)['compound'] for s in sentences if len(s) > 10]
            gc.collect()

            if scores:
                window = max(1, len(scores) // 20)
                smoothed = [sum(scores[i:i+window])/window for i in range(len(scores)-window+1)]
                x_perc = [(i / len(smoothed)) * 100 for i in range(len(smoothed))]
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(x_perc, smoothed, color='#8856a7', linewidth=2.5)
                ax.fill_between(x_perc, smoothed, 0, where=pd.Series(smoothed) > 0, color='green', alpha=0.2)
                ax.fill_between(x_perc, smoothed, 0, where=pd.Series(smoothed) < 0, color='red', alpha=0.2)
                sns.despine()
                st.pyplot(fig)
                plt.close(fig)

    # ==========================================
    # TAB 3: SOCIAL NETWORK & RELATIONSHIPS
    # ==========================================
    with tab3:
        st.header("Social Web & Relationship Dynamics")
        
        # Stap 1: Genereer het netwerk
        if st.button("Run Social Analysis"):
            progress_bar = st.progress(0)
            raw_char_counts = Counter()
            raw_interactions = {}

            for i, doc in enumerate(nlp.pipe(chunks, disable=["tagger", "parser", "lemmatizer", "textcat", "custom"])):
                for sent in doc.sents:
                    raw_chars = [ent.text for ent in sent.ents if ent.label_ == "PERSON" and len(ent.text) > 2]
                    
                    # Nieuwe aggressieve wasstraat voor entiteiten
                    cleaned_chars = set()
                    for char in raw_chars:
                        clean_name = clean_entity_name(char)
                        if len(clean_name) > 2: # Negeer weesletters
                            cleaned_chars.add(clean_name)
                    
                    for char in cleaned_chars:
                        raw_char_counts[char] += 1
                    
                    if len(cleaned_chars) > 1:
                        sent_score = sia.polarity_scores(sent.text)['compound']
                        for c1, c2 in itertools.combinations(cleaned_chars, 2):
                            pair = tuple(sorted([c1, c2]))
                            if pair not in raw_interactions:
                                raw_interactions[pair] = {'weight': 0, 'sentiment': []}
                            raw_interactions[pair]['weight'] += 1
                            raw_interactions[pair]['sentiment'].append(sent_score)
                
                progress_bar.progress((i + 1) / total_chunks)
                gc.collect()

            # Deduplicatie (Fusie van namen)
            sorted_names = sorted(raw_char_counts.keys(), key=len, reverse=True)
            name_mapping = {}
            for name in sorted_names:
                mapped = False
                for longer_name in list(set(name_mapping.values())):
                    if re.search(r'\b' + re.escape(name) + r'\b', longer_name):
                        name_mapping[name] = longer_name
                        mapped = True
                        break
                if not mapped:
                    name_mapping[name] = name

            char_counts = Counter()
            for name, count in raw_char_counts.items():
                char_counts[name_mapping.get(name, name)] += count

            interactions = {}
            for (c1, c2), data in raw_interactions.items():
                m1 = name_mapping.get(c1, c1)
                m2 = name_mapping.get(c2, c2)
                if m1 != m2: 
                    pair = tuple(sorted([m1, m2]))
                    if pair not in interactions:
                        interactions[pair] = {'weight': 0, 'sentiment': []}
                    interactions[pair]['weight'] += data['weight']
                    interactions[pair]['sentiment'].extend(data['sentiment'])

            if char_counts:
                top_chars = [c for c, count in char_counts.most_common(15)]
                # Sla op in sessie zodat we er later drop-downs mee kunnen vullen
                st.session_state['social_data'] = {
                    'top_chars': top_chars,
                    'char_counts': char_counts,
                    'interactions': interactions
                }
            progress_bar.empty()

        # Stap 2: Laat het netwerk en de tracker zien als de data beschikbaar is
        if 'social_data' in st.session_state:
            top_chars = st.session_state['social_data']['top_chars']
            char_counts = st.session_state['social_data']['char_counts']
            interactions = st.session_state['social_data']['interactions']

            G = nx.Graph()
            for char in top_chars:
                G.add_node(char, weight=char_counts[char])
            for (c1, c2), data in interactions.items():
                if c1 in top_chars and c2 in top_chars:
                    avg_sent = sum(data['sentiment']) / len(data['sentiment']) if data['sentiment'] else 0
                    if avg_sent > 0.15: color = '#2ec4b6'
                    elif avg_sent < -0.15: color = '#e71d36'
                    else: color = '#cccccc'
                    G.add_edge(c1, c2, weight=data['weight'], color=color)

            if len(G.nodes) > 0:
                fig, ax = plt.subplots(figsize=(14, 10), facecolor='#fafafa')
                ax.set_facecolor('#fafafa')
                pos = nx.spring_layout(G, k=0.8, iterations=60, seed=42)
                
                node_weights = [nx.get_node_attributes(G, 'weight')[n] for n in G.nodes()]
                max_node_w = max(node_weights) if node_weights else 1
                node_sizes = [(w / max_node_w) * 2200 + 400 for w in node_weights]
                
                edge_colors = [nx.get_edge_attributes(G, 'color')[e] for e in G.edges()]
                weights = [nx.get_edge_attributes(G, 'weight')[e] for e in G.edges()]
                max_edge_w = max(weights) if weights else 1
                scaled_edge_widths = [(w / max_edge_w) * 6 + 1.5 for w in weights]
                
                nx.draw_networkx_edges(G, pos, width=scaled_edge_widths, edge_color=edge_colors, alpha=0.4, connectionstyle="arc3,rad=0.2", ax=ax)
                nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_weights, cmap=plt.cm.Blues, edgecolors='#1e3d59', linewidths=1.8, ax=ax)
                nx.draw_networkx_labels(G, pos, font_size=11, font_family='sans-serif', font_weight='bold', font_color='#1e3d59',
                                        bbox=dict(facecolor='#ffffff', edgecolor='#1e3d59', alpha=0.9, boxstyle='round,pad=0.3', linewidth=1), ax=ax)
                
                ax.set_title("✨ Dynamic Character Interaction Network", fontsize=18, fontweight='bold', color='#1e3d59', pad=20)
                plt.axis('off')
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("---")
            st.subheader("📈 Relationship Evolution Tracker")
            st.info("Select two characters from the network above to track their emotional dynamic over time.")
            
            # Automatische Dropdowns gebaseerd op de netwerkanalyse
            col1, col2 = st.columns(2)
            char1 = col1.selectbox("First Character", top_chars, index=0)
            char2 = col2.selectbox("Second Character", top_chars, index=1 if len(top_chars) > 1 else 0)

            if st.button("Trace Relationship"):
                with st.spinner(f"Tracing {char1} & {char2}..."):
                    sentences = re.split(r'(?<=[.!?]) +', text_data)
                    progression, sentiments = [], []

                    for i, s in enumerate(sentences):
                        # Simpele check: de eerste naam (bijv 'Tom') van de volle naam moet in de zin staan
                        c1_base = char1.split()[0]
                        c2_base = char2.split()[0]
                        
                        if re.search(rf'\b{re.escape(c1_base)}\b', s, re.IGNORECASE) and re.search(rf'\b{re.escape(c2_base)}\b', s, re.IGNORECASE):
                            score = sia.polarity_scores(s)['compound']
                            sentiments.append(score)
                            progression.append((i / len(sentences)) * 100)
                    gc.collect()

                    if len(sentiments) < 3:
                        st.warning(f"Not enough interactions between **{char1}** and **{char2}** in this selection to draw a trendline.")
                    else:
                        df = pd.DataFrame({'Progress': progression, 'Sentiment': sentiments})
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.scatterplot(data=df, x='Progress', y='Sentiment', color='#1f77b4', s=60, alpha=0.6, ax=ax)
                        sns.regplot(data=df, x='Progress', y='Sentiment', scatter=False, order=3, color='#e41a1c')
                        ax.axhline(0, color='black', linewidth=1, linestyle='--')
                        ax.set_title(f"Relationship Evolution: {char1} & {char2}", fontsize=14, fontweight='bold')
                        sns.despine()
                        st.pyplot(fig)
                        plt.close(fig)

    # ==========================================
    # TAB 4 & 5 (Verkorte versies voor code overzicht)
    # ==========================================
    with tab4:
        st.header("Gender-Bias Agency Scanner")
        st.info("Run this scanner to compare verb usage.")
        # Code voor Tab 4 blijft identiek

    with tab5:
        st.header("The Aesthetic Color Palette")
        st.info("Extract color mentions.")
        # Code voor Tab 5 blijft identiek

else:
    st.info("Please upload a PDF file in the sidebar to begin.")
