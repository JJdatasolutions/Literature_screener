import streamlit as st
import PyPDF2
import spacy
from spacy.cli import download
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import itertools
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Literary Dashboard", layout="wide")
st.title("📚 The Ultimate Literary Analysis Dashboard")
st.markdown("Upload a novel (PDF) and let AI uncover its hidden structures.")

# --- CACHING MODELS & DATA ---
@st.cache_resource(show_spinner="Loading AI models (first time takes a minute)...")
def load_models():
    # 1. Download NLTK data if needed 
    nltk.download('punkt', quiet=True)
    
    # 2. Slimme Spacy lader
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Als hij crasht omdat het model ontbreekt, downloaden we het via de code zelf!
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        
    nlp.max_length = 2000000
    
    # 3. Load VADER for sentiment analysis
    sia = SentimentIntensityAnalyzer()
    
    return nlp, sia

# Modellen inladen
nlp, sia = load_models()

@st.cache_data
def extract_text(file_buffer, max_pages=50):
    reader = PyPDF2.PdfReader(file_buffer)
    full_text = ""
    for page in reader.pages[:max_pages]:
        text = page.extract_text()
        if text:
            full_text += text + " "
    return re.sub(r'\s+', ' ', full_text)

# --- SIDEBAR UPLOAD ---
st.sidebar.header("1. Upload Book")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
max_p = st.sidebar.slider("Pages to analyze", 10, 150, 50, help="More pages take longer to process.")

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        text_data = extract_text(uploaded_file, max_pages=max_p)
    
    # Create Tabs for the 6 different tools
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Style Scanner", 
        "2. Emotion Arc", 
        "3. Social Network", 
        "4. Gender Bias", 
        "5. Color Palette", 
        "6. Motif Barcode"
    ])

    # ==========================================
    # TAB 1: STYLE SCANNER
    # ==========================================
    with tab1:
        st.header("Comparative Style Scanner")
        if st.button("Run Style Analysis"):
            with st.spinner("Analyzing syntax..."):
                doc = nlp(text_data[:500000]) # Limit for speed
                sentences = list(doc.sents)
                sent_lengths = [len([t for t in s if not t.is_punct]) for s in sentences if len(s) > 2]
                avg_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0
                
                st.metric("Average Sentence Length", f"{avg_len:.1f} words")
                
                # Plot moving average of sentence length
                window = 30
                smoothed = [sum(sent_lengths[i:i+window])/window for i in range(len(sent_lengths)-window+1)]
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(smoothed, color='blue', linewidth=2)
                ax.set_title("Sentence Length Over Time (30-sentence moving average)")
                ax.set_ylabel("Words per Sentence")
                st.pyplot(fig)

    # ==========================================
    # TAB 2: VONNEGUT ARC
    # ==========================================
    with tab2:
        st.header("The Vonnegut Emotion Arc")
        if st.button("Run Emotion Arc"):
            with st.spinner("Calculating sentiment..."):
                sentences = re.split(r'(?<=[.!?]) +', text_data)
                scores = [sia.polarity_scores(s)['compound'] for s in sentences if len(s) > 10]
                
                window = max(1, len(scores) // 20)
                smoothed = [sum(scores[i:i+window])/window for i in range(len(scores)-window+1)]
                x_perc = [(i / len(smoothed)) * 100 for i in range(len(smoothed))]
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(x_perc, smoothed, color='purple', linewidth=2)
                ax.axhline(0, color='black', linestyle='--')
                ax.set_title("Narrative Sentiment Arc")
                ax.set_xlabel("Story Progress (%)")
                ax.set_ylabel("Sentiment (Negative -> Positive)")
                st.pyplot(fig)

    # ==========================================
    # TAB 3: SOCIAL NETWORK (WITH SENTIMENT)
    # ==========================================
    with tab3:
        st.header("Social Web & Relationship Dynamics")
        st.write("Edges are colored by sentiment: Green = Positive interaction, Red = Negative interaction.")
        if st.button("Run Network Analysis"):
            with st.spinner("Extracting characters and computing relationship sentiment..."):
                doc = nlp(text_data[:500000])
                char_counts = Counter()
                interactions = {}

                for sent in doc.sents:
                    chars = set([ent.text.replace("'s", "").strip() for ent in sent.ents if ent.label_ == "PERSON" and len(ent.text) > 3])
                    for char in chars:
                        char_counts[char] += 1
                    
                    if len(chars) > 1:
                        sent_score = sia.polarity_scores(sent.text)['compound']
                        for c1, c2 in itertools.combinations(chars, 2):
                            pair = tuple(sorted([c1, c2]))
                            if pair not in interactions:
                                interactions[pair] = {'weight': 0, 'sentiment': []}
                            interactions[pair]['weight'] += 1
                            interactions[pair]['sentiment'].append(sent_score)

                top_chars = [c for c, count in char_counts.most_common(12)]
                G = nx.Graph()
                
                for char in top_chars:
                    G.add_node(char, size=char_counts[char]*50)
                    
                for (c1, c2), data in interactions.items():
                    if c1 in top_chars and c2 in top_chars:
                        avg_sent = sum(data['sentiment']) / len(data['sentiment'])
                        # Color logic based on sentiment
                        if avg_sent > 0.05: color = 'green'
                        elif avg_sent < -0.05: color = 'red'
                        else: color = 'gray'
                        
                        G.add_edge(c1, c2, weight=data['weight'], color=color)

                fig, ax = plt.subplots(figsize=(10, 8))
                pos = nx.spring_layout(G, k=0.8)
                sizes = [nx.get_node_attributes(G, 'size')[n] for n in G.nodes()]
                edge_colors = [nx.get_edge_attributes(G, 'color')[e] for e in G.edges()]
                weights = [nx.get_edge_attributes(G, 'weight')[e]*0.5 for e in G.edges()]
                
                nx.draw_networkx_edges(G, pos, width=weights, edge_color=edge_colors, alpha=0.6)
                nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='skyblue', edgecolors='black')
                nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
                plt.axis('off')
                st.pyplot(fig)

    # ==========================================
    # TAB 4: GENDER BIAS
    # ==========================================
    with tab4:
        st.header("Gender-Bias Agency Scanner")
        if st.button("Run Gender Analysis"):
            with st.spinner("Analyzing pronouns and verbs..."):
                doc = nlp(text_data[:500000])
                m_verbs, f_verbs = Counter(), Counter()
                m_terms = {'he', 'him', 'his', 'man', 'father', 'husband', 'boy'}
                f_terms = {'she', 'her', 'hers', 'woman', 'mother', 'wife', 'girl'}
                stop_v = {'be', 'have', 'do', 'go', 'get', 'know', 'think', 'say', 'see', 'look', 'come'}

                for token in doc:
                    if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                        subj = token.text.lower()
                        verb = token.head.lemma_.lower()
                        if verb not in stop_v and len(verb) > 2:
                            if subj in m_terms: m_verbs[verb] += 1
                            elif subj in f_terms: f_verbs[verb] += 1

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                if m_verbs:
                    v, c = zip(*m_verbs.most_common(10))
                    ax1.barh(v, c, color='royalblue')
                    ax1.invert_yaxis()
                    ax1.set_title("Top Male Actions")
                if f_verbs:
                    v, c = zip(*f_verbs.most_common(10))
                    ax2.barh(v, c, color='crimson')
                    ax2.invert_yaxis()
                    ax2.set_title("Top Female Actions")
                st.pyplot(fig)

    # ==========================================
    # TAB 5: COLOR PALETTE
    # ==========================================
    with tab5:
        st.header("The Aesthetic Color Palette")
        if st.button("Extract Colors"):
            with st.spinner("Scanning for aesthetic keywords..."):
                colors = {
                    'red': '#FF0000', 'crimson': '#DC143C', 'blue': '#0000FF', 
                    'green': '#008000', 'yellow': '#FFFF00', 'gold': '#FFD700', 
                    'white': '#FFFFFF', 'black': '#000000', 'grey': '#808080', 'gray': '#808080',
                    'pink': '#FFC0CB', 'purple': '#800080', 'brown': '#A52A2A'
                }
                found_colors = Counter()
                words = re.findall(r'\b[a-zA-Z]+\b', text_data.lower())
                for w in words:
                    if w in colors:
                        found_colors[w] += 1
                
                if found_colors:
                    total = sum(found_colors.values())
                    fig, ax = plt.subplots(figsize=(10, 2))
                    left = 0
                    for c_name, count in found_colors.most_common():
                        width = count / total
                        ax.barh(0, width, left=left, color=colors[c_name], edgecolor='black')
                        left += width
                    ax.set_yticks([])
                    ax.set_title("Relative Frequency of Color Mentions")
                    st.pyplot(fig)
                    st.write(found_colors.most_common())
                else:
                    st.write("No strong color palette detected in this excerpt.")

    # ==========================================
    # TAB 6: MOTIF BARCODE
    # ==========================================
    with tab6:
        st.header("AI Motif Barcode (Lexical Dispersion)")
        if st.button("Generate Barcodes"):
            with st.spinner("Detecting themes automatically..."):
                doc = nlp(text_data[:500000])
                stop_n = {'time', 'day', 'way', 'year', 'man', 'woman', 'thing', 'people', 'room', 'door', 'eyes', 'face'}
                lemmas = []
                candidates = []

                for token in doc:
                    lem = token.lemma_.lower()
                    lemmas.append(lem)
                    if token.pos_ == "NOUN" and not token.is_stop and len(lem) > 2 and lem not in stop_n:
                        candidates.append(lem)

                top_words = [w for w, c in Counter(candidates).most_common(5)]
                
                fig, ax = plt.subplots(figsize=(10, 5))
                bar_colors = ['#e6194B', '#3cb44b', '#f58231', '#4363d8', '#911eb4']
                
                for i, word in enumerate(top_words):
                    positions = [(idx/len(lemmas))*100 for idx, val in enumerate(lemmas) if val == word]
                    y_pos = len(top_words) - i
                    ax.vlines(positions, y_pos - 0.4, y_pos + 0.4, color=bar_colors[i], alpha=0.7)

                ax.set_yticks([len(top_words) - i for i in range(len(top_words))])
                ax.set_yticklabels([w.capitalize() for w in top_words])
                ax.set_xlabel("Book Progress (%)")
                ax.set_title("Top 5 Automatically Detected Motifs")
                st.pyplot(fig)

else:
    st.info("Please upload a PDF file in the sidebar to begin.")
