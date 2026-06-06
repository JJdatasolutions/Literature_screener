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
import gc  # De vuilnisman van Python

# --- VISUAL DESIGN SETTINGS ---
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Literary Dashboard", layout="wide")
st.title("📚 The Ultimate Literary Analysis Dashboard")
st.markdown("Upload a novel (PDF) and let AI uncover its hidden structures.")

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
    
    # We voegen een lichte zinnen-splitser toe om de zware 'parser' te kunnen omzeilen bij de netwerkanalyse
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
        
    nlp.max_length = 2000000 
    sia = SentimentIntensityAnalyzer()
    return nlp, sia

nlp, sia = load_models()

@st.cache_data
def extract_text(file_buffer):
    reader = PyPDF2.PdfReader(file_buffer)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + " "
    return re.sub(r'\s+', ' ', full_text)

# --- MEMORY SAFE CHUNKING WITH PROGRESS ---
def get_chunks(text, chunk_size=50000):
    """Hak tekst in nog kleinere stukken (50k ipv 100k) voor absolute veiligheid."""
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

# --- SIDEBAR UPLOAD ---
st.sidebar.header("1. Upload Book")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting full text from PDF..."):
        text_data = extract_text(uploaded_file)
        chunks = list(get_chunks(text_data)) # Sla chunks lokaal op om voortgang te kunnen meten
        total_chunks = len(chunks)
    
    # Create Tabs
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
        st.info("**Legend:** The blue line represents the average length of sentences over a rolling window. High peaks mean long, complex sentences. Deep valleys mean short, punchy sentences.")
        st.success("**Guiding Questions for Students:**\n- Where does the author use short, punchy sentences? Does this correlate with action sequences or dialogue?\n- Where are the peaks? Do these long sentences indicate heavy description, philosophical thoughts, or a 'stream of consciousness'?")

        if st.button("Run Style Analysis"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            sent_lengths = []
            
            # Disable NER en andere zware modules die we niet nodig hebben voor zinslengte
            for i, doc in enumerate(nlp.pipe(chunks, disable=["ner", "lemmatizer", "textcat", "custom"])):
                for s in doc.sents:
                    length = len([t for t in s if not t.is_punct])
                    if length > 2:
                        sent_lengths.append(length)
                
                # Update UI om timeout te voorkomen en ruim geheugen op
                progress_bar.progress((i + 1) / total_chunks)
                status_text.text(f"Processing chunk {i + 1} of {total_chunks}...")
                gc.collect()

            status_text.text("Generating visual...")
            if sent_lengths:
                avg_len = sum(sent_lengths) / len(sent_lengths)
                st.metric("Average Sentence Length", f"{avg_len:.1f} words")
                
                window = max(30, len(sent_lengths) // 100) 
                smoothed = [sum(sent_lengths[i:i+window])/window for i in range(len(sent_lengths)-window+1)]
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(smoothed, color='#2b8cbe', linewidth=2)
                ax.fill_between(range(len(smoothed)), smoothed, color='#2b8cbe', alpha=0.2)
                ax.set_title("Sentence Length Over Time", fontsize=14, fontweight='bold')
                ax.set_ylabel("Words per Sentence")
                ax.set_xlabel("Story Progression (Sentences)")
                sns.despine()
                st.pyplot(fig)
                plt.close(fig)
                progress_bar.empty()
                status_text.empty()

    # ==========================================
    # TAB 2: VONNEGUT ARC
    # ==========================================
    with tab2:
        st.header("The Vonnegut Emotion Arc")
        st.info("**Legend:** The Y-axis represents the emotional tone (-1 is extreme negativity, +1 is extreme positivity). The X-axis represents the progression of the book.")
        st.success("**Guiding Questions for Students:**\n- Where is the lowest point (the darkest moment) of the story? What happens in the plot at this exact percentage?\n- According to Kurt Vonnegut, stories have emotional 'shapes' (like 'Man in Hole' or 'Boy Meets Girl'). What shape is this novel?")

        if st.button("Run Emotion Arc"):
            with st.spinner("Calculating sentiment..."):
                # Geen SpaCy nodig hier, puur Python regex (super snel)
                sentences = re.split(r'(?<=[.!?]) +', text_data)
                scores = []
                for s in sentences:
                    if len(s) > 10:
                        scores.append(sia.polarity_scores(s)['compound'])
                
                gc.collect() # Clean up

                if scores:
                    window = max(1, len(scores) // 20)
                    smoothed = [sum(scores[i:i+window])/window for i in range(len(scores)-window+1)]
                    x_perc = [(i / len(smoothed)) * 100 for i in range(len(smoothed))]
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    import pandas as pd
                    ax.plot(x_perc, smoothed, color='#8856a7', linewidth=2.5)
                    ax.fill_between(x_perc, smoothed, 0, where=pd.Series(smoothed) > 0, color='green', alpha=0.2, interpolate=True)
                    ax.fill_between(x_perc, smoothed, 0, where=pd.Series(smoothed) < 0, color='red', alpha=0.2, interpolate=True)
                    
                    ax.axhline(0, color='black', linewidth=1, linestyle='--')
                    ax.set_title("Narrative Sentiment Arc", fontsize=14, fontweight='bold')
                    ax.set_xlabel("Story Progress (%)")
                    ax.set_ylabel("Sentiment Score")
                    sns.despine()
                    st.pyplot(fig)
                    plt.close(fig)

    # ==========================================
    # TAB 3: SOCIAL NETWORK
    # ==========================================
    with tab3:
        st.header("Social Web & Relationship Dynamics")
        st.info("**Legend:**\n- **Node Size:** Mentions.\n- **Edge Thickness:** Interactions.\n- **Edge Color:** Green = Positive, Red = Negative.")
        st.success("**Guiding Questions for Students:**\n- Who is the central 'hub' of the novel? Are there characters isolated on the edges?\n- Look at the red lines: which characters drive the central conflict of the story?")

        if st.button("Run Network Analysis"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            char_counts = Counter()
            interactions = {}

            # Disable tagger, parser, and lemmatizer. We ONLY need NER (Named Entity Recognition) and the light sentencizer.
            for i, doc in enumerate(nlp.pipe(chunks, disable=["tagger", "parser", "lemmatizer", "textcat", "custom"])):
                for sent in doc.sents:
                    raw_chars = [ent.text for ent in sent.ents if ent.label_ == "PERSON" and len(ent.text) > 3]
                    cleaned_chars = set([re.sub(r"['’]s$", "", char).strip() for char in raw_chars])
                    
                    for char in cleaned_chars:
                        char_counts[char] += 1
                    
                    if len(cleaned_chars) > 1:
                        sent_score = sia.polarity_scores(sent.text)['compound']
                        for c1, c2 in itertools.combinations(cleaned_chars, 2):
                            pair = tuple(sorted([c1, c2]))
                            if pair not in interactions:
                                interactions[pair] = {'weight': 0, 'sentiment': []}
                            interactions[pair]['weight'] += 1
                            interactions[pair]['sentiment'].append(sent_score)
                
                progress_bar.progress((i + 1) / total_chunks)
                status_text.text(f"Extracting entities: chunk {i + 1} of {total_chunks}...")
                gc.collect()

            status_text.text("Drawing the network... (This might take a moment for large webs)")
            if char_counts:
                top_chars = [c for c, count in char_counts.most_common(15)]
                G = nx.Graph()
                
                for char in top_chars:
                    G.add_node(char, size=char_counts[char]*60)
                    
                for (c1, c2), data in interactions.items():
                    if c1 in top_chars and c2 in top_chars:
                        avg_sent = sum(data['sentiment']) / len(data['sentiment'])
                        if avg_sent > 0.1: color = '#2ca02c'
                        elif avg_sent < -0.1: color = '#d62728'
                        else: color = '#b0b0b0'
                        G.add_edge(c1, c2, weight=data['weight'], color=color)

                if len(G.nodes) > 0:
                    fig, ax = plt.subplots(figsize=(12, 9))
                    pos = nx.kamada_kawai_layout(G)
                    
                    sizes = [nx.get_node_attributes(G, 'size')[n] for n in G.nodes()]
                    edge_colors = [nx.get_edge_attributes(G, 'color')[e] for e in G.edges()]
                    weights = [nx.get_edge_attributes(G, 'weight')[e] for e in G.edges()]
                    max_w = max(weights) if weights else 1
                    scaled_weights = [(w/max_w)*5 + 1 for w in weights]
                    
                    nx.draw_networkx_edges(G, pos, width=scaled_weights, edge_color=edge_colors, 
                                        alpha=0.6, connectionstyle="arc3,rad=0.1", ax=ax)
                    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='#a6bddb', 
                                        edgecolors='black', linewidths=1.5, ax=ax)
                    nx.draw_networkx_labels(G, pos, font_size=11, font_family='sans-serif', font_weight='bold',
                                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1), ax=ax)
                    
                    ax.set_title("Character Interaction Network", fontsize=16, fontweight='bold')
                    plt.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
            
            progress_bar.empty()
            status_text.empty()

    # ==========================================
    # TAB 4: GENDER BIAS
    # ==========================================
    with tab4:
        st.header("Gender-Bias Agency Scanner")
        st.info("**Legend:** Top 10 verbs directly associated with male vs female entities. Neutral verbs ('turn', 'go') are filtered out.")
        st.success("**Guiding Questions for Students:**\n- Are male characters assigned more active, aggressive, or physical verbs? \n- Are female characters associated with passive, emotional, or reactive verbs?")

        if st.button("Run Gender Analysis"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            m_verbs, f_verbs = Counter(), Counter()
            m_terms = {'he', 'him', 'his', 'man', 'men', 'boy', 'boys', 'father', 'brother', 'husband', 'uncle', 'gentleman'}
            f_terms = {'she', 'her', 'hers', 'woman', 'women', 'girl', 'girls', 'mother', 'sister', 'wife', 'aunt', 'lady'}
            stop_v = {
                'be', 'have', 'do', 'go', 'get', 'know', 'think', 'say', 'see', 'look', 
                'come', 'tell', 'ask', 'seem', 'turn', 'move', 'start', 'begin', 'stop', 
                'use', 'try', 'feel', 'leave', 'make', 'take', 'give', 'find', 'call', 
                'want', 'let', 'put', 'keep', 'show', 'hold', 'bring', 'become', 'mean'
            }

            # Disable NER, we only need syntax parsing and tagging here
            for i, doc in enumerate(nlp.pipe(chunks, disable=["ner", "textcat", "custom"])):
                for token in doc:
                    if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                        subj = token.text.lower()
                        verb = token.head.lemma_.lower()
                        if verb not in stop_v and len(verb) > 2:
                            if subj in m_terms: 
                                m_verbs[verb] += 1
                            elif subj in f_terms: 
                                f_verbs[verb] += 1
                
                progress_bar.progress((i + 1) / total_chunks)
                status_text.text(f"Scanning grammar: chunk {i + 1} of {total_chunks}...")
                gc.collect()

            status_text.text("Generating visual...")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            if m_verbs:
                v, c = zip(*m_verbs.most_common(10))
                sns.barplot(x=list(c), y=list(v), ax=ax1, color='#3182bd')
                ax1.set_title("Top Male Actions", fontsize=12, fontweight='bold')
                ax1.set_xlabel("Frequency")
            
            if f_verbs:
                v, c = zip(*f_verbs.most_common(10))
                sns.barplot(x=list(c), y=list(v), ax=ax2, color='#e6550d')
                ax2.set_title("Top Female Actions", fontsize=12, fontweight='bold')
                ax2.set_xlabel("Frequency")
            
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            progress_bar.empty()
            status_text.empty()

    # ==========================================
    # TAB 5: COLOR PALETTE
    # ==========================================
    with tab5:
        st.header("The Aesthetic Color Palette")
        st.info("**Legend:** Proportion of how often specific colors are explicitly mentioned.")
        st.success("**Guiding Questions for Students:**\n- What is the dominant aesthetic or atmosphere based on these colors?\n- Do certain colors carry symbolic meaning in this novel?")

        if st.button("Extract Colors"):
            with st.spinner("Scanning for aesthetic keywords..."):
                colors = {
                    'red': '#e41a1c', 'crimson': '#bd0026', 'blue': '#377eb8', 
                    'green': '#4daf4a', 'yellow': '#dede00', 'gold': '#ff7f00', 
                    'white': '#f0f0f0', 'black': '#252525', 'grey': '#969696', 'gray': '#969696',
                    'pink': '#f781bf', 'purple': '#984ea3', 'brown': '#a65628'
                }
                found_colors = Counter()
                words = re.findall(r'\b[a-zA-Z]+\b', text_data.lower())
                for w in words:
                    if w in colors:
                        found_colors[w] += 1
                
                gc.collect()

                if found_colors:
                    total = sum(found_colors.values())
                    fig, ax = plt.subplots(figsize=(10, 2))
                    left = 0
                    for c_name, count in found_colors.most_common():
                        width = count / total
                        ax.barh(0, width, left=left, color=colors[c_name], edgecolor='black', linewidth=1.5)
                        if width > 0.05:
                            ax.text(left + width/2, 0, f"{c_name}\n{int(width*100)}%", 
                                    ha='center', va='center', color='white' if c_name not in ['white', 'yellow', 'gold', 'f0f0f0'] else 'black',
                                    fontweight='bold', fontsize=10)
                        left += width
                        
                    ax.set_yticks([])
                    ax.set_xticks([])
                    ax.set_title("Novel Aesthetic Proportion", fontsize=14, fontweight='bold')
                    sns.despine(left=True, bottom=True)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.write("No strong color palette detected in this excerpt.")

else:
    st.info("Please upload a PDF file in the sidebar to begin.")
