import streamlit as st
import PyPDF2
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import pandas as pd
import itertools
import re
import sys
import os
import subprocess

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Literary Analysis Dashboard", layout="wide")
st.title("📚 Scientific Literary Analysis Dashboard")
st.markdown("A comprehensive tool for 12th-grade Modern Languages students to analyze literature through data.")

# --- CACHING MODELS & NLP SETUP ---
@st.cache_resource(show_spinner="Loading NLP Models (This may take a moment on first run)...")
def load_models():
    """Loads spaCy and VADER models securely."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.warning("Downloading spaCy model 'en_core_web_sm'...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    
    # Increase max length for full books
    nlp.max_length = 2500000 
    
    # Add sentencizer for faster sentence boundary detection
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
        
    sia = SentimentIntensityAnalyzer()
    return nlp, sia

nlp, sia = load_models()

# --- HELPER FUNCTIONS ---
@st.cache_data
def extract_pdf_pages(file_buffer):
    """Extracts text page by page from a PDF with robust error handling."""
    pages = []
    try:
        reader = PyPDF2.PdfReader(file_buffer)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return pages
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return []

def count_syllables(word):
    """Simple heuristic to count syllables for Flesch-Kincaid."""
    word = word.lower()
    if len(word) <= 3:
        return 1
    word = re.sub(r'(?:[^laeiouy]es|ed|[^laeiouy]e)$', '', word)
    word = re.sub(r'^y', '', word)
    matches = re.findall(r'[aeiouy]{1,2}', word)
    return max(1, len(matches))

def clean_entity_name(name):
    """Cleans up character names (removes titles and possessives)."""
    name = re.sub(r"['’]s\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r'\b(Mr\.|Mrs\.|Ms\.|Dr\.|Lord|Lady|Miss)\s', '', name, flags=re.IGNORECASE)
    return name.strip()

# --- SIDEBAR: FILE UPLOAD ---
st.sidebar.header("1. Upload Literature")
uploaded_file = st.sidebar.file_uploader("Upload an English novel/text (PDF format)", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        pdf_pages = extract_pdf_pages(uploaded_file)
        total_pages = len(pdf_pages)
        full_text = " ".join(pdf_pages)
    
    st.sidebar.success(f"Successfully loaded {total_pages} pages.")
    
    # Optional: Limit analysis scope for performance
    st.sidebar.markdown("---")
    max_pages = st.sidebar.slider("Pages to analyze (Limit to avoid memory overload)", 
                                  min_value=1, max_value=total_pages, value=min(total_pages, 50))
    
    analyzed_pages = pdf_pages[:max_pages]
    analyzed_text = " ".join(analyzed_pages)

    # --- TABS CREATION ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🕸️ 1. Network & Narrative Arc", 
        "🧠 2. Thematic Analysis", 
        "📐 3. Linguistic Style & Register", 
        "🤔 4. AI Reflection & Evaluation"
    ])

    # ==========================================
    # TAB 1: CHARACTER NETWORK & NARRATIVE ARC
    # ==========================================
    with tab1:
        st.header("Character Network & Narrative Sentiment Arc")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Narrative Sentiment Arc")
            st.info("Displays the emotional trajectory (positive vs. negative) across the selected pages.")
            
            if st.button("Generate Narrative Arc"):
                with st.spinner("Calculating sentiment per page..."):
                    page_scores = []
                    for i, page_text in enumerate(analyzed_pages):
                        score = sia.polarity_scores(page_text)['compound']
                        page_scores.append(score)
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(range(1, len(page_scores) + 1), page_scores, color='#8856a7', marker='o', linestyle='-', markersize=4)
                    ax.axhline(0, color='black', linewidth=1, linestyle='--')
                    ax.set_xlabel("Page Number")
                    ax.set_ylabel("Sentiment (Compound Score)")
                    ax.set_title("Emotional Arc of the Text")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)

        with col2:
            st.subheader("Character Interaction Map")
            st.info("Maps out which characters frequently appear together in the same sentences.")
            
            if st.button("Generate Character Network"):
                with st.spinner("Extracting entities and building network..."):
                    interactions = Counter()
                    char_counts = Counter()
                    
                    # Process text in chunks to manage memory
                    doc = nlp(analyzed_text[:1000000]) # Cap at 1M chars for safety
                    
                    for sent in doc.sents:
                        chars_in_sent = set()
                        for ent in sent.ents:
                            if ent.label_ == "PERSON" and len(ent.text.split()) < 4:
                                clean_name = clean_entity_name(ent.text)
                                if len(clean_name) > 2 and clean_name.istitle():
                                    chars_in_sent.add(clean_name)
                        
                        for char in chars_in_sent:
                            char_counts[char] += 1
                            
                        if len(chars_in_sent) > 1:
                            for c1, c2 in itertools.combinations(sorted(chars_in_sent), 2):
                                interactions[(c1, c2)] += 1
                                
                    # Filter top characters
                    top_chars = [c for c, count in char_counts.most_common(15)]
                    
                    G = nx.Graph()
                    for char in top_chars:
                        G.add_node(char, size=char_counts[char])
                        
                    for (c1, c2), weight in interactions.items():
                        if c1 in top_chars and c2 in top_chars:
                            G.add_edge(c1, c2, weight=weight)
                            
                    if len(G.nodes) > 0:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        pos = nx.spring_layout(G, k=0.5, seed=42)
                        
                        node_sizes = [nx.get_node_attributes(G, 'size')[n] * 50 for n in G.nodes()]
                        edge_widths = [nx.get_edge_attributes(G, 'weight')[e] * 0.5 for e in G.edges()]
                        
                        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#a8dadc', edgecolors='#1d3557', ax=ax)
                        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, ax=ax)
                        nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)
                        
                        ax.axis("off")
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.warning("Not enough character interactions found in the selected text.")

    # ==========================================
    # TAB 2: THEMATIC ANALYSIS
    # ==========================================
    with tab2:
        st.header("Thematic Analysis (Lemmatization)")
        st.markdown("**Objective:** Discover the core themes of the text by filtering out grammatical noise and focusing on semantically rich words (Nouns, Verbs, Adjectives).")
        
        if st.button("Analyze Themes"):
            with st.spinner("Lemmatizing and filtering vocabulary..."):
                doc = nlp(analyzed_text[:1000000])
                lemmas = Counter()
                
                # Semantic Filtering
                allowed_pos = {"NOUN", "VERB", "ADJ"}
                
                for token in doc:
                    if token.pos_ in allowed_pos and not token.is_stop and token.is_alpha and len(token.text) > 2:
                        lemmas[token.lemma_.lower()] += 1
                
                if lemmas:
                    top_20 = lemmas.most_common(20)
                    df_themes = pd.DataFrame(top_20, columns=["Lemma", "Frequency"])
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.bar(df_themes["Lemma"], df_themes["Frequency"], color="#457b9d")
                    plt.xticks(rotation=45, ha='right')
                    ax.set_ylabel("Frequency")
                    ax.set_title("Top 20 Semantic Keywords")
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("Could not extract enough data for thematic analysis.")

    # ==========================================
    # TAB 3: LINGUISTIC STYLE & REGISTER
    # ==========================================
    with tab3:
        st.header("Linguistic Style & Register")
        st.markdown("Analyze the author's syntactic choices and text complexity (**WD3_02.07.04** & **WD3_02.09.01**).")
        
        if st.button("Calculate Style Metrics"):
            with st.spinner("Parsing syntax and calculating readability..."):
                doc = nlp(analyzed_text[:500000]) # Cap for speed
                
                total_sentences = 0
                total_words = 0
                total_syllables = 0
                adj_count = 0
                verb_count = 0
                
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
                
                # Metrics Calculation
                if total_sentences > 0 and total_words > 0:
                    # Descriptive Density
                    total_action_desc = adj_count + verb_count
                    if total_action_desc > 0:
                        adj_ratio = (adj_count / total_action_desc) * 100
                        verb_ratio = (verb_count / total_action_desc) * 100
                    else:
                        adj_ratio = verb_ratio = 0
                        
                    # Flesch Reading Ease Formula
                    flesch_score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
                    
                    st.subheader("1. Descriptive Density (Adjectives vs. Verbs)")
                    col1, col2 = st.columns(2)
                    col1.metric("Adjective Ratio (Descriptive)", f"{adj_ratio:.1f}%")
                    col2.metric("Verb Ratio (Action-Oriented)", f"{verb_ratio:.1f}%")
                    
                    st.progress(int(adj_ratio))
                    st.caption("👈 More Adjectives (Poetic/Descriptive) | More Verbs (Action-driven) 👉")
                    
                    st.markdown("---")
                    st.subheader("2. Text Complexity & Readability")
                    st.markdown("Calculated using the **Flesch-Kincaid Reading Ease** formula:")
                    st.latex(r"FRE = 206.835 - 1.015 \left( \frac{\text{Total Words}}{\text{Total Sentences}} \right) - 84.6 \left( \frac{\text{Total Syllables}}{\text{Total Words}} \right)")
                    
                    if flesch_score > 70: register = "Conversational / Accessible"
                    elif flesch_score > 50: register = "Standard / Intermediate"
                    elif flesch_score > 30: register = "Formal / Complex"
                    else: register = "Academic / Highly Complex"
                    
                    col3, col4 = st.columns(2)
                    col3.metric("Flesch Reading Ease Score", f"{flesch_score:.2f}")
                    col4.metric("Assessed Register", register)

    # ==========================================
    # TAB 4: AI REFLECTION & EVALUATION
    # ==========================================
    with tab4:
        st.header("Critical AI Evaluation & Research Log")
        st.markdown("**WD3_02.07.03:** Evaluate the limitations of AI when dealing with figurative language, irony, or subtext.")
        
        if st.button("Find Extreme Sentiment Fragments"):
            with st.spinner("Scanning for extreme emotional polarity..."):
                # Split text into rough sentences based on punctuation
                raw_sentences = re.split(r'(?<=[.!?]) +', analyzed_text)
                scored_sentences = []
                
                for s in raw_sentences:
                    clean_s = s.strip().replace('\n', ' ')
                    if len(clean_s.split()) > 8: # Only consider full sentences
                        score = sia.polarity_scores(clean_s)['compound']
                        scored_sentences.append({'text': clean_s, 'score': score})
                
                # Sort by sentiment score
                scored_sentences.sort(key=lambda x: x['score'])
                
                top_3_negative = scored_sentences[:3]
                top_3_positive = scored_sentences[-3:]
                top_3_positive.reverse()
                
                st.subheader("AI's Most Extreme Classifications")
                
                st.markdown("#### 🔴 Top 3 Most Negative Fragments")
                for i, item in enumerate(top_3_negative):
                    st.error(f"**Score: {item['score']:.2f}** | \"{item['text']}\"")
                    
                st.markdown("#### 🟢 Top 3 Most Positive Fragments")
                for i, item in enumerate(top_3_positive):
                    st.success(f"**Score: {item['score']:.2f}** | \"{item['text']}\"")
                    
                st.markdown("---")
                st.subheader("Student Reflection Task")
                st.markdown("""
                *Read the fragments above carefully.*
                1. **Irony & Sarcasm:** Did the AI label an ironic or sarcastic sentence as genuinely positive/negative?
                2. **Figurative Language:** Did the AI misinterpret a metaphor (e.g., "killing it") literally?
                3. **Contextual Nuance:** Is the emotion of the fragment different when placed in the broader context of the story?
                """)
                
        st.markdown("---")
        st.subheader("Research Cycle Log (WD3_01.01.01)")
        student_notes = st.text_area("Record your observations, answers to the reflection tasks, and formulate your research conclusions here:", height=250)
        
        if st.button("Save Notes (Session Only)"):
            st.success("Notes temporarily saved to session state. Be sure to copy them to your final research report!")

else:
    st.info("Please upload a PDF document in the sidebar to begin your literary analysis.")
