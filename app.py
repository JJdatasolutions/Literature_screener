import streamlit as st
import PyPDF2
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import pandas as pd
import itertools
import re
import sys
import subprocess

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Scientific Literary Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("📚 Scientific Literary Analysis Dashboard")
st.markdown("A comprehensive, data-driven tool for 12th-grade Modern Languages students to analyze literature.")

# --- CACHING MODELS & NLP SETUP ---
@st.cache_resource(show_spinner="Loading AI Models (This may take a moment on first run)...")
def load_models():
    """Loads spaCy and VADER models securely."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.warning("Downloading spaCy model 'en_core_web_sm'...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    
    nlp.max_length = 2500000 
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
        
    sia = SentimentIntensityAnalyzer()
    return nlp, sia

nlp, sia = load_models()

# --- HELPER FUNCTIONS ---
@st.cache_data
def extract_pdf_pages(file_buffer):
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
    word = word.lower()
    if len(word) <= 3: return 1
    word = re.sub(r'(?:[^laeiouy]es|ed|[^laeiouy]e)$', '', word)
    word = re.sub(r'^y', '', word)
    matches = re.findall(r'[aeiouy]{1,2}', word)
    return max(1, len(matches))

def clean_entity_name(name):
    name = re.sub(r"['’]s\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r'\b(Mr\.|Mrs\.|Ms\.|Dr\.|Lord|Lady|Miss|Uncle|Aunt|Sir)\s', '', name, flags=re.IGNORECASE)
    blacklist = ["Suppose", "Suddenly", "Well", "Yes", "Oh", "And", "But", "Then", "Let", "How"]
    words = name.split()
    if words and words[0] in blacklist:
        words.pop(0)
    return " ".join(words).strip()

@st.cache_data
def get_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)

# --- SESSION STATE INITIALIZATION ---
# Prevents components from disappearing when interacting with widgets
for key in ['net_done', 'arc_done', 'themes_done', 'style_done', 'ai_done']:
    if key not in st.session_state:
        st.session_state[key] = False

# --- SIDEBAR: FILE UPLOAD & SCOPE ---
st.sidebar.header("1. Upload Literature")
uploaded_file = st.sidebar.file_uploader("Upload an English novel/text (PDF)", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting and structuring text..."):
        pdf_pages = extract_pdf_pages(uploaded_file)
        total_pages = len(pdf_pages)
    
    st.sidebar.success(f"Successfully loaded {total_pages} pages.")
    
    st.sidebar.markdown("---")
    st.sidebar.header("2. Analysis Scope")
    st.sidebar.markdown("Select a specific part of the book. **Maximum 250 pages** allowed per analysis to ensure stability.")
    
    if total_pages > 1:
        start_page, end_page = st.sidebar.slider(
            "Select Page Range",
            min_value=1,
            max_value=total_pages,
            value=(1, min(total_pages, 250))
        )
        
        # Enforce the 250-page cap
        if (end_page - start_page) >= 250:
            st.sidebar.warning("Selection automatically capped at 250 pages.")
            end_page = start_page + 249
            
        analyzed_pages = pdf_pages[start_page-1:end_page]
    else:
        analyzed_pages = pdf_pages
        
    analyzed_text = " ".join(analyzed_pages)
    sentences = get_sentences(analyzed_text)

    # Reset session state if text length changes (meaning the user adjusted the slider)
    if 'prev_text_len' not in st.session_state or st.session_state['prev_text_len'] != len(analyzed_text):
        st.session_state['prev_text_len'] = len(analyzed_text)
        for key in ['net_done', 'arc_done', 'themes_done', 'style_done', 'ai_done']:
            st.session_state[key] = False

    # --- TABS CREATION ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🕸️ 1. Character Network", 
        "📈 2. Narrative Arc", 
        "🎭 3. Character Emotions",
        "🧠 4. Themes", 
        "📐 5. Style & Register", 
        "🤔 6. AI Reflection"
    ])

    # ==========================================
    # TAB 1: CHARACTER NETWORK
    # ==========================================
    with tab1:
        st.header("Social Interaction Network")
        st.markdown("Maps out which characters frequently appear together. Node size and color reflect how central the character is.")
        
        if st.button("Generate Character Network") or st.session_state['net_done']:
            st.session_state['net_done'] = True
            with st.spinner("Extracting entities and calculating centrality..."):
                @st.cache_data
                def build_network_data(text):
                    interactions = Counter()
                    char_counts = Counter()
                    doc = nlp(text[:1000000])
                    for sent in doc.sents:
                        chars_in_sent = set()
                        for ent in sent.ents:
                            if ent.label_ == "PERSON" and len(ent.text.split()) < 4:
                                clean_name = clean_entity_name(ent.text)
                                if len(clean_name) > 2 and clean_name.istitle():
                                    chars_in_sent.add(clean_name)
                        for char in chars_in_sent: char_counts[char] += 1
                        if len(chars_in_sent) > 1:
                            for c1, c2 in itertools.combinations(sorted(chars_in_sent), 2):
                                interactions[(c1, c2)] += 1
                    return char_counts, interactions
                
                char_counts, interactions = build_network_data(analyzed_text)
                top_chars = [c for c, count in char_counts.most_common(20)]
                st.session_state['top_chars'] = top_chars 
                
                G = nx.Graph()
                for char in top_chars: G.add_node(char, size=char_counts[char])
                for (c1, c2), weight in interactions.items():
                    if c1 in top_chars and c2 in top_chars: G.add_edge(c1, c2, weight=weight)
                        
                if len(G.nodes) > 0:
                    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#ffffff')
                    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
                    degrees = dict(G.degree(weight='weight'))
                    node_sizes = [v * 50 for v in degrees.values()]
                    node_colors = list(degrees.values())
                    edge_widths = [nx.get_edge_attributes(G, 'weight')[e] * 0.5 for e in G.edges()]
                    
                    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                                           cmap=plt.cm.coolwarm, edgecolors='white', linewidths=1.5, ax=ax)
                    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', alpha=0.3, ax=ax)
                    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax,
                                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.3))
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("Not enough character interactions found.")

    # ==========================================
    # TAB 2: NARRATIVE ARC
    # ==========================================
    with tab2:
        st.header("Global Narrative Sentiment Arc")
        st.markdown("Displays the emotional trajectory across the selected pages.")
        
        if st.button("Generate Narrative Arc") or st.session_state['arc_done']:
            st.session_state['arc_done'] = True
            with st.spinner("Calculating sentiment per page..."):
                @st.cache_data
                def calculate_arc(pages):
                    return [sia.polarity_scores(p)['compound'] for p in pages]
                
                page_scores = calculate_arc(analyzed_pages)
                df_arc = pd.DataFrame({'Page': range(start_page, start_page + len(page_scores)), 'Sentiment': page_scores})
                
                fig = px.line(df_arc, x='Page', y='Sentiment', title="Overall Emotional Arc of the Text",
                              markers=True, template="plotly_white", color_discrete_sequence=['#4361ee'])
                fig.update_layout(yaxis_title="Sentiment (Compound Score)", xaxis_title="Actual Page Number")
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                fig.update_traces(line=dict(width=3), marker=dict(size=8))
                st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # TAB 3: CHARACTER EMOTIONS & CONTEXT
    # ==========================================
    with tab3:
        st.header("Character Emotional Profiling")
        st.markdown("Trace a character's emotional journey and uncover the adjectives associated with them.")
        
        if 'top_chars' in st.session_state and len(st.session_state['top_chars']) > 0:
            selected_char = st.selectbox("Select a Character to analyze:", st.session_state['top_chars'])
            
            if st.button(f"Analyze Emotion for {selected_char}") or st.session_state.get(f'char_done_{selected_char}'):
                st.session_state[f'char_done_{selected_char}'] = True
                with st.spinner(f"Profiling {selected_char}..."):
                    @st.cache_data
                    def profile_character(char_name, sents):
                        progress, sentiment, char_sents = [], [], []
                        first_name = char_name.split()[0] 
                        for i, s in enumerate(sents):
                            if re.search(rf'\b{re.escape(first_name)}\b', s, re.IGNORECASE):
                                score = sia.polarity_scores(s)['compound']
                                sentiment.append(score)
                                progress.append((i / len(sents)) * 100)
                                char_sents.append((s, score))
                        return progress, sentiment, char_sents

                    char_progress, char_sentiment, char_sentences = profile_character(selected_char, sentences)
                    
                    if len(char_sentiment) > 3:
                        pos_adj, neg_adj = Counter(), Counter()
                        for sent_text, score in char_sentences:
                            doc_sent = nlp(sent_text)
                            for token in doc_sent:
                                if token.pos_ == "ADJ" and not token.is_stop and token.is_alpha:
                                    if score > 0.3: pos_adj[token.lemma_.lower()] += 1
                                    elif score < -0.3: neg_adj[token.lemma_.lower()] += 1
                                    
                        avg_sentiment = sum(char_sentiment) / len(char_sentiment)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Mentions in text", len(char_sentiment))
                        col2.metric("Baseline Sentiment", f"{avg_sentiment:.2f}")
                        col3.metric("Dominant Vibe", "Positive 🟢" if avg_sentiment > 0 else "Negative 🔴")
                        
                        # Graph
                        df_char = pd.DataFrame({'Story Progress (%)': char_progress, 'Sentiment': char_sentiment})
                        df_char['Smoothed'] = df_char['Sentiment'].rolling(window=max(2, len(char_sentiment)//8), min_periods=1).mean()

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_char['Story Progress (%)'], y=df_char['Smoothed'],
                                                 fill='tozeroy', mode='lines', line=dict(color='#7209b7', width=3),
                                                 name='Emotional Trend', fillcolor='rgba(114, 9, 183, 0.2)'))
                        fig.add_trace(go.Scatter(x=df_char['Story Progress (%)'], y=df_char['Sentiment'],
                                                 mode='markers', marker=dict(color='gray', size=5, opacity=0.5),
                                                 name='Individual Sentences'))
                        fig.add_hline(y=0, line_dash="dash", line_color="black")
                        fig.update_layout(title=f"Emotional Journey of {selected_char}", xaxis_title="Story Progress (%)", 
                                          yaxis_title="Sentiment Score", template="plotly_white", yaxis_range=[-1.1, 1.1])
                        st.plotly_chart(fig, use_container_width=True)

                        # --- CONTEXT EXPLORER ---
                        st.markdown("---")
                        st.subheader("🔍 Context Explorer")
                        st.markdown(f"Select a specific part of the story to read the exact sentences involving **{selected_char}**.")
                        
                        prog_min, prog_max = st.slider("Select Story Progress Range (%)", 0, 100, (0, 100))
                        
                        filtered_sentences = [(p, t, s) for p, t, s in zip(char_progress, [s[0] for s in char_sentences], char_sentiment) 
                                              if prog_min <= p <= prog_max]
                        
                        if filtered_sentences:
                            st.write(f"Found **{len(filtered_sentences)}** occurrences in this range. *(Displaying max 25)*")
                            for p, txt, sc in filtered_sentences[:25]:
                                color = "#d8f3dc" if sc > 0.3 else ("#ffe5d9" if sc < -0.3 else "#f8f9fa")
                                st.markdown(f"""
                                <div style='background-color:{color}; padding:10px; border-radius:5px; margin-bottom:8px; border-left: 4px solid gray;'>
                                    <small style='color:gray;'>Progress: {p:.1f}% | Emotion Score: {sc:.2f}</small><br>
                                    {txt}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No mentions found in this specific range.")
                    else:
                        st.warning(f"Not enough data points found to profile {selected_char}.")
        else:
            st.info("💡 Please generate the 'Character Network' in Tab 1 first to identify the main characters.")

    # ==========================================
    # TAB 4: THEMATIC ANALYSIS & CONTEXT
    # ==========================================
    with tab4:
        st.header("Thematic Analysis")
        st.markdown("Discover core themes by exclusively extracting **Nouns**.")
        
        if st.button("Analyze Themes") or st.session_state['themes_done']:
            st.session_state['themes_done'] = True
            with st.spinner("Extracting nouns..."):
                @st.cache_data
                def extract_themes(text):
                    doc = nlp(text[:1000000])
                    lemmas = Counter()
                    for token in doc:
                        if token.pos_ == "NOUN" and not token.is_stop and token.is_alpha and len(token.text) > 2:
                            lemmas[token.lemma_.lower()] += 1
                    return lemmas.most_common(20)

                top_20 = extract_themes(analyzed_text)
                
                if top_20:
                    df_themes = pd.DataFrame(top_20, columns=["Noun", "Frequency"])
                    fig = px.bar(df_themes, x='Noun', y='Frequency', title="Top 20 Thematic Nouns",
                                 color='Frequency', color_continuous_scale='Viridis', template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- CONTEXT EXPLORER ---
                    st.markdown("---")
                    st.subheader("🔍 Context Explorer")
                    st.markdown("Select a theme to see the exact sentences where it is used in the text.")
                    
                    theme_words = [n for n, c in top_20]
                    selected_theme = st.selectbox("Select a noun:", theme_words)
                    
                    if selected_theme:
                        matches = [s for s in sentences if re.search(rf'\b{re.escape(selected_theme)}\b', s, re.IGNORECASE)]
                        st.write(f"Found **{len(matches)}** sentences containing '*{selected_theme}*'. *(Displaying max 25)*")
                        
                        for m in matches[:25]:
                            # Highlight the word in the sentence
                            highlighted = re.sub(rf'(\b{re.escape(selected_theme)}\b)', r'**\1**', m, flags=re.IGNORECASE)
                            st.markdown(f"- {highlighted}")
                else:
                    st.warning("Could not extract enough data for thematic analysis.")

    # ==========================================
    # TAB 5: LINGUISTIC STYLE & REGISTER
    # ==========================================
    with tab5:
        st.header("Linguistic Style & Register")
        if st.button("Calculate Style Metrics") or st.session_state['style_done']:
            st.session_state['style_done'] = True
            with st.spinner("Parsing syntax and calculating readability..."):
                @st.cache_data
                def calc_style(text):
                    doc = nlp(text[:500000]) 
                    tot_sents = tot_words = tot_syllables = adj_c = verb_c = 0
                    for sent in doc.sents:
                        tot_sents += 1
                        for token in sent:
                            if token.is_alpha:
                                tot_words += 1
                                tot_syllables += count_syllables(token.text)
                                if token.pos_ == "ADJ": adj_c += 1
                                elif token.pos_ == "VERB": verb_c += 1
                    return tot_sents, tot_words, tot_syllables, adj_c, verb_c
                
                tot_sents, tot_words, tot_syllables, adj_count, verb_count = calc_style(analyzed_text)
                
                if tot_sents > 0 and tot_words > 0:
                    total_action_desc = adj_count + verb_count
                    verb_ratio = (verb_count / total_action_desc) * 100 if total_action_desc > 0 else 0
                    flesch_score = 206.835 - 1.015 * (tot_words / tot_sents) - 84.6 * (tot_syllables / tot_words)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_style = go.Figure(go.Indicator(
                            mode="gauge+number", value=verb_ratio, title={'text': "Style: Descriptive vs Action", 'font': {'size': 18}},
                            number={'suffix': "% Verbs"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#3a0ca3"},
                                   'steps': [{'range': [0, 45], 'color': "#4cc9f0"}, {'range': [45, 55], 'color': "#e9ecef"}, {'range': [55, 100], 'color': "#f72585"}]}
                        ))
                        st.plotly_chart(fig_style, use_container_width=True)

                    with col2:
                        register = "Conversational" if flesch_score > 70 else "Standard" if flesch_score > 50 else "Complex" if flesch_score > 30 else "Academic"
                        fig_read = go.Figure(go.Indicator(
                            mode="gauge+number", value=max(0, min(flesch_score, 100)), title={'text': f"Flesch-Kincaid Score<br><span style='font-size:0.8em;color:gray'>Register: {register}</span>", 'font': {'size': 18}},
                            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "black"},
                                   'steps': [{'range': [0, 30], 'color': "#d00000"}, {'range': [30, 60], 'color': "#ffba08"}, {'range': [60, 100], 'color': "#3f88c5"}]}
                        ))
                        st.plotly_chart(fig_read, use_container_width=True)

    # ==========================================
    # TAB 6: AI REFLECTION
    # ==========================================
    with tab6:
        st.header("Critical AI Evaluation")
        if st.button("Find Extreme Sentiment Fragments") or st.session_state['ai_done']:
            st.session_state['ai_done'] = True
            with st.spinner("Scanning for extreme emotional polarity..."):
                @st.cache_data
                def get_extremes(sents):
                    scored = []
                    for s in sents:
                        c_s = s.strip().replace('\n', ' ')
                        if len(c_s.split()) > 8: 
                            scored.append({'text': c_s, 'score': sia.polarity_scores(c_s)['compound']})
                    scored.sort(key=lambda x: x['score'])
                    return scored[:3], scored[-3:][::-1]

                top_3_neg, top_3_pos = get_extremes(sentences)
                
                col_neg, col_pos = st.columns(2)
                with col_neg:
                    st.error("#### 🔴 Top 3 Negative Fragments")
                    for item in top_3_neg: st.markdown(f"<div style='background:#ffe5d9;padding:10px;border-radius:5px;margin-bottom:10px;'><strong>Score: {item['score']:.2f}</strong><br><em>\"{item['text']}\"</em></div>", unsafe_allow_html=True)
                with col_pos:
                    st.success("#### 🟢 Top 3 Positive Fragments")
                    for item in top_3_pos: st.markdown(f"<div style='background:#d8f3dc;padding:10px;border-radius:5px;margin-bottom:10px;'><strong>Score: {item['score']:.2f}</strong><br><em>\"{item['text']}\"</em></div>", unsafe_allow_html=True)

else:
    st.info("Please upload a PDF document in the sidebar to begin your literary analysis.")
