import streamlit as st
import PyPDF2
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

# --- SIDEBAR: FILE UPLOAD ---
st.sidebar.header("1. Upload Literature")
uploaded_file = st.sidebar.file_uploader("Upload an English novel/text (PDF)", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting and structuring text..."):
        pdf_pages = extract_pdf_pages(uploaded_file)
        total_pages = len(pdf_pages)
    
    st.sidebar.success(f"Successfully loaded {total_pages} pages.")
    
    st.sidebar.markdown("---")
    st.sidebar.header("2. Analysis Scope")
    max_pages = st.sidebar.slider("Pages to analyze (Limit to avoid memory overload)", 
                                  min_value=1, max_value=total_pages, value=min(total_pages, 80))
    
    analyzed_pages = pdf_pages[:max_pages]
    analyzed_text = " ".join(analyzed_pages)
    sentences = re.split(r'(?<=[.!?]) +', analyzed_text)

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
        st.markdown("Maps out which characters frequently appear together. **Node size and color** reflect how central the character is to the narrative.")
        
        if st.button("Generate Character Network", key="btn_network"):
            with st.spinner("Extracting entities and calculating centrality..."):
                interactions = Counter()
                char_counts = Counter()
                doc = nlp(analyzed_text[:1000000])
                
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
                            
                top_chars = [c for c, count in char_counts.most_common(20)]
                st.session_state['top_chars'] = top_chars 
                
                G = nx.Graph()
                for char in top_chars:
                    G.add_node(char, size=char_counts[char])
                    
                for (c1, c2), weight in interactions.items():
                    if c1 in top_chars and c2 in top_chars:
                        G.add_edge(c1, c2, weight=weight)
                        
                if len(G.nodes) > 0:
                    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#ffffff')
                    
                    # Optimization: Use spring layout with a higher 'k' to push nodes apart
                    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
                    
                    # Optimization: Centrality-based styling
                    degrees = dict(G.degree(weight='weight'))
                    node_sizes = [v * 50 for v in degrees.values()]
                    node_colors = list(degrees.values())
                    edge_widths = [nx.get_edge_attributes(G, 'weight')[e] * 0.5 for e in G.edges()]
                    
                    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
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
        st.markdown("Displays the emotional trajectory (positive vs. negative) across the selected pages.")
        
        if st.button("Generate Narrative Arc", key="btn_arc"):
            with st.spinner("Calculating sentiment per page..."):
                page_scores = []
                for page_text in analyzed_pages:
                    score = sia.polarity_scores(page_text)['compound']
                    page_scores.append(score)
                
                df_arc = pd.DataFrame({'Page': range(1, len(page_scores) + 1), 'Sentiment': page_scores})
                
                fig = px.line(df_arc, x='Page', y='Sentiment', title="Overall Emotional Arc of the Text",
                              markers=True, template="plotly_white", color_discrete_sequence=['#4361ee'])
                fig.update_layout(yaxis_title="Sentiment (Compound Score)", xaxis_title="Page Number")
                fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Neutral Line")
                fig.update_traces(line=dict(width=3), marker=dict(size=8))
                
                st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # TAB 3: CHARACTER EMOTIONS (INSIGHTFUL UPDATE)
    # ==========================================
    with tab3:
        st.header("Character Emotional Profiling")
        st.markdown("Trace a character's emotional journey and uncover the **adjectives** most associated with them in positive and negative contexts.")
        
        if 'top_chars' in st.session_state and len(st.session_state['top_chars']) > 0:
            selected_char = st.selectbox("Select a Character to analyze:", st.session_state['top_chars'])
            
            if st.button(f"Analyze Emotion for {selected_char}"):
                with st.spinner(f"Profiling {selected_char}..."):
                    char_progress = []
                    char_sentiment = []
                    char_sentences = []
                    
                    first_name = selected_char.split()[0] 
                    for i, s in enumerate(sentences):
                        if re.search(rf'\b{re.escape(first_name)}\b', s, re.IGNORECASE):
                            score = sia.polarity_scores(s)['compound']
                            char_sentiment.append(score)
                            char_progress.append((i / len(sentences)) * 100)
                            char_sentences.append((s, score))
                    
                    if len(char_sentiment) > 3:
                        # 1. Linguistic Insight: Associated Adjectives
                        pos_adjectives = Counter()
                        neg_adjectives = Counter()
                        
                        for sent_text, score in char_sentences:
                            doc_sent = nlp(sent_text)
                            for token in doc_sent:
                                if token.pos_ == "ADJ" and not token.is_stop and token.is_alpha:
                                    if score > 0.3: pos_adjectives[token.lemma_.lower()] += 1
                                    elif score < -0.3: neg_adjectives[token.lemma_.lower()] += 1
                                    
                        avg_sentiment = sum(char_sentiment) / len(char_sentiment)
                        
                        # Display Insights Layout
                        st.markdown(f"### Emotional Profile: {selected_char}")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Mentions in text", len(char_sentiment))
                        col2.metric("Baseline Sentiment", f"{avg_sentiment:.2f}")
                        col3.metric("Dominant Vibe", "Positive 🟢" if avg_sentiment > 0 else "Negative 🔴")
                        
                        st.markdown("#### Linguistic Context (Associated Adjectives)")
                        col_adj1, col_adj2 = st.columns(2)
                        with col_adj1:
                            st.success("**Top Adjectives in Positive Contexts:**\n" + 
                                       ", ".join([f"*{adj}*" for adj, count in pos_adjectives.most_common(5)]) if pos_adjectives else "Not enough data.")
                        with col_adj2:
                            st.error("**Top Adjectives in Negative Contexts:**\n" + 
                                     ", ".join([f"*{adj}*" for adj, count in neg_adjectives.most_common(5)]) if neg_adjectives else "Not enough data.")
                        
                        # 2. Visual Insight: Filled Area Graph
                        df_char = pd.DataFrame({'Story Progress (%)': char_progress, 'Sentiment': char_sentiment})
                        df_char['Smoothed Trend'] = df_char['Sentiment'].rolling(window=max(2, len(char_sentiment)//8), min_periods=1).mean()

                        fig = go.Figure()
                        
                        # Add smooth filled area
                        fig.add_trace(go.Scatter(
                            x=df_char['Story Progress (%)'], y=df_char['Smoothed Trend'],
                            fill='tozeroy', mode='lines', line=dict(color='#7209b7', width=3),
                            name='Emotional Trend', fillcolor='rgba(114, 9, 183, 0.2)'
                        ))
                        
                        # Add individual scatter points
                        fig.add_trace(go.Scatter(
                            x=df_char['Story Progress (%)'], y=df_char['Sentiment'],
                            mode='markers', marker=dict(color='gray', size=5, opacity=0.5),
                            name='Individual Sentences'
                        ))

                        fig.add_hline(y=0, line_dash="dash", line_color="black")
                        fig.update_layout(title=f"The Emotional Journey of {selected_char}",
                                          xaxis_title="Story Progress (%)", yaxis_title="Sentiment Score",
                                          template="plotly_white", yaxis_range=[-1.1, 1.1])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"Not enough data points found to profile {selected_char}.")
        else:
            st.info("💡 Please generate the 'Character Network' in Tab 1 first to identify the main characters.")

    # ==========================================
    # TAB 4: THEMATIC ANALYSIS (NOUNS ONLY)
    # ==========================================
    with tab4:
        st.header("Thematic Analysis")
        st.markdown("Discover core themes by exclusively extracting **Nouns**. This removes descriptive noise and focuses strictly on the subjects, objects, and concepts of the story.")
        
        if st.button("Analyze Themes", key="btn_themes"):
            with st.spinner("Extracting nouns..."):
                doc = nlp(analyzed_text[:1000000])
                lemmas = Counter()
                
                # OPTIMIZATION: Strictly limited to NOUNS
                allowed_pos = {"NOUN"}
                
                for token in doc:
                    if token.pos_ in allowed_pos and not token.is_stop and token.is_alpha and len(token.text) > 2:
                        lemmas[token.lemma_.lower()] += 1
                
                if lemmas:
                    top_20 = lemmas.most_common(20)
                    df_themes = pd.DataFrame(top_20, columns=["Noun", "Frequency"])
                    
                    fig = px.bar(df_themes, x='Noun', y='Frequency', title="Top 20 Thematic Nouns",
                                 color='Frequency', color_continuous_scale='Viridis', template="plotly_white")
                    fig.update_layout(xaxis_title="Theme (Noun Lemma)", yaxis_title="Frequency Count", showlegend=False)
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not extract enough data for thematic analysis.")

    # ==========================================
    # TAB 5: LINGUISTIC STYLE & REGISTER
    # ==========================================
    with tab5:
        st.header("Linguistic Style & Register")
        st.markdown("Analyze the author's syntactic choices and text complexity.")
        
        if st.button("Calculate Style Metrics", key="btn_style"):
            with st.spinner("Parsing syntax and calculating readability..."):
                doc = nlp(analyzed_text[:500000]) 
                
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
                            if token.pos_ == "ADJ": adj_count += 1
                            elif token.pos_ == "VERB": verb_count += 1
                
                if total_sentences > 0 and total_words > 0:
                    total_action_desc = adj_count + verb_count
                    if total_action_desc > 0:
                        adj_ratio = (adj_count / total_action_desc) * 100
                        verb_ratio = (verb_count / total_action_desc) * 100
                    else:
                        adj_ratio = verb_ratio = 0
                        
                    flesch_score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_style = go.Figure(go.Indicator(
                            mode="gauge+number", value=verb_ratio,
                            title={'text': "Style: Descriptive vs Action", 'font': {'size': 18}},
                            number={'suffix': "% Verbs"},
                            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#3a0ca3"},
                                   'steps': [{'range': [0, 45], 'color': "#4cc9f0"},
                                             {'range': [45, 55], 'color': "#e9ecef"},
                                             {'range': [55, 100], 'color': "#f72585"}]}
                        ))
                        st.plotly_chart(fig_style, use_container_width=True)
                        st.caption("Blue = More Adjectives (Poetic) | Pink = More Verbs (Action-driven)")

                    with col2:
                        if flesch_score > 70: register = "Conversational"
                        elif flesch_score > 50: register = "Standard/Intermediate"
                        elif flesch_score > 30: register = "Formal/Complex"
                        else: register = "Academic/Difficult"
                        
                        fig_read = go.Figure(go.Indicator(
                            mode="gauge+number", value=max(0, min(flesch_score, 100)),
                            title={'text': f"Flesch-Kincaid Score<br><span style='font-size:0.8em;color:gray'>Register: {register}</span>", 'font': {'size': 18}},
                            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "black"},
                                   'steps': [{'range': [0, 30], 'color': "#d00000"},
                                             {'range': [30, 60], 'color': "#ffba08"},
                                             {'range': [60, 100], 'color': "#3f88c5"}]}
                        ))
                        st.plotly_chart(fig_read, use_container_width=True)
                        st.caption("Lower Score = More Complex/Academic | Higher Score = Easier to Read")

    # ==========================================
    # TAB 6: AI REFLECTION
    # ==========================================
    with tab6:
        st.header("Critical AI Evaluation")
        st.markdown("Evaluate the limitations of AI when dealing with figurative language, irony, or subtext.")
        
        if st.button("Find Extreme Sentiment Fragments", key="btn_ai"):
            with st.spinner("Scanning for extreme emotional polarity..."):
                scored_sentences = []
                for s in sentences:
                    clean_s = s.strip().replace('\n', ' ')
                    if len(clean_s.split()) > 8: 
                        score = sia.polarity_scores(clean_s)['compound']
                        scored_sentences.append({'text': clean_s, 'score': score})
                
                scored_sentences.sort(key=lambda x: x['score'])
                top_3_negative = scored_sentences[:3]
                top_3_positive = scored_sentences[-3:]
                top_3_positive.reverse()
                
                st.markdown("### AI's Most Extreme Classifications")
                col_neg, col_pos = st.columns(2)
                
                with col_neg:
                    st.error("#### 🔴 Top 3 Negative Fragments")
                    for item in top_3_negative:
                        st.markdown(f"""
                        <div style="background-color:#ffe5d9;padding:10px;border-radius:5px;margin-bottom:10px;">
                            <strong>Score: {item['score']:.2f}</strong><br><em>"{item['text']}"</em>
                        </div>""", unsafe_allow_html=True)
                        
                with col_pos:
                    st.success("#### 🟢 Top 3 Positive Fragments")
                    for item in top_3_positive:
                        st.markdown(f"""
                        <div style="background-color:#d8f3dc;padding:10px;border-radius:5px;margin-bottom:10px;">
                            <strong>Score: {item['score']:.2f}</strong><br><em>"{item['text']}"</em>
                        </div>""", unsafe_allow_html=True)
                    
                st.markdown("---")
                st.subheader("Student Reflection Task")
                st.markdown("""
                *Read the fragments above carefully. Log your thoughts in your research journal:*
                1. **Irony & Sarcasm:** Did the AI label an ironic or sarcastic sentence as genuinely positive/negative?
                2. **Figurative Language:** Did the AI misinterpret a metaphor (e.g., "killing it") literally?
                3. **Contextual Nuance:** Is the emotion of the fragment different when placed in the broader context of the story?
                """)

else:
    st.info("Please upload a PDF document in the sidebar to begin your literary analysis.")
