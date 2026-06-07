import streamlit as st
import PyPDF2
import spacy
import networkx as nx
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

def truncate_text(text, max_words=100):
    """Beperkt fragmenten strikt tot een maximaal aantal woorden."""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + " [...]"
    return text

# --- SESSION STATE INITIALIZATION ---
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

    # Reset session state if text length changes
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
        st.markdown("Maps out which characters frequently interact. Node size reflects centrality. Uses a sliding window of 3 sentences to detect broader conversational contexts.")
        
        if st.button("Generate Character Network") or st.session_state['net_done']:
            st.session_state['net_done'] = True
            with st.spinner("Extracting entities, resolving names, and computing Plotly network..."):
                @st.cache_data
                def build_network_data(text):
                    doc = nlp(text[:1000000])
                    sents = list(doc.sents)
                    
                    raw_entities = []
                    for sent in sents:
                        sent_ents = [clean_entity_name(ent.text) for ent in sent.ents if ent.label_ == "PERSON" and len(ent.text.split()) < 4]
                        raw_entities.append([e for e in sent_ents if len(e) > 2 and e.istitle()])
                        
                    unique_names = set(e for sublist in raw_entities for e in sublist)
                    alias_map = {name: name for name in unique_names}
                    for name in unique_names:
                        if len(name.split()) == 1:
                            for long_name in unique_names:
                                if name in long_name.split() and len(long_name.split()) > 1:
                                    alias_map[name] = long_name
                                    break
                                    
                    char_counts = Counter()
                    interactions = Counter()
                    
                    for i in range(len(raw_entities)):
                        window_chars = set()
                        for j in range(i, min(i + 3, len(raw_entities))):
                            for char in raw_entities[j]:
                                resolved_char = alias_map[char]
                                window_chars.add(resolved_char)
                                if i == j: 
                                    char_counts[resolved_char] += 1
                                    
                        if len(window_chars) > 1:
                            for c1, c2 in itertools.combinations(sorted(window_chars), 2):
                                interactions[(c1, c2)] += 1
                                
                    return char_counts, interactions
                
                char_counts, interactions = build_network_data(analyzed_text)
                top_chars = [c for c, count in char_counts.most_common(20)]
                st.session_state['top_chars'] = top_chars 
                
                G = nx.Graph()
                for char in top_chars: G.add_node(char)
                for (c1, c2), weight in interactions.items():
                    if c1 in top_chars and c2 in top_chars: G.add_edge(c1, c2, weight=weight)
                        
                if len(G.nodes) > 0:
                    pos = nx.spring_layout(G, k=0.6, iterations=50, seed=42)
                    
                    edge_x = []
                    edge_y = []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                        
                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=0.8, color='#A9A9A9'),
                        hoverinfo='none',
                        mode='lines')
                        
                    node_x = []
                    node_y = []
                    node_text = []
                    node_size = []
                    node_color = []
                    
                    degrees = [G.degree(n, weight='weight') for n in G.nodes()]
                    max_deg = max(degrees) if degrees else 1
                    
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        node_text.append(node)
                        deg = G.degree(node, weight='weight')
                        node_size.append(max(15, (deg / max_deg) * 60))
                        node_color.append(deg)
                        
                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=node_text,
                        textposition="top center",
                        hovertext=[f"Interactions: {c}" for c in node_color],
                        hoverinfo="text",
                        marker=dict(
                            showscale=True,
                            colorscale='Viridis',
                            size=node_size,
                            color=node_color,
                            line=dict(width=2, color='white')
                        ))
                        
                    fig = go.Figure(data=[edge_trace, node_trace],
                                    layout=go.Layout(
                                        title='Interactive Character Network',
                                        showlegend=False,
                                        hovermode='closest',
                                        margin=dict(b=20,l=5,r=5,t=40),
                                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                        plot_bgcolor='white'
                                    ))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough character interactions found.")

    # ==========================================
    # TAB 2: NARRATIVE ARC
    # ==========================================
    with tab2:
        st.header("Global Narrative Sentiment Arc")
        st.markdown("Displays the emotional trajectory over **100 chronological segments**, utilizing a smoothing algorithm to reveal the macro-emotional structure.")
        
        if st.button("Generate Narrative Arc") or st.session_state['arc_done']:
            st.session_state['arc_done'] = True
            with st.spinner("Calculating timeline sentiment and applying smoothing..."):
                @st.cache_data
                def calculate_arc(text):
                    words = text.split()
                    chunk_size = max(1, len(words) // 100)
                    segments = []
                    for i in range(100):
                        start = i * chunk_size
                        end = (i + 1) * chunk_size if i < 99 else len(words)
                        chunk_text = " ".join(words[start:end])
                        score = sia.polarity_scores(chunk_text)['compound']
                        segments.append(score)
                    return segments
                
                segment_scores = calculate_arc(analyzed_text)
                
                df_arc = pd.DataFrame({
                    'Story Progress (%)': range(1, 101),
                    'Raw Sentiment': segment_scores
                })
                
                df_arc['Smoothed Narrative Arc'] = df_arc['Raw Sentiment'].rolling(window=8, min_periods=1).mean()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_arc['Story Progress (%)'], y=df_arc['Smoothed Narrative Arc'],
                                         mode='lines', line=dict(color='#ef476f', width=4),
                                         name='Narrative Arc'))
                fig.add_trace(go.Scatter(x=df_arc['Story Progress (%)'], y=df_arc['Raw Sentiment'],
                                         mode='lines', line=dict(color='gray', width=1, dash='dot'),
                                         opacity=0.4, name='Micro-Sentiment (Raw)'))
                
                fig.update_layout(title="Smoothed Emotional Progression", 
                                  yaxis_title="Sentiment (Compound Score)", 
                                  xaxis_title="Story Progress (%)",
                                  template="plotly_white")
                fig.add_hline(y=0, line_dash="dash", line_color="black")
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

                        st.markdown("---")
                        st.subheader("🔍 Context Explorer")
                        st.markdown(f"Select a specific part of the story to read the exact sentences involving **{selected_char}**.")
                        
                        prog_min, prog_max = st.slider("Select Story Progress Range (%)", 0, 100, (0, 100))
                        
                        filtered_sentences = [(p, t, s) for p, t, s in zip(char_progress, [s[0] for s in char_sentences], char_sentiment) 
                                              if prog_min <= p <= prog_max]
                        
                        if filtered_sentences:
                            st.write(f"Found **{len(filtered_sentences)}** occurrences in this range. *(Displaying max 25, capped at 100 words)*")
                            for p, txt, sc in filtered_sentences[:25]:
                                color = "#d8f3dc" if sc > 0.3 else ("#ffe5d9" if sc < -0.3 else "#f8f9fa")
                                txt_disp = truncate_text(txt, 100)
                                st.markdown(f"""
                                <div style='background-color:{color}; padding:10px; border-radius:5px; margin-bottom:8px; border-left: 4px solid gray;'>
                                    <small style='color:gray;'>Progress: {p:.1f}% | Emotion Score: {sc:.2f}</small><br>
                                    {txt_disp}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No mentions found in this specific range.")
                    else:
                        st.warning(f"Not enough data points found to profile {selected_char}.")
        else:
            st.info("💡 Please generate the 'Character Network' in Tab 1 first to identify the main characters.")

    # ==========================================
    # TAB 4: THEMATIC ANALYSIS & CONTEXT (Refactored)
    # ==========================================
    with tab4:
        st.header("Advanced Thematic Density & Semantic Anchors")
        st.markdown("In de literatuurwetenschap weerspiegelen zelfstandige naamwoorden (*nouns*) de materiële, psychologische of filosofische bouwstenen van de fictieve wereld. Deze module isoleert de top 20 thematische kernwoorden.")
        
        if st.button("Analyze Themes") or st.session_state['themes_done']:
            st.session_state['themes_done'] = True
            with st.spinner("Extracting nouns and filtering lexical noise..."):
                @st.cache_data
                def extract_themes(text):
                    doc = nlp(text[:1000000])
                    lemmas = Counter()
                    for token in doc:
                        # Filter out general noise, short tokens, and strictly fetch common nouns
                        if token.pos_ == "NOUN" and not token.is_stop and token.is_alpha and len(token.text) > 2:
                            lemmas[token.lemma_.lower()] += 1
                    return lemmas.most_common(20)

                top_20 = extract_themes(analyzed_text)
                
                if top_20:
                    df_themes = pd.DataFrame(top_20, columns=["Noun", "Frequency"])
                    fig = px.bar(df_themes, x='Noun', y='Frequency', title="Top 20 Thematic Nouns (Lexical Density)",
                                 color='Frequency', color_continuous_scale='Plasma', template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- CRITICAL VISUAL DESCRIPTION ---
                    with st.expander("📊 Hoe interpreteer ik deze grafiek? (Literair-wetenschappelijke gids)", expanded=True):
                        st.markdown("""
                        **Wat zie je?** Deze staafgrafiek toont de absolute frequentie van zelfstandige naamwoorden. 
                        
                        **Literaire Analyse:**
                        - **Materiële vs. Abstracte Wereld:** Domineren er concrete woorden zoals *house, money, letter, carriage*? Dit wijst vaak op een realistische of sociaal-economische focus (denk aan Jane Austen of Charles Dickens). Domineren abstracte zelfstandige naamwoorden zoals *thought, fear, love, soul*? Dan ligt de nadruk op de psychologische of filosofische binnenwereld van de personages.
                        - **Thematische Anchors:** Woorden die uitzonderlijk hoog scoren, fungeren vaak als motieven of leidmotieven doorheen het werk.
                        """)
                    
                    st.markdown("---")
                    st.subheader("🔍 Context Explorer: Close-Reading van Thema's")
                    st.markdown("Kies een thematisch kernwoord om de exacte zinnen te isoleren en de context te analyseren.")
                    
                    theme_words = [n for n, c in top_20]
                    selected_theme = st.selectbox("Select a noun to explore context:", theme_words)
                    
                    if selected_theme:
                        matches = [s for s in sentences if re.search(rf'\b{re.escape(selected_theme)}\b', s, re.IGNORECASE)]
                        st.write(f"Gevonden: **{len(matches)}** zinnen met het woord '*{selected_theme}*'. *(Max. 25 getoond, gecapped op 100 woorden)*")
                        
                        for m in matches[:25]:
                            m_trunc = truncate_text(m, 100)
                            # Highlight the keyword for high readability
                            highlighted = re.sub(rf'(\b{re.escape(selected_theme)}\b)', r'**\1**', m_trunc, flags=re.IGNORECASE)
                            st.markdown(f"- {highlighted}")
                else:
                    st.warning("Could not extract enough data for thematic analysis.")

    # ==========================================
    # TAB 5: LINGUISTIC STYLE & REGISTER (Refactored)
    # ==========================================
    with tab5:
        st.header("Linguistic Register & Stylistic Metrics")
        st.markdown("Stijlanalyse (stylometrie) maakt het mogelijk om de syntactische complexiteit en de lexicale variëteit van een auteur objectief te meten.")
        
        if st.button("Calculate Style Metrics") or st.session_state['style_done']:
            st.session_state['style_done'] = True
            with st.spinner("Parsing syntax, counting syllables, and computing lexical richness..."):
                @st.cache_data
                def calc_style(text):
                    doc = nlp(text[:500000]) 
                    tot_sents = tot_words = tot_syllables = adj_c = verb_c = 0
                    unique_words = set()
                    
                    for sent in doc.sents:
                        tot_sents += 1
                        for token in sent:
                            if token.is_alpha:
                                tot_words += 1
                                unique_words.add(token.text.lower())
                                tot_syllables += count_syllables(token.text)
                                if token.pos_ == "ADJ" or token.pos_ == "ADV": adj_c += 1
                                elif token.pos_ == "VERB": verb_c += 1
                                
                    # Calculate Type-Token Ratio (Lexical Diversity) based on a sample to avoid length bias
                    ttr = (len(unique_words) / tot_words * 100) if tot_words > 0 else 0
                    return tot_sents, tot_words, tot_syllables, adj_c, verb_c, ttr
                
                tot_sents, tot_words, tot_syllables, adj_count, verb_count, ttr_score = calc_style(analyzed_text)
                
                if tot_sents > 0 and tot_words > 0:
                    total_action_desc = adj_count + verb_count
                    verb_ratio = (verb_count / total_action_desc) * 100 if total_action_desc > 0 else 0
                    
                    # Flesch Reading Ease Formula (Adapted for English text evaluation)
                    flesch_score = 206.835 - 1.015 * (tot_words / tot_sents) - 84.6 * (tot_syllables / tot_words)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        fig_style = go.Figure(go.Indicator(
                            mode="gauge+number", value=verb_ratio, title={'text': "Modificatiebalans<br><span style='font-size:0.8em;color:gray'>Beschrijvend vs. Actie</span>", 'font': {'size': 16}},
                            number={'suffix': "% Verbs"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#3a0ca3"},
                                   'steps': [{'range': [0, 45], 'color': "#4cc9f0"}, {'range': [45, 55], 'color': "#e9ecef"}, {'range': [55, 100], 'color': "#f72585"}]}
                        ))
                        st.plotly_chart(fig_style, use_container_width=True)

                    with col2:
                        register = "Conversational (Eenvoudig)" if flesch_score > 70 else "Standard Fiction" if flesch_score > 50 else "Complex Narrative" if flesch_score > 30 else "Academic / High-Literature"
                        fig_read = go.Figure(go.Indicator(
                            mode="gauge+number", value=max(0, min(flesch_score, 100)), title={'text': f"Flesch Reading Ease Score<br><span style='font-size:0.8em;color:gray'>Syntactisch Register: {register}</span>", 'font': {'size': 16}},
                            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "black"},
                                   'steps': [{'range': [0, 30], 'color': "#d00000"}, {'range': [30, 60], 'color': "#ffba08"}, {'range': [60, 100], 'color': "#3f88c5"}]}
                        ))
                        st.plotly_chart(fig_read, use_container_width=True)
                        
                    with col3:
                        fig_ttr = go.Figure(go.Indicator(
                            mode="gauge+number", value=ttr_score, title={'text': "Lexicale Diversiteit (TTR)<br><span style='font-size:0.8em;color:gray'>Woordenschat-rijkdom</span>", 'font': {'size': 16}},
                            number={'suffix': "%"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#2a9d8f"},
                                   'steps': [{'range': [0, 20], 'color': "#f1faee"}, {'range': [20, 50], 'color': "#a8dadc"}, {'range': [50, 100], 'color': "#457b9d"}]}
                        ))
                        st.plotly_chart(fig_ttr, use_container_width=True)

                    # --- CRITICAL VISUAL DESCRIPTION ---
                    with st.expander("📊 Hoe interpreteer ik deze meters? (Stylometrische gids)", expanded=True):
                        st.markdown(f"""
                        **1. Modificatiebalans:** Geeft de verhouding weer tussen werkwoorden (actie/dynamiek) en bijvoeglijke naamwoorden/bijwoorden (beschrijvend/statisch). Een zeer laag percentage betekent een overdaad aan adjectieven (bijv. Romantische literatuur of Gothic novels). Een hoog percentage wijst op een strakke, actiegerichte vertelstijl (bijv. Ernest Hemingway).
                        
                        **2. Flesch Reading Ease Score:** Meet de syntactische complexiteit op basis van zinslengte en lettergrepen per woord. Jouw huidige selectie scoort **{flesch_score:.1f}**. Hoe lager de score, hoe complexer en academischer de zinsconstructies (typisch voor victoriaanse literatuur).
                        
                        **3. Lexicale Diversiteit (Type-Token Ratio):** Dit meet het percentage unieke woorden ten opzichte van het totaal aantal woorden. Een hogere TTR wijst op een rijke, gevarieerde woordenschat en een complexe world-building.
                        """)

    # ==========================================
    # TAB 6: AI REFLECTION (Refactored)
    # ==========================================
    with tab6:
        st.header("Algoritmische Close-Reading & Distant-Reading Paradox")
        st.markdown("Sentimentanalyse is een krachtig hulpmiddel voor *distant reading*, maar kan subtiele literaire instrumenten zoals ironie, sarcasme of verborgen spanningen missen. Deze module confronteert je met de uitersten van het algoritme.")
        
        if st.button("Find Extreme Sentiment Fragments") or st.session_state['ai_done']:
            st.session_state['ai_done'] = True
            with st.spinner("Scanning data for semantic polarization..."):
                @st.cache_data
                def get_extremes(sents):
                    scored = []
                    for s in sents:
                        c_s = s.strip().replace('\n', ' ')
                        # Filter out sentences that are too short to contain semantic weight
                        if len(c_s.split()) > 10: 
                            scored.append({'text': c_s, 'score': sia.polarity_scores(c_s)['compound']})
                    scored.sort(key=lambda x: x['score'])
                    return scored[:3], scored[-3:][::-1]

                top_3_neg, top_3_pos = get_extremes(sentences)
                
                col_neg, col_pos = st.columns(2)
                with col_neg:
                    st.error("#### 🔴 Top 3 Meest Negatieve Fragmenten (Distant Reading)")
                    for idx, item in enumerate(top_3_neg): 
                        trunc_txt = truncate_text(item['text'], 100)
                        st.markdown(f"""
                        <div style='background:#ffe5d9; padding:12px; border-radius:5px; margin-bottom:10px; border-left: 5px solid #d00000;'>
                            <strong>Fragment {idx+1} | VADER Score: {item['score']:.2f}</strong><br>
                            <span style='font-style: italic;'>\"{trunc_txt}\"</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                with col_pos:
                    st.success("#### 🟢 Top 3 Meest Positieve Fragmenten (Distant Reading)")
                    for idx, item in enumerate(top_3_pos): 
                        trunc_txt = truncate_text(item['text'], 100)
                        st.markdown(f"""
                        <div style='background:#d8f3dc; padding:12px; border-radius:5px; margin-bottom:10px; border-left: 5px solid #2a9d8f;'>
                            <strong>Fragment {idx+1} | VADER Score: {item['score']:.2f}</strong><br>
                            <span style='font-style: italic;'>\"{trunc_txt}\"</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                # --- CRITICAL LITERARY EXPLANATION ---
                st.markdown("""
                <div style='background-color:#f8f9fa; padding:15px; border-radius:5px; border:1px solid #e9ecef; margin-top:15px;'>
                    <h5>🧐 Kritische Reflectieopdracht voor de Leerling:</h5>
                    <p>Bekijk de bovenstaande fragmenten die het computerprogramma als 'extreem positief' of 'extreem negatief' heeft geclassificeerd. 
                    Gebruik je menselijke vaardigheden (Close-Reading) om het algoritme te beoordelen:</p>
                    <ul>
                        <li>Klopt de emotie die de computer meet wel met de werkelijke, literaire betekenis?</li>
                        <li>Is er sprake van <strong>ironie</strong> of een personage dat liegt/beleefd probeert te zijn? (Bijvoorbeeld: een beleefde afwijzing kan door de computer als 'positief' worden gezien door woorden als <em>pleasure, honor, good</em>).</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    # ==========================================
    # INTERACTIVE AI PROMPT WORKSPACE (End of Code)
    # ==========================================
    st.markdown("---")
    st.subheader("🤖 Interactive AI Literary Assistant & Hypothesis Tester")
    st.markdown("Gebruik deze interactieve prompt-omgeving om je eigen literaire stellingen of vragen over de geüploade tekst te formuleren. Dit veld verwerkt je invoer veilig zonder de bovenstaande analyseresultaten te overschrijven.")

    user_hypothesis = st.text_input(
        "Formuleer je stelling of onderzoeksvraag (bijv: 'Is Darcy de echte antagonist van het verhaal?' of 'Hoe beïnvloedt de industriële revolutie het taalgebruik?'):",
        placeholder="Type hier je onderzoeksvraag..."
    )

    if user_hypothesis:
        st.markdown("#### 🛠️ Voorgesteld Onderzoeksplan op basis van jouw Dashboard-data:")
        
        # We parse the input keywords to dynamically tailor the guide without relying on an external LLM call, keeping code fully stable.
        guidelines = []
        if any(w in user_hypothesis.lower() for w in ["personage", "character", "darcy", "elizabeth", "antagonist", "wie"]):
            guidelines.append("- **Character Network (Tab 1):** Controleer de centraliteit en de directe connecties van dit personage. Heeft hij of zij een hoge interactiedichtheid met onverwachte figuren?")
            guidelines.append("- **Character Emotions (Tab 3):** Analyseer de baseline sentiment-score en de adjectieven die aan dit personage gekoppeld zijn. Matchen ze met je stelling?")
        if any(w in user_hypothesis.lower() for w in ["thema", "maatschappij", "revolutie", "geld", "money", "class", "social", "liefde"]):
            guidelines.append("- **Thematic Analysis (Tab 4):** Bekijk of verwante zelfstandige naamwoorden in de top 20 voorkomen en onderzoek de 'Context Explorer' om te zien in welke sociaal-culturele context ze vallen.")
        if any(w in user_hypothesis.lower() for w in ["taal", "stijl", "schrijfstijl", "moeilijk", "complex", "zinnen"]):
            guidelines.append("- **Style & Register (Tab 5):** Leg de Flesch Reading Ease score en de Modificatiebalans naast representatieve stijlperiodes (bijv. Romantiek vs. Realisme).")
            
        if not guidelines:
            guidelines.append("- **Narrative Arc (Tab 2):** Bekijk op welk punt in het verhaal (0-100%) de spanning stijgt of daalt en leg dit naast je hypothese.")
            guidelines.append("- **Close-Reading (Tab 6):** Analyseer of de uitersten van het algoritme jouw stelling ondersteunen of juist tegenspreken.")

        st.info("\n".join(guidelines))
        st.markdown(f"**Volgende stap voor de leerling:** Noteer je bevindingen in je leesdossier en gebruik minimaal 2 concrete tekstfragmenten (max. 100 woorden) uit de *Context Explorers* als bewijsvoering.")

else:
    st.info("Please upload a PDF document in the sidebar to begin your literary analysis.")
