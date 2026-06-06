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
        
    nlp.max_length = 2000000
    sia = SentimentIntensityAnalyzer()
    return nlp, sia

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
    
    # Create Tabs (Now exactly 5)
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
        st.success("**Guiding Questions for Students:**\n- Where does the author use short, punchy sentences? Does this correlate with action sequences or dialogue?\n- Where are the peaks? Do these long sentences indicate heavy description, philosophical thoughts, or a 'stream of consciousness'?\n- Can you identify a rhythm or pattern in the author's writing style?")

        if st.button("Run Style Analysis"):
            with st.spinner("Analyzing syntax..."):
                doc = nlp(text_data[:500000])
                sentences = list(doc.sents)
                sent_lengths = [len([t for t in s if not t.is_punct]) for s in sentences if len(s) > 2]
                avg_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0
                
                st.metric("Average Sentence Length", f"{avg_len:.1f} words")
                
                window = 30
                smoothed = [sum(sent_lengths[i:i+window])/window for i in range(len(sent_lengths)-window+1)]
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(smoothed, color='#2b8cbe', linewidth=2)
                ax.fill_between(range(len(smoothed)), smoothed, color='#2b8cbe', alpha=0.2)
                ax.set_title("Sentence Length Over Time", fontsize=14, fontweight='bold')
                ax.set_ylabel("Words per Sentence")
                ax.set_xlabel("Story Progression (Sentences)")
                sns.despine()
                st.pyplot(fig)

    # ==========================================
    # TAB 2: VONNEGUT ARC
    # ==========================================
    with tab2:
        st.header("The Vonnegut Emotion Arc")
        
        st.info("**Legend:** The Y-axis represents the emotional tone (-1 is extreme negativity/tragedy, +1 is extreme positivity/joy). The X-axis represents the progression of the book from start to finish.")
        st.success("**Guiding Questions for Students:**\n- Where is the lowest point (the darkest moment) of the story? What happens in the plot at this exact percentage?\n- Does the story end on a high note (comedy/resolution) or a low note (tragedy)?\n- According to Kurt Vonnegut, stories have emotional 'shapes' (like 'Man in Hole' or 'Boy Meets Girl'). What shape is this novel?")

        if st.button("Run Emotion Arc"):
            with st.spinner("Calculating sentiment..."):
                sentences = re.split(r'(?<=[.!?]) +', text_data)
                scores = [sia.polarity_scores(s)['compound'] for s in sentences if len(s) > 10]
                
                window = max(1, len(scores) // 20)
                smoothed = [sum(scores[i:i+window])/window for i in range(len(scores)-window+1)]
                x_perc = [(i / len(smoothed)) * 100 for i in range(len(smoothed))]
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(x_perc, smoothed, color='#8856a7', linewidth=2.5)
                ax.fill_between(x_perc, smoothed, 0, where=(pd.Series(smoothed) > 0) if 'pd' in locals() else [s > 0 for s in smoothed], color='green', alpha=0.2, interpolate=True)
                ax.fill_between(x_perc, smoothed, 0, where=(pd.Series(smoothed) < 0) if 'pd' in locals() else [s < 0 for s in smoothed], color='red', alpha=0.2, interpolate=True)
                
                ax.axhline(0, color='black', linewidth=1, linestyle='--')
                ax.set_title("Narrative Sentiment Arc", fontsize=14, fontweight='bold')
                ax.set_xlabel("Story Progress (%)")
                ax.set_ylabel("Sentiment Score")
                sns.despine()
                st.pyplot(fig)

    # ==========================================
    # TAB 3: SOCIAL NETWORK (UPGRADED DESIGN)
    # ==========================================
    with tab3:
        st.header("Social Web & Relationship Dynamics")
        
        st.info("**Legend:**\n- **Node Size:** How often a character is mentioned.\n- **Edge Thickness:** How often two characters interact.\n- **Edge Color:** **Green** = Positive interaction (alliance/affection). **Red** = Negative interaction (conflict/tension). **Grey** = Neutral.")
        st.success("**Guiding Questions for Students:**\n- Who is the central 'hub' of the novel? Are there characters isolated on the edges?\n- Look at the red lines: which characters drive the central conflict of the story?\n- Look at the green lines: where are the alliances or romances? Do these align with your reading experience?")

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

                top_chars = [c for c, count in char_counts.most_common(15)]
                G = nx.Graph()
                
                for char in top_chars:
                    G.add_node(char, size=char_counts[char]*60)
                    
                for (c1, c2), data in interactions.items():
                    if c1 in top_chars and c2 in top_chars:
                        avg_sent = sum(data['sentiment']) / len(data['sentiment'])
                        if avg_sent > 0.1: color = '#2ca02c' # Green
                        elif avg_sent < -0.1: color = '#d62728' # Red
                        else: color = '#b0b0b0' # Grey
                        G.add_edge(c1, c2, weight=data['weight'], color=color)

                fig, ax = plt.subplots(figsize=(12, 9))
                # Better layout algorithm for spacing
                pos = nx.kamada_kawai_layout(G)
                
                sizes = [nx.get_node_attributes(G, 'size')[n] for n in G.nodes()]
                edge_colors = [nx.get_edge_attributes(G, 'color')[e] for e in G.edges()]
                # Scale weights for better visual thickness
                weights = [nx.get_edge_attributes(G, 'weight')[e] for e in G.edges()]
                max_w = max(weights) if weights else 1
                scaled_weights = [(w/max_w)*5 + 1 for w in weights]
                
                # Draw edges with a sleek curve
                nx.draw_networkx_edges(G, pos, width=scaled_weights, edge_color=edge_colors, 
                                       alpha=0.6, connectionstyle="arc3,rad=0.1", ax=ax)
                
                # Draw nodes with borders
                nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='#a6bddb', 
                                       edgecolors='black', linewidths=1.5, ax=ax)
                
                # Draw labels with background boxes for readability
                nx.draw_networkx_labels(G, pos, font_size=11, font_family='sans-serif', font_weight='bold',
                                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1), ax=ax)
                
                ax.set_title("Character Interaction Network", fontsize=16, fontweight='bold')
                plt.axis('off')
                st.pyplot(fig)

    # ==========================================
    # TAB 4: GENDER BIAS
    # ==========================================
    with tab4:
        st.header("Gender-Bias Agency Scanner")
        
        st.info("**Legend:** Displays the top 10 verbs directly associated with male pronouns (he/him) versus female pronouns (she/her).")
        st.success("**Guiding Questions for Students:**\n- Are male characters assigned more active, aggressive, or physical verbs? \n- Are female characters associated with passive, emotional, or reactive verbs?\n- How does the grammar of the book reflect the societal norms or gender biases of the time it was written?")

        if st.button("Run Gender Analysis"):
            with st.spinner("Analyzing pronouns and verbs..."):
                doc = nlp(text_data[:500000])
                m_verbs, f_verbs = Counter(), Counter()
                m_terms = {'he', 'him', 'his'}
                f_terms = {'she', 'her', 'hers'}
                stop_v = {'be', 'have', 'do', 'go', 'get', 'know', 'think', 'say', 'see', 'look', 'come', 'tell', 'ask', 'seem'}

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
                    sns.barplot(x=list(c), y=list(v), ax=ax1, color='#3182bd')
                    ax1.set_title("Top Male Actions (He/Him)", fontsize=12, fontweight='bold')
                    ax1.set_xlabel("Frequency")
                
                if f_verbs:
                    v, c = zip(*f_verbs.most_common(10))
                    sns.barplot(x=list(c), y=list(v), ax=ax2, color='#e6550d')
                    ax2.set_title("Top Female Actions (She/Her)", fontsize=12, fontweight='bold')
                    ax2.set_xlabel("Frequency")
                
                sns.despine()
                plt.tight_layout()
                st.pyplot(fig)

    # ==========================================
    # TAB 5: COLOR PALETTE
    # ==========================================
    with tab5:
        st.header("The Aesthetic Color Palette")
        
        st.info("**Legend:** A proportional visual breakdown of how often specific colors are explicitly mentioned in the text.")
        st.success("**Guiding Questions for Students:**\n- What is the dominant aesthetic or atmosphere of the text based on these colors?\n- Do certain colors carry symbolic meaning in this novel? (e.g., Green in *The Great Gatsby*, Red in *The Handmaid's Tale*)\n- How would a movie adaptation look based strictly on this data?")

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
                
                if found_colors:
                    total = sum(found_colors.values())
                    fig, ax = plt.subplots(figsize=(10, 2))
                    left = 0
                    for c_name, count in found_colors.most_common():
                        width = count / total
                        ax.barh(0, width, left=left, color=colors[c_name], edgecolor='black', linewidth=1.5)
                        # Add percentage label inside the bar if it's wide enough
                        if width > 0.05:
                            ax.text(left + width/2, 0, f"{c_name}\n{int(width*100)}%", 
                                    ha='center', va='center', color='white' if c_name not in ['white', 'yellow', 'gold'] else 'black',
                                    fontweight='bold', fontsize=10)
                        left += width
                        
                    ax.set_yticks([])
                    ax.set_xticks([])
                    ax.set_title("Novel Aesthetic Proportion", fontsize=14, fontweight='bold')
                    sns.despine(left=True, bottom=True)
                    st.pyplot(fig)
                else:
                    st.write("No strong color palette detected in this excerpt.")

else:
    st.info("Please upload a PDF file in the sidebar to begin.")
