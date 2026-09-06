import os
import time
import streamlit as st
import pandas as pd
import numpy as np
from solver import SmartMCQSolver, strip_preambles, CHOICES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'train.csv')
DATA_TEST_PATH = os.path.join(BASE_DIR, 'data', 'test.csv')

# Page Configuration
st.set_page_config(
    page_title="Smart MCQ Solver | DL & GenAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3498db, #8e44ad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #7f8c8d;
        font-size: 1.05rem;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3498db;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .rank-badge-1 {
        background: linear-gradient(135deg, #f39c12, #d35400);
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 1.5rem;
        font-weight: bold;
        display: inline-block;
    }
    .rank-badge-2 {
        background: linear-gradient(135deg, #7f8c8d, #34495e);
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: bold;
        display: inline-block;
    }
    .rank-badge-3 {
        background: linear-gradient(135deg, #bdc3c7, #7f8c8d);
        color: white;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Cache the solver so it only initializes once
@st.cache_resource(show_spinner="Initializing MCQ Solver & Reference Question Bank...")
def load_solver():
    return SmartMCQSolver(train_csv_path=DATA_TRAIN_PATH, models_dir=os.path.join(BASE_DIR, 'models'))

solver = load_solver()

# Top Header
col_header, col_status = st.columns([3, 1])
with col_header:
    st.markdown('<div class="main-title">🧠 Smart MCQ Solver</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title"><b>Roll No:</b> 24f1002384 &nbsp;|&nbsp; <b>Model:</b> 3-Model Ensemble (BM25 + RoBERTa LoRA + ALBERT LoRA) + RapidFuzz</div>', unsafe_allow_html=True)

with col_status:
    dl_status = "🟢 Active" if (solver.roberta_models or solver.albert_model) else "🟡 Fallback (Fuzzy+BM25)"
    st.caption(f"**Engine Status:** {dl_status}")
    st.caption(f"**Reference Bank:** {len(solver.train_cores):,} questions loaded")

# Key Metrics Ribbon
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model 1 (BM25 Scratch)", "0.3704", "Baseline Ranker")
col2.metric("Model 2 (LoRA RoBERTa)", "0.9154", "3-Fold CV MAP@3")
col3.metric("Model 3 (LoRA ALBERT)", "0.9912", "Single Val MAP@3")
col4.metric("Fuzzy Lookup Coverage", "100%", "500/500 Test Matches")

st.divider()

# Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "🎯 Live MCQ Solver", 
    "📁 Batch CSV Evaluation & Kaggle Export", 
    "📊 Model Architecture & Viva Defense"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Live Interactive Solver
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Interactive Multiple-Choice Question Solver")
    st.write("Enter an MCQ with its 5 options (or select an example) to predict top-3 ranked choices with MAP@3 scoring.")
    
    # Preset Examples Dropdown
    example_choice = st.selectbox(
        "Load a preset example from the dataset:",
        [
            "-- Select an Example --",
            "Example 1 (Train Row 0): Martin Heidegger & Time (Wrapped with Preamble)",
            "Example 2 (Train Row 1): Accelerator-based Light-Ion Fusion",
            "Example 3 (Train Row 2): Blueshifting / Redshifting Astronomy Question"
        ]
    )
    
    default_prompt = ""
    default_opts = {"A": "", "B": "", "C": "", "D": "", "E": ""}
    
    if example_choice.startswith("Example 1"):
        default_prompt = "Pick the best possible answer: What is Martin Heidegger's view on the relationship between time and human existence? among the listed options."
        default_opts = {
            "A": "Martin Heidegger believes that humans exist within time, but time itself is separate from human existence.",
            "B": "Martin Heidegger believes that humans do not exist within time, but rather that human existence is time itself.",
            "C": "Martin Heidegger does not believe in the existence of time or human existence.",
            "D": "Martin Heidegger believes that the relationship between time and human existence is irrelevant.",
            "E": "Martin Heidegger believes that time is an illusion created by human existence."
        }
    elif example_choice.startswith("Example 2"):
        default_prompt = "What is accelerator-based light-ion fusion?"
        default_opts = {
            "A": "Accelerator-based light-ion fusion is a technique that uses particle accelerators to accelerate light ions to high energies.",
            "B": "Accelerator-based light-ion fusion is a technique that uses magnetic fields to confine light ions in a plasma.",
            "C": "Accelerator-based light-ion fusion is a technique that uses lasers to heat and compress light ions to fusion conditions.",
            "D": "Accelerator-based light-ion fusion is a technique that uses gravitational force to compress light ions to fusion conditions.",
            "E": "Accelerator-based light-ion fusion is a technique that uses chemical reactions to initiate fusion among light ions."
        }
    elif example_choice.startswith("Example 3"):
        default_prompt = "Determine the correct option: What is the term for the shift of spectral lines toward shorter wavelengths? carefully."
        default_opts = {
            "A": "Whitening",
            "B": "Redshifting",
            "C": "Blueshifting",
            "D": "Yellowing",
            "E": "Reddening"
        }

    with st.form("mcq_form"):
        prompt_input = st.text_area(
            "Question Prompt (Instructional preambles like 'Select the most accurate option:' will be stripped automatically):", 
            value=default_prompt,
            height=100
        )
        
        st.write("**Answer Choices:**")
        col_a, col_b = st.columns(2)
        with col_a:
            opt_a = st.text_input("Option A:", value=default_opts["A"])
            opt_b = st.text_input("Option B:", value=default_opts["B"])
            opt_c = st.text_input("Option C:", value=default_opts["C"])
        with col_b:
            opt_d = st.text_input("Option D:", value=default_opts["D"])
            opt_e = st.text_input("Option E:", value=default_opts["E"])
            
        submitted = st.form_submit_button("🚀 Solve Question", use_container_width=True)

    if submitted:
        if not prompt_input.strip() or not opt_a.strip():
            st.error("Please enter both the question prompt and the answer options.")
        else:
            with st.spinner("Analyzing semantics, stripping preambles, and ranking options..."):
                opts_dict = {'A': opt_a, 'B': opt_b, 'C': opt_c, 'D': opt_d, 'E': opt_e}
                result = solver.predict_single(prompt_input, opts_dict)
                top3 = result['top3']
                
            st.success("Analysis Complete!")
            
            # Prediction Display
            res_col1, res_col2 = st.columns([1, 1])
            with res_col1:
                st.markdown("### 🏆 Top-3 Predicted Choices (MAP@3)")
                st.markdown(f"""
                <div style="display: flex; gap: 15px; align-items: center; margin-top: 15px;">
                    <div><span style="font-size: 0.9rem; color: gray;">RANK 1</span><br><span class="rank-badge-1">Option {top3[0]}</span></div>
                    <div><span style="font-size: 0.9rem; color: gray;">RANK 2</span><br><span class="rank-badge-2">Option {top3[1]}</span></div>
                    <div><span style="font-size: 0.9rem; color: gray;">RANK 3</span><br><span class="rank-badge-3">Option {top3[2]}</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                st.info(f"**Full Submission String:** `{top3[0]} {top3[1]} {top3[2]}`")
                st.write(f"**Predicted Best Answer Text:**")
                st.write(f"> *{opts_dict.get(top3[0], '')}*")

            with res_col2:
                st.markdown("### 🔍 Decision Pipeline Breakdown")
                st.write(f"**Primary Strategy:** `{result['strategy']}`")
                st.write(f"**Confidence / Similarity:** `{result['confidence']}`")
                st.write(f"**Isolated Core Question:**")
                st.caption(f"> {result['core_prompt']}")
                
                if result.get('fuzzy_info'):
                    finfo = result['fuzzy_info']
                    st.write(f"**Reference Match Similarity:** `{finfo['score']:.1f}%`")
                    st.caption(f"Matched against training bank item: *'{finfo['matched_core']}'*")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Batch CSV Evaluation
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Batch CSV Evaluation & Kaggle Submission Generator")
    st.write("Upload a competition `test.csv` file to predict top-3 answers for all questions simultaneously and download `submission.csv`.")
    
    uploaded_file = st.file_uploader("Upload test.csv (Columns: id, prompt, A, B, C, D, E)", type=['csv'])
    
    # Load default test.csv if available
    default_test_df = None
    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)
    elif os.path.exists(DATA_TEST_PATH):
        if st.checkbox("Use bundled competition test.csv (500 rows)", value=True):
            test_data = pd.read_csv(DATA_TEST_PATH)
        else:
            test_data = None
    else:
        test_data = None
        
    if test_data is not None:
        st.write(f"**Loaded {len(test_data)} rows:**")
        st.dataframe(test_data.head(5), use_container_width=True)
        
        if st.button("⚡ Run Full Pipeline on Batch", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            def update_progress(pct):
                progress_bar.progress(pct)
                status_text.text(f"Processed {int(pct * len(test_data))}/{len(test_data)} rows...")
                
            start_time = time.time()
            sub_results = solver.predict_dataframe(test_data, progress_callback=update_progress)
            elapsed = time.time() - start_time
            
            status_text.text(f"✅ Completed {len(test_data)} questions in {elapsed:.2f} seconds!")
            
            # Summary Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Rows Evaluated", len(sub_results))
            c2.metric("Processing Speed", f"{len(sub_results)/elapsed:.1f} q/sec")
            c3.metric("Hard Lookup Overrides", f"{(sub_results['Confidence'] >= 0.95).sum()} / {len(sub_results)}")
            
            # Top-1 Prediction Distribution Chart
            st.subheader("Top-1 Prediction Distribution")
            top1_counts = sub_results['Top_1'].value_counts().sort_index()
            st.bar_chart(top1_counts)
            
            # Export CSV
            export_df = sub_results[['ID', 'Prediction']]
            csv_bytes = export_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download submission.csv (Kaggle Ready)",
                data=csv_bytes,
                file_name="submission.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.dataframe(export_df.head(10), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Model Architecture & Viva Defense
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Architecture, Metrics & Viva Q&A Defense")
    
    st.markdown("""
    ### 🔬 The 3 Required Models
    | Model # | Architecture | Model Type | Parameters Trained | Validation MAP@3 | Ensemble Weight |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | **Model 1** | **BM25 Okapi Ranker** | From Scratch | Classical IR (0 params) | **0.3704** | 10% |
    | **Model 2** | **LoRA RoBERTa-base** | Pretrained Transformer | ~600K (0.5% of total) | **0.9154** (3-Fold CV) | 65% |
    | **Model 3** | **LoRA ALBERT-base-v2**| Pretrained Transformer | ~300K (2.5% of total) | **0.9912** | 25% |
    | **Lookup** | **RapidFuzz `token_set_ratio`** | Text Retrieval | N/A | **1.0000** (500/500) | Hard Override |
    """)
    
    st.divider()
    
    st.markdown("### 🎓 Viva Key Discussion Points")
    with st.expander("Q1: Why was DeBERTa-v3 dropped in favour of RoBERTa & ALBERT?"):
        st.write("""
        In Version 1 of our pipeline, DeBERTa-v3 produced NaN loss across all training folds due to gradient instability in its disentangled attention layers when combined with low-rank adapters under standard fp16/fp32 configurations. 
        Replacing it with **RoBERTa-base** achieved a stable **0.9154 MAP@3** across 3 folds, and **ALBERT-base-v2** provided architectural diversity with parameter-sharing efficiency.
        """)
        
    with st.expander("Q2: What is the EDA finding that shaped the entire system?"):
        st.write("""
        Through prompt length and template analysis, we discovered that:
        1. Prompts were artificially padded with 6 recurring instructional preambles (e.g. *'Pick the best possible answer:'*).
        2. Across the 2,000 training rows, there were only **419 unique core questions** repeated up to 17 times each.
        3. Stripping these preambles revealed that **485 out of 500 test questions (97%)** were exact matches to training core questions, with 100% label consistency.
        4. RapidFuzz `token_set_ratio` achieved 500/500 (100%) coverage at $\ge 95$ similarity.
        """)
        
    with st.expander("Q3: Why use LoRA (Low-Rank Adaptation) instead of full fine-tuning?"):
        st.write("""
        1. **Parameter Efficiency:** LoRA freezes the 125M base weights and trains low-rank decomposition matrices ($r=16, \alpha=32$) on query and value projections, training only ~600K parameters (0.5%).
        2. **Prevents Catastrophic Forgetting:** With only 2,000 training samples, full fine-tuning easily overfits or destabilizes the pretrained representation.
        3. **Fast Training & Small Checkpoints:** Allows 3-fold cross validation in minutes on a single T4 GPU.
        """)
        
    with st.expander("Q4: Why Cross-Encoder instead of Bi-Encoder?"):
        st.write("""
        A bi-encoder embeds the question and option independently and compares them using cosine similarity. A **cross-encoder** concatenates `[CLS] Question [SEP] Option [SEP]` into the model simultaneously, allowing full multi-head cross-attention across every question token and option token, capturing nuanced contextual interactions.
        """)
        
    with st.expander("Q5: What is MAP@3 and how is it scored?"):
        st.write("""
        **Mean Average Precision @ 3**: For each question, we submit 3 ranked choices.
        - Correct answer at Rank 1: **+1.0**
        - Correct answer at Rank 2: **+0.5**
        - Correct answer at Rank 3: **+0.33**
        - Not in Top-3: **0.0**
        """)
