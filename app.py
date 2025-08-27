import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import graphviz
import itertools
import google.generativeai as genai
import requests

# -----------------------------
# API Keys
genai.api_key = st.secrets.get("GENIE_KEY")
PERPLEXITY_KEY = st.secrets.get("PERPLEXITY_KEY")

# -----------------------------
# Gemini API Explanation
def get_gemini_explanation(verilog_code):
    try:
        response = genai.TextGeneration.create(
            model="gemini-1",
            prompt=f"Explain this Verilog code and generate a testbench:\n{verilog_code}",
            temperature=0.3,
            max_output_tokens=500
        )
        return response.text
    except Exception:
        st.warning("⚠️ Gemini API not available, using Perplexity API fallback.")
        return get_perplexity_explanation(verilog_code)

# -----------------------------
# Perplexity API fallback
def get_perplexity_explanation(verilog_code):
    headers = {"Authorization": f"Bearer {PERPLEXITY_KEY}"}
    data = {"query": f"Explain this Verilog code and generate testbench:\n{verilog_code}"}
    try:
        response = requests.post("https://api.perplexity.ai/ask", headers=headers, json=data)
        return response.json().get("answer", "Explanation not available.")
    except Exception:
        return "Explanation could not be generated due to API issues."

# -----------------------------
# Parse Verilog module (combinational)
def parse_verilog(verilog_code):
    inputs, outputs, assigns = [], [], []
    try:
        lines = verilog_code.splitlines()
        for line in lines:
            line = line.strip()
            if line.startswith("input"):
                inputs += [x.strip() for x in line.replace("input", "").replace(";", "").split(",")]
            elif line.startswith("output"):
                outputs += [x.strip() for x in line.replace("output", "").replace(";", "").split(",")]
            elif line.startswith("assign"):
                assigns.append(line.replace("assign", "").replace(";", "").strip())
        return inputs, outputs, assigns
    except Exception:
        return [], [], []

# -----------------------------
# Generate truth table and waveform
def generate_truth_table(inputs, outputs, assigns):
    n = len(inputs)
    if n == 0 or len(assigns) == 0:
        return None, None, None

    waveform = {var: [] for var in inputs + outputs}
    table_rows = []

    for combination in itertools.product([0, 1], repeat=n):
        env = dict(zip(inputs, combination))
        row = dict(env)
        for assign in assigns:
            # Evaluate simple combinational expressions safely
            try:
                var, expr = assign.split("=")
                var = var.strip()
                expr = expr.strip()
                expr_eval = expr.replace("&", " and ").replace("|", " or ").replace("~", " not ")
                row[var] = int(eval(expr_eval, {}, env))
                env[var] = row[var]
            except Exception:
                row[var] = "?"
        table_rows.append(row)
        for var in waveform:
            waveform[var].append(row.get(var, "?"))

    df = pd.DataFrame(table_rows)
    waveform_df = pd.DataFrame(waveform)
    return df, waveform, waveform_df

# -----------------------------
# Plot individual waveforms
def plot_waveforms(waveform_df):
    fig = go.Figure()
    x = list(range(len(waveform_df)))
    for i, col in enumerate(waveform_df.columns):
        fig.add_trace(go.Scatter(y=waveform_df[col], x=x, mode='lines+markers', name=col, line_shape='hv'))
    fig.update_layout(title="Input/Output Waveforms", xaxis_title="Combination Index", yaxis_title="Value (0/1)")
    return fig

# -----------------------------
# Generate block diagram
def generate_block_diagram(inputs, outputs):
    dot = graphviz.Digraph(comment='Module Block Diagram')
    for i in inputs:
        dot.node(i, i, shape='circle', color='blue')
    dot.node("MODULE", "Module", shape='box', style='filled', color='orange')
    for o in outputs:
        dot.node(o, o, shape='circle', color='green')
    for i in inputs:
        dot.edge(i, "MODULE")
    for o in outputs:
        dot.edge("MODULE", o)
    return dot

# -----------------------------
# Streamlit App
st.title("Verilog Testbench Generator & Explainer (Phase 2)")

verilog_code = st.text_area("Enter your Verilog code:")

if st.button("Generate Outputs"):
    inputs, outputs, assigns = parse_verilog(verilog_code)
    if not inputs or not outputs or not assigns:
        st.error("⚠️ Could not parse inputs/outputs properly.")
    else:
        st.subheader("Explanation & Testbench")
        explanation = get_gemini_explanation(verilog_code)
        st.code(explanation)

        st.subheader("Truth Table")
        df, waveform, waveform_df = generate_truth_table(inputs, outputs, assigns)
        if df is not None:
            st.dataframe(df)
            st.subheader("Waveforms")
            fig = plot_waveforms(waveform_df)
            st.plotly_chart(fig)
        else:
            st.warning("⚠️ Could not generate truth table or waveforms.")

        st.subheader("Block Diagram")
        try:
            dot = generate_block_diagram(inputs, outputs)
            st.graphviz_chart(dot)
        except Exception as e:
            st.warning("⚠️ Could not generate block diagram. Make sure Graphviz is installed.")
