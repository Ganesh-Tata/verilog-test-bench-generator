import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from graphviz import Digraph
import google.generativeai as genai
import os
import itertools

# ---------------------------
# Gemini API setup
# ---------------------------
genai.api_key = os.getenv("GEN_API_KEY")

st.set_page_config(page_title="Verilog Testbench Generator", layout="wide")

st.title("Phase 2: Verilog Testbench & Waveform Generator")

# ---------------------------
# User Input
# ---------------------------
code_input = st.text_area("Paste your combinational Verilog code here:", height=200)

if code_input.strip():
    st.success("Verilog code received.")
else:
    st.warning("Please input your Verilog code.")

# ---------------------------
# Function to parse inputs/outputs
# ---------------------------
def parse_verilog(code):
    inputs = []
    outputs = []
    assigns = []

    for line in code.splitlines():
        line = line.strip()
        if line.startswith("input"):
            for token in line.replace("input", "").replace(";", "").split(","):
                inputs.append(token.strip())
        elif line.startswith("output"):
            for token in line.replace("output", "").replace(";", "").split(","):
                outputs.append(token.strip())
        elif line.startswith("assign"):
            assigns.append(line.replace("assign", "").replace(";", "").strip())
    return inputs, outputs, assigns

# ---------------------------
# Generate truth table and waveforms
# ---------------------------
def generate_truth_table(inputs, outputs, assigns):
    try:
        combinations = list(itertools.product([0, 1], repeat=len(inputs)))
        df_rows = []
        waveform = {signal: [] for signal in inputs + outputs}

        for combo in combinations:
            env = dict(zip(inputs, combo))
            row = env.copy()
            for assign in assigns:
                left, expr = assign.split("=")
                left = left.strip()
                expr = expr.strip().replace("&", " and ").replace("|", " or ").replace("~", " not ")
                try:
                    row[left] = int(eval(expr, {}, env))
                except Exception:
                    row[left] = 0
            for key in waveform.keys():
                waveform[key].append(row.get(key, 0))
            df_rows.append(row)
        df = pd.DataFrame(df_rows)
        waveform_df = pd.DataFrame(waveform)
        return df, waveform, waveform_df
    except Exception:
        st.error("⚠️ Could not parse inputs/outputs properly.")
        return None, None, None

# ---------------------------
# Generate block diagram
# ---------------------------
def generate_block_diagram(inputs, outputs):
    dot = Digraph(comment="Combinational Block Diagram")
    for i in inputs:
        dot.node(i, i, shape="circle", color="lightblue2", style="filled")
    dot.node("MODULE", "Module", shape="box", color="orange", style="filled")
    for o in outputs:
        dot.node(o, o, shape="circle", color="lightgreen", style="filled")
    for i in inputs:
        dot.edge(i, "MODULE")
    for o in outputs:
        dot.edge("MODULE", o)
    return dot

# ---------------------------
# Generate testbench
# ---------------------------
def generate_testbench(inputs, outputs):
    tb = "module testbench;\n"
    for i in inputs:
        tb += f"  reg {i};\n"
    for o in outputs:
        tb += f"  wire {o};\n"
    tb += "  // Instantiate your module here\n"
    tb += "  initial begin\n"
    tb += "    // Test all combinations\n"
    tb += f"    for(integer i=0; i<{2**len(inputs)}; i=i+1) begin\n"
    tb += "      {"
    tb += ", ".join(inputs)
    tb += "} = i;\n"
    tb += "      #10;\n"
    tb += "    end\n"
    tb += "  end\nendmodule"
    return tb

# ---------------------------
# Generate explanation
# ---------------------------
def generate_explanation(code_input):
    try:
        explanation = model.generate_content(
            f"Explain the following Verilog code in simple terms:\n{code_input}"
        )
        return explanation.text
    except Exception:
        return "⚠️ Gemini API not available, using fallback explanation."

# ---------------------------
# Main Logic
# ---------------------------
if code_input.strip():
    inputs, outputs, assigns = parse_verilog(code_input)
    df, waveform, waveform_df = generate_truth_table(inputs, outputs, assigns)

    if df is not None:
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Truth Table", "Waveforms", "Block Diagram", "Testbench/Explanation"])

        with tab1:
            st.subheader("Truth Table")
            st.dataframe(df)

        with tab2:
            st.subheader("Input/Output Waveforms")
            fig = go.Figure()
            for signal in waveform.keys():
                fig.add_trace(go.Scatter(
                    y=waveform[signal],
                    name=signal,
                    mode='lines+markers'
                ))
            fig.update_layout(
                xaxis_title="Combination index",
                yaxis_title="Signal value (0/1)",
                yaxis=dict(tickvals=[0,1])
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Block Diagram")
            dot = generate_block_diagram(inputs, outputs)
            st.graphviz_chart(dot)

        with tab4:
            st.subheader("Testbench Code")
            tb_code = generate_testbench(inputs, outputs)
            st.code(tb_code, language="verilog")

            st.subheader("Explanation")
            explanation_text = "Fallback: This module implements combinational logic with inputs " + ", ".join(inputs)
            st.write(explanation_text)
