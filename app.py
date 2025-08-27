import streamlit as st
import google.generativeai as genai
import re
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import graphviz

# -----------------------------
# Configure Gemini
# -----------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

st.set_page_config(page_title="Verilog Testbench Generator", layout="wide")

st.title("⚡ Verilog Testbench Generator & Visualizer")

# -----------------------------
# Input Section
# -----------------------------
code_input = st.text_area("✍️ Enter your Verilog Code:", height=250)

def parse_verilog(verilog_code):
    """Extract module, inputs, outputs, assigns."""
    inputs = re.findall(r"input\s+([\w, ]+)", verilog_code)
    outputs = re.findall(r"output\s+([\w, ]+)", verilog_code)
    assigns = re.findall(r"assign\s+(\w+)\s*=\s*(.*?);", verilog_code)

    inputs = [i.strip() for group in inputs for i in group.split(",")]
    outputs = [o.strip() for group in outputs for o in group.split(",")]
    return inputs, outputs, assigns

def verilog_to_python(expr):
    expr = expr.replace("&", " and ")
    expr = expr.replace("|", " or ")
    expr = expr.replace("~", " not ")
    expr = expr.replace("^", "!=")
    return expr

def generate_truth_table(inputs, outputs, assigns):
    """Generate truth table and waveform data."""
    table = []
    waveform = {sig: [] for sig in inputs + outputs}
    assign_map = {out: verilog_to_python(expr) for out, expr in assigns}

    for idx, combo in enumerate(itertools.product([0,1], repeat=len(inputs)), start=1):
        env = dict(zip(inputs, combo))
        for out, expr in assign_map.items():
            try:
                env[out] = int(eval(expr, {}, env))
            except Exception:
                env[out] = 0
        row = [idx] + [env[i] for i in inputs] + ["|"] + [env[o] for o in outputs]
        table.append(row)
        for sig in inputs + outputs:
            waveform[sig].append(env.get(sig, 0))

    columns = ["#"] + inputs + ["|"] + outputs
    df = pd.DataFrame(table, columns=columns)
    waveform_df = pd.DataFrame(waveform)
    return df, waveform, waveform_df

def plot_waveforms(waveform):
    """Plot timing diagram style waveforms."""
    fig, ax = plt.subplots(len(waveform), 1, figsize=(8, 2*len(waveform)), sharex=True)
    if len(waveform) == 1:
        ax = [ax]

    time = np.arange(len(next(iter(waveform.values()))))

    for i, (sig, values) in enumerate(waveform.items()):
        y = np.array(values)
        ax[i].step(time, y, where="post", linewidth=2)
        ax[i].set_ylim(-0.5, 1.5)
        ax[i].set_yticks([0,1])
        ax[i].set_yticklabels([f"{sig}=0", f"{sig}=1"])
        ax[i].grid(True, linestyle="--", alpha=0.6)

    ax[-1].set_xlabel("Time Steps")
    plt.tight_layout()
    return fig

def generate_block_diagram(inputs, outputs, assigns):
    """Generate block diagram using Graphviz."""
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="LR")

    # Module block
    dot.node("module", "Module", shape="box", style="filled", color="lightblue")

    # Inputs
    for i in inputs:
        dot.node(i, i, shape="circle", color="green")
        dot.edge(i, "module")

    # Outputs
    for o in outputs:
        dot.node(o, o, shape="circle", color="red")
        dot.edge("module", o)

    return dot

# -----------------------------
# Main Logic
# -----------------------------
if st.button("🚀 Generate Testbench & Visualizations"):
    if code_input.strip():
        inputs, outputs, assigns = parse_verilog(code_input)

        if not inputs or not outputs:
            st.error("❌ Could not detect inputs/outputs properly. Please check your Verilog code.")
        else:
            # Explanation
            st.subheader("📖 Code Explanation")
            explanation = model.generate_content(
                f"Explain the following Verilog code in simple terms:\n{code_input}"
            )
            st.write(explanation.text)

            # Testbench
            st.subheader("🧪 Generated Testbench")
            testbench = model.generate_content(
                f"Write a Verilog testbench for the following code:\n{code_input}"
            )
            st.code(testbench.text, language="verilog")

            # Truth Table
            st.subheader("📊 Truth Table")
            df, waveform, waveform_df = generate_truth_table(inputs, outputs, assigns)
            st.dataframe(df.style.set_properties(**{'text-align': 'center'}))

            # Waveforms
            st.subheader("📈 Waveform Visualization")
            fig = plot_waveforms(waveform)
            st.pyplot(fig)

            # Block Diagram
            st.subheader("🔲 Block Diagram")
            try:
                dot = generate_block_diagram(inputs, outputs, assigns)
                st.graphviz_chart(dot)
            except Exception as e:
                st.error(f"⚠️ Could not generate block diagram: {e}")
    else:
        st.warning("⚠️ Please enter Verilog code first.")
