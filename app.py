import streamlit as st
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import graphviz
import re
import os

# -----------------------------
# Configure Gemini API
# -----------------------------
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

st.title("🔧 Verilog Testbench & Visualization Tool")

# -----------------------------
# Helper: Evaluate simple combinational assign
# -----------------------------
def eval_logic(assign_stmt, inputs_dict):
    expr = assign_stmt
    # Replace logical operators
    expr = expr.replace("&", " and ")
    expr = expr.replace("|", " or ")
    expr = expr.replace("~", " not ")
    expr = expr.replace("^", " != ")  # XOR
    # Replace variables with values
    for k, v in inputs_dict.items():
        expr = re.sub(rf"\b{k}\b", str(bool(v)), expr)
    try:
        return int(eval(expr))
    except:
        return "?"

# -----------------------------
# User Input
# -----------------------------
code_input = st.text_area("Paste your Verilog code here:")

if st.button("Generate Analysis") and code_input.strip():

    # 1. Explain the code
    with st.spinner("Analyzing code..."):
        explanation = model.generate_content(
            f"Explain the following Verilog code in simple terms:\n{code_input}"
        )
    st.subheader("📘 Code Explanation")
    st.write(explanation.text)

    # 2. Generate Testbench
    with st.spinner("Generating testbench..."):
        testbench = model.generate_content(
            f"Write a Verilog testbench for the following code:\n{code_input}"
        )
    st.subheader("🧪 Generated Testbench")
    st.code(testbench.text, language="verilog")

    # 3. Extract Inputs & Outputs
    inputs = re.findall(r"input\s+(\w+)", code_input)
    outputs = re.findall(r"output\s+(\w+)", code_input)
    assigns = re.findall(r"assign\s+(\w+)\s*=\s*(.*?);", code_input)

    st.subheader("🔌 Detected Ports")
    st.write(f"**Inputs:** {inputs}")
    st.write(f"**Outputs:** {outputs}")

    # 4. Truth Table (if <= 3 inputs)
    if len(inputs) > 0 and len(inputs) <= 3 and len(outputs) > 0:
        st.subheader("📊 Truth Table (calculated)")

        rows = []
        for combo in itertools.product([0,1], repeat=len(inputs)):
            row = dict(zip(inputs, combo))
            for out in outputs:
                logic_expr = next((rhs for lhs, rhs in assigns if lhs == out), None)
                if logic_expr:
                    row[out] = eval_logic(logic_expr, row)
                else:
                    row[out] = "?"
            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df)

        # 5. Waveforms
        st.subheader("📈 Waveforms")
        t = np.arange(len(df) * 2)  # two steps per row for flat edges
        fig, axes = plt.subplots(len(inputs)+len(outputs), 1, figsize=(8, 6), sharex=True)

        # Repeat each value twice for step-like waveform
        for i, sig in enumerate(inputs):
            sig_vals = np.repeat(df[sig].values, 2)
            axes[i].step(t, sig_vals, where="post")
            axes[i].set_ylabel(sig)
            axes[i].set_ylim(-0.5, 1.5)

        for j, sig in enumerate(outputs):
            sig_vals = np.repeat(df[sig].values, 2)
            axes[len(inputs)+j].step(t, sig_vals, where="post", color="red")
            axes[len(inputs)+j].set_ylabel(sig)
            axes[len(inputs)+j].set_ylim(-0.5, 1.5)

        axes[-1].set_xlabel("Simulation Step")
        st.pyplot(fig)

    else:
        st.info("Truth table & waveforms available only if input count ≤ 3.")

    # 6. Block Diagram
    st.subheader("🖼 Block Diagram")
    with st.spinner("Generating block diagram..."):
        try:
            diagram_resp = model.generate_content(
                f"""
                Generate a Graphviz DOT description of a block diagram 
                for this Verilog module. Inputs on left, outputs on right, 
                module box in center.
                Verilog:
                {code_input}
                """
            )
            dot_code = diagram_resp.text.strip()
            if "digraph" not in dot_code:
                # fallback basic block diagram
                dot_code = "digraph G { rankdir=LR; " \
                           + "; ".join([f"{i} -> module" for i in inputs]) + "; " \
                           + "module [shape=box]; " \
                           + "; ".join([f"module -> {o}" for o in outputs]) + " }"
            st.graphviz_chart(dot_code)
        except Exception as e:
            st.error(f"Block diagram failed. Error: {e}")
