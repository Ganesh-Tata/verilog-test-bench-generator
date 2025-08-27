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

    # 3. Extract Inputs & Outputs (Regex based)
    module_header = re.search(r"module\s+\w+\s*\((.*?)\);", code_input, re.S)
    ports = module_header.group(1) if module_header else ""
    inputs = re.findall(r"input\s+(\w+)", code_input)
    outputs = re.findall(r"output\s+(\w+)", code_input)

    st.subheader("🔌 Detected Ports")
    st.write(f"**Inputs:** {inputs}")
    st.write(f"**Outputs:** {outputs}")

    # 4. Generate Truth Table (only if <= 3 inputs)
    if len(inputs) > 0 and len(inputs) <= 3 and len(outputs) > 0:
        st.subheader("📊 Truth Table (auto-generated)")
        rows = []
        for combo in itertools.product([0,1], repeat=len(inputs)):
            row = dict(zip(inputs, combo))
            # Let Gemini evaluate outputs
            logic_prompt = f"""
            Given the Verilog code:
            {code_input}

            For inputs {row}, what should be the output values of {outputs}? 
            Respond with only a Python dictionary mapping.
            """
            try:
                resp = model.generate_content(logic_prompt)
                out_dict = eval(resp.text.strip())
                row.update(out_dict)
            except:
                row.update({sig: "?" for sig in outputs})
            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df)

        # 5. Generate Waveforms
        st.subheader("📈 Waveforms")
        t = np.arange(len(df))  # one step per row
        fig, axes = plt.subplots(len(inputs)+len(outputs), 1, figsize=(8, 5), sharex=True)

        # Plot inputs
        for i, sig in enumerate(inputs):
            axes[i].step(t, df[sig], where="post")
            axes[i].set_ylabel(sig)
            axes[i].set_ylim(-0.5, 1.5)

        # Plot outputs
        for j, sig in enumerate(outputs):
            axes[len(inputs)+j].step(t, df[sig], where="post", color="red")
            axes[len(inputs)+j].set_ylabel(sig)
            axes[len(inputs)+j].set_ylim(-0.5, 1.5)

        axes[-1].set_xlabel("Simulation Step")
        st.pyplot(fig)

    else:
        st.info("Truth table & waveforms available only if input count ≤ 3.")

    # 6. Block Diagram
    st.subheader("🖼 Block Diagram")
    with st.spinner("Generating block diagram..."):
        diagram_resp = model.generate_content(
            f"""
            Generate a Graphviz DOT format description of a block diagram 
            for the following Verilog module. Show inputs on the left, 
            outputs on the right, and the module as a central box.
            
            Verilog code:
            {code_input}
            """
        )
    try:
        dot_code = diagram_resp.text.strip()
        st.graphviz_chart(dot_code)
    except:
        st.error("Failed to generate block diagram.")
