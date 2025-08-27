import streamlit as st
import google.generativeai as genai
import os
import itertools
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Verilog Testbench Generator", layout="wide")

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

st.title("🔧 Verilog Testbench Generator with Truth Table, Waveforms & Block Diagram")

# -----------------------------
# USER INPUT
# -----------------------------
code_input = st.text_area("Paste your Verilog module code here:", height=300)

if st.button("Generate Testbench & Analysis"):
    if not code_input.strip():
        st.error("⚠️ Please paste your Verilog code.")
    else:
        # -----------------------------
        # 1. Extract Module Info
        # -----------------------------
        inputs = re.findall(r'input\s+([a-zA-Z0-9_]+)', code_input)
        outputs = re.findall(r'output\s+([a-zA-Z0-9_]+)', code_input)

        st.subheader("📌 Detected Ports")
        st.write(f"**Inputs:** {inputs}")
        st.write(f"**Outputs:** {outputs}")

        # -----------------------------
        # 2. Generate Testbench (Gemini)
        # -----------------------------
        with st.spinner("Generating testbench..."):
            tb = model.generate_content(
                f"Write a Verilog testbench for the following code:\n{code_input}"
            )
        st.subheader("📝 Generated Testbench")
        st.code(tb.text, language="verilog")

        # -----------------------------
        # 3. Explain the Code (Gemini)
        # -----------------------------
        with st.spinner("Explaining Verilog code..."):
            explanation = model.generate_content(
                f"Explain the following Verilog code in simple terms:\n{code_input}"
            )
        st.subheader("📖 Code Explanation")
        st.write(explanation.text)

        # -----------------------------
        # 4. Truth Table & Waveforms
        # -----------------------------
        st.subheader("📊 Truth Table & Waveforms")

        if inputs and outputs:
            n_inputs = len(inputs)
            all_combos = list(itertools.product([0, 1], repeat=n_inputs))

            # For now, just mark outputs as "?" (since parsing expressions is non-trivial)
            # Later this can be extended to parse assign statements safely
            truth_data = []
            for combo in all_combos:
                row = list(combo) + ["?"] * len(outputs)
                truth_data.append(row)

            import pandas as pd
            df = pd.DataFrame(truth_data, columns=inputs + outputs)
            st.write(df)

            # Waveforms: show each signal separately
            fig, axes = plt.subplots(len(inputs) + len(outputs), 1,
                                     figsize=(8, 1.5 * (len(inputs) + len(outputs))),
                                     sharex=True)

            time = np.arange(len(all_combos))
            for i, sig in enumerate(inputs):
                vals = [combo[i] for combo in all_combos]
                axes[i].step(time, vals, where="post")
                axes[i].set_ylabel(sig)
                axes[i].set_ylim(-0.2, 1.2)

            for j, sig in enumerate(outputs):
                vals = [0 if row[-len(outputs) + j] == "0" else 1 if row[-len(outputs) + j] == "1" else 0 for row in truth_data]
                axes[len(inputs) + j].step(time, vals, where="post", color="red")
                axes[len(inputs) + j].set_ylabel(sig)
                axes[len(inputs) + j].set_ylim(-0.2, 1.2)

            plt.xlabel("Time step")
            st.pyplot(fig)

        else:
            st.warning("⚠️ Could not detect valid inputs/outputs for truth table & waveforms.")

        # -----------------------------
        # 5. Block Diagram (Matplotlib)
        # -----------------------------
        st.subheader("📐 Block Diagram")

        fig, ax = plt.subplots(figsize=(6, 4))

        # Draw main module box
        ax.add_patch(Rectangle((0.3, 0.3), 0.4, 0.4, fill=None, edgecolor="black", linewidth=2))
        ax.text(0.5, 0.5, "Module", ha="center", va="center", fontsize=12, weight="bold")

        # Place inputs on left
        for i, inp in enumerate(inputs):
            y_pos = 0.7 - i * (0.6 / max(1, len(inputs) - 1))
            ax.text(0.25, y_pos, inp, ha="right", va="center", fontsize=10)
            ax.add_patch(FancyArrow(0.25, y_pos, 0.05, 0, width=0.005))

        # Place outputs on right
        for i, out in enumerate(outputs):
            y_pos = 0.7 - i * (0.6 / max(1, len(outputs) - 1))
            ax.text(0.75, y_pos, out, ha="left", va="center", fontsize=10)
            ax.add_patch(FancyArrow(0.7, y_pos, 0.05, 0, width=0.005))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        st.pyplot(fig)
