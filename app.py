import streamlit as st
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import io

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Verilog Testbench Generator", layout="wide")
st.title("🔧 Verilog Testbench & Analyzer")
st.write("Paste your Verilog code and get explanation, testbench, truth table, and basic waveforms.")

# -----------------------------
# API Key Setup
# -----------------------------
api_key = st.text_input("🔑 Enter your Google Gemini API Key:", type="password")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
else:
    st.warning("Please enter your Gemini API key to continue.")
    st.stop()

# -----------------------------
# Code Input
# -----------------------------
code_input = st.text_area("✍️ Paste your Verilog code here:", height=300)

if st.button("Generate Testbench & Explanation"):
    if not code_input.strip():
        st.error("Please paste Verilog code before proceeding.")
    else:
        # -----------------------------
        # Gemini: Explanation
        # -----------------------------
        try:
            explanation = model.generate_content(
                f"Explain the following Verilog code in simple terms (max 200 words):\n{code_input[:1500]}"
            )
            explanation_text = explanation.text
        except Exception as e:
            explanation_text = f"⚠️ Error generating explanation: {str(e)}"

        # -----------------------------
        # Gemini: Testbench
        # -----------------------------
        try:
            testbench = model.generate_content(
                f"Generate a Verilog testbench for the following module:\n{code_input[:1500]}"
            )
            testbench_code = testbench.text
        except Exception as e:
            testbench_code = f"⚠️ Error generating testbench: {str(e)}"

        # -----------------------------
        # Display Results
        # -----------------------------
        st.subheader("📘 Explanation")
        st.write(explanation_text)

        st.subheader("🧪 Generated Testbench")
        st.code(testbench_code, language="verilog")

        # Allow download
        st.download_button(
            label="⬇️ Download Testbench",
            data=testbench_code,
            file_name="testbench.v",
            mime="text/plain",
        )

        # -----------------------------
        # Simple Simulation (Truth Table)
        # -----------------------------
        st.subheader("📊 Truth Table (Sample Simulation)")
        # Assume max 3 inputs for simplicity
        inputs = ["A", "B", "C"]
        num_inputs = 3
        truth_table = []
        for i in range(2**num_inputs):
            bits = [(i >> j) & 1 for j in range(num_inputs)]
            output = bits[0] & bits[1]  # Example logic (A AND B)
            truth_table.append(bits + [output])

        st.table(truth_table)

        # -----------------------------
        # Waveform Plot
        # -----------------------------
        st.subheader("📈 Sample Waveform (Simulated)")
        t = np.linspace(0, 1, 16)
        a = (np.sin(2 * np.pi * 2 * t) > 0).astype(int)
        b = (np.sin(2 * np.pi * 4 * t) > 0).astype(int)
        y = a & b

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.step(t, a, where="post", label="A")
        ax.step(t, b, where="post", label="B")
        ax.step(t, y, where="post", label="Y = A AND B")
        ax.set_ylim(-0.5, 1.5)
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        st.pyplot(fig)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("⚡ Built with Streamlit + Gemini API")
