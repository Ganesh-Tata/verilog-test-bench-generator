import streamlit as st
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import re
import json
import os

# -----------------------------
# Configure Gemini API
# -----------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-pro")

# -----------------------------
# Helper: Detect if sequential or combinational
# -----------------------------
def is_sequential(verilog_code: str) -> bool:
    return any(kw in verilog_code.lower() for kw in ["always", "posedge", "negedge", "clk", "clock"])

# -----------------------------
# Helper: Extract inputs/outputs
# -----------------------------
def extract_io(verilog_code: str):
    inputs = re.findall(r"input\s+(?:\[.*?\]\s*)?(\w+)", verilog_code)
    outputs = re.findall(r"output\s+(?:\[.*?\]\s*)?(\w+)", verilog_code)
    return inputs, outputs

# -----------------------------
# Helper: Simulate combinational logic
# -----------------------------
def simulate_combinational(verilog_code: str, inputs, outputs):
    # Extract assign statements
    assigns = re.findall(r"assign\s+(\w+)\s*=\s*(.*?);", verilog_code)

    truth_table = []
    n = len(inputs)
    for i in range(2 ** n):
        values = {inp: (i >> (n - 1 - idx)) & 1 for idx, inp in enumerate(inputs)}
        row = values.copy()

        for outp in outputs:
            exprs = [a for a in assigns if a[0] == outp]
            if exprs:
                expr = exprs[0][1]
                # Replace logical ops with Python equivalents
                pyexpr = expr.replace("&", " and ").replace("|", " or ").replace("~", " not ")
                for var in values:
                    pyexpr = re.sub(rf"\b{var}\b", str(values[var]), pyexpr)
                try:
                    row[outp] = int(eval(pyexpr))
                except:
                    row[outp] = "?"
            else:
                row[outp] = "?"

        truth_table.append(row)

    return truth_table

# -----------------------------
# Helper: Plot waveform
# -----------------------------
def plot_waveform(truth_table, inputs, outputs):
    fig, ax = plt.subplots(len(inputs) + len(outputs), 1, figsize=(8, 2*(len(inputs)+len(outputs))), sharex=True)

    all_signals = inputs + outputs
    time = np.arange(len(truth_table))

    for idx, signal in enumerate(all_signals):
        values = [row[signal] if row[signal] in [0, 1] else 0 for row in truth_table]
        ax[idx].step(time, values, where="post")
        ax[idx].set_ylim(-0.5, 1.5)
        ax[idx].set_ylabel(signal, rotation=0, labelpad=30)
        ax[idx].grid(True)

    ax[-1].set_xlabel("Time steps")
    st.pyplot(fig)

# -----------------------------
# Streamlit App
# -----------------------------
st.title("🔧 Verilog Testbench & Simulation Generator")

code_input = st.text_area("Paste your Verilog module code:", height=250)

if st.button("Generate Testbench & Analysis"):
    if code_input.strip() == "":
        st.error("Please provide Verilog code!")
    else:
        inputs, outputs = extract_io(code_input)

        # -----------------------------
        # Step 1: Code Explanation
        # -----------------------------
        with st.spinner("Generating explanation..."):
            explanation = model.generate_content(
                f"Explain the following Verilog code in simple terms:\n{code_input}"
            )
            st.subheader("📘 Code Explanation")
            st.write(explanation.text)

        # -----------------------------
        # Step 2: Testbench Generation
        # -----------------------------
        with st.spinner("Generating testbench..."):
            testbench = model.generate_content(
                f"Write a Verilog testbench for this code:\n{code_input}"
            )
            st.subheader("📝 Generated Testbench")
            st.code(testbench.text, language="verilog")

        # -----------------------------
        # Step 3: Simulation / Waveforms
        # -----------------------------
        st.subheader("📊 Simulation & Waveforms")

        if not is_sequential(code_input):
            st.success("✅ Detected **Combinational Logic** → Using real simulation")
            truth_table = simulate_combinational(code_input, inputs, outputs)

            # Display truth table
            st.write("### Truth Table")
            st.dataframe(truth_table)

            # Plot waveform
            st.write("### Waveform")
            plot_waveform(truth_table, inputs, outputs)

        else:
            st.warning("⚠️ Detected **Sequential Logic** → Using AI-predicted waveforms")
            sim_result = model.generate_content(
                f"Generate a JSON of waveforms (0/1 values for each timestep) for inputs and outputs "
                f"of this Verilog code:\n{code_input}\n"
                f"Format strictly as: {{'signal_name':[0,1,0,...]}}"
            )
            try:
                waveforms = json.loads(sim_result.text.replace("```json", "").replace("```", "").strip())
                fig, ax = plt.subplots(len(waveforms), 1, figsize=(8, 2*len(waveforms)), sharex=True)
                time = np.arange(len(list(waveforms.values())[0]))
                for idx, (signal, values) in enumerate(waveforms.items()):
                    ax[idx].step(time, values, where="post")
                    ax[idx].set_ylim(-0.5, 1.5)
                    ax[idx].set_ylabel(signal, rotation=0, labelpad=30)
                    ax[idx].grid(True)
                ax[-1].set_xlabel("Time steps")
                st.pyplot(fig)
            except Exception as e:
                st.error("Failed to generate AI waveform ❌")
                st.text(str(e))
