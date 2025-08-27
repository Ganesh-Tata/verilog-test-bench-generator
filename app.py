import streamlit as st
import google.generativeai as genai
import matplotlib.pyplot as plt
import itertools
import re
import os
import pandas as pd

# -----------------------------
# Setup Gemini API
# -----------------------------
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

st.set_page_config(page_title="Verilog Testbench & Simulator", layout="wide")

st.title("⚡ Verilog Testbench Generator & Visualizer")

# -----------------------------
# Input Code
# -----------------------------
code_input = st.text_area("Paste your Verilog code here:", height=250)

if st.button("Generate Testbench & Analysis"):
    if not code_input.strip():
        st.warning("⚠️ Please enter Verilog code.")
    else:
        # Detect Sequential vs Combinational
        is_sequential = any(kw in code_input for kw in ["always", "posedge", "negedge"])

        # -----------------------------
        # 1. Explanation of Code
        # -----------------------------
        try:
            explanation = model.generate_content(
                f"Explain the following Verilog code step by step:\n{code_input[:2000]}"
            )
            st.subheader("📘 Code Explanation")
            st.write(explanation.text)
        except Exception as e:
            st.error(f"Error generating explanation: {e}")

        # -----------------------------
        # 2. Testbench Generation
        # -----------------------------
        try:
            tb = model.generate_content(
                f"Generate a Verilog testbench for the following module:\n{code_input[:2000]}"
            )
            st.subheader("🧪 Generated Testbench")
            st.code(tb.text, language="verilog")
        except Exception as e:
            st.error(f"Error generating testbench: {e}")

        # -----------------------------
        # 3. Block Diagram (AI-generated)
        # -----------------------------
        try:
            block = model.generate_content(
                f"Generate a simple text-based block diagram for the following Verilog module:\n{code_input[:1500]}\n"
                f"Use only ASCII or Markdown-compatible SVG so it can be displayed directly."
            )
            st.subheader("📦 Block Diagram")
            st.code(block.text, language="markdown")
        except Exception as e:
            st.error(f"Error generating block diagram: {e}")

        # -----------------------------
        # 4. Simulation / Truth Table
        # -----------------------------
        if not is_sequential:
            st.subheader("📊 Truth Table & Waveforms")

            # Extract inputs and outputs
            inputs = re.findall(r"input\s+(?:\[\d+:\d+\]\s*)?(\w+)", code_input)
            outputs = re.findall(r"output\s+(?:\[\d+:\d+\]\s*)?(\w+)", code_input)

            if inputs and outputs:
                # Generate input combinations
                combinations = list(itertools.product([0, 1], repeat=len(inputs)))
                truth_data = []

                for combo in combinations:
                    combo_dict = dict(zip(inputs, combo))
                    # Let Gemini predict outputs for each combo
                    try:
                        sim = model.generate_content(
                            f"Given this Verilog module:\n{code_input}\n"
                            f"Inputs: {combo_dict}\n"
                            f"Predict the outputs."
                        )
                        output_vals = sim.text.strip()
                    except:
                        output_vals = "?"

                    truth_data.append({**combo_dict, "Outputs": output_vals})

                df = pd.DataFrame(truth_data)
                st.dataframe(df)

                # Waveform plotting
                fig, ax = plt.subplots(figsize=(8, 3))
                time = list(range(len(combinations)))
                for i, inp in enumerate(inputs):
                    ax.step(time, [row[inp] for row in truth_data], label=inp, where="post")
                # For outputs we just show as textual overlay
                for idx, row in enumerate(truth_data):
                    ax.text(idx, -0.5, str(row["Outputs"]), ha="center", fontsize=8)

                ax.set_ylim(-1.5, len(inputs) + 1)
                ax.set_yticks(range(len(inputs)))
                ax.set_yticklabels(inputs)
                ax.set_xlabel("Time (steps)")
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("Could not detect input/output signals for truth table.")
        else:
            st.subheader("⏱️ Sequential Timing Diagram")
            try:
                timing = model.generate_content(
                    f"Generate a timing diagram (clock, inputs, outputs) in ASCII table format for:\n{code_input[:2000]}"
                )
                st.text(timing.text)
            except Exception as e:
                st.error(f"Error generating timing diagram: {e}")
