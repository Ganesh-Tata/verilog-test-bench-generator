import streamlit as st
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import re

# Configure Gemini API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

st.title("🔧 Verilog Testbench & Analyzer")

code_input = st.text_area("Paste your Verilog code here:", height=300)

if st.button("Generate Testbench & Analysis"):
    if code_input.strip() == "":
        st.error("Please paste some Verilog code first!")
    else:
        with st.spinner("Analyzing code..."):

            # ---------------- EXPLANATION ----------------
            explanation = model.generate_content(
                f"Explain the following Verilog code in simple terms:\n{code_input}"
            ).text
            st.subheader("📖 Code Explanation")
            st.write(explanation)

            # ---------------- TESTBENCH ----------------
            testbench = model.generate_content(
                f"Write a Verilog testbench for the following code:\n{code_input}"
            ).text
            st.subheader("🧪 Generated Testbench")
            st.code(testbench, language="verilog")

            # Save & download
            st.download_button("⬇️ Download Testbench",
                               testbench,
                               file_name="generated_tb.v")

            # ---------------- TRUTH TABLE + WAVEFORMS ----------------
            st.subheader("📊 Truth Table & Waveforms")

            # Parse module header for inputs & outputs
            header_match = re.search(r"module\s+\w+\s*\((.*?)\);", code_input, re.S)
            if header_match:
                ports = header_match.group(1).replace("\n", "").split(",")
                ports = [p.strip() for p in ports]

                inputs = [p.split()[-1] for p in ports if "input" in p]
                outputs = [p.split()[-1] for p in ports if "output" in p]

                if inputs and outputs:
                    n = len(inputs)
                    rows = []
                    waveforms = {sig: [] for sig in inputs + outputs}

                    # Simulate all input combinations (brute force)
                    for i in range(2 ** n):
                        in_vals = [(i >> bit) & 1 for bit in range(n)]
                        row = {inputs[j]: in_vals[j] for j in range(n)}

                        # simple heuristic: check if assign matches
                        for out in outputs:
                            m = re.search(rf"assign\s+{out}\s*=\s*(.*?);", code_input)
                            if m:
                                expr = m.group(1)
                                expr_eval = expr
                                for j, inp in enumerate(inputs):
                                    expr_eval = expr_eval.replace(inp, str(in_vals[j]))
                                expr_eval = expr_eval.replace("&", " and ").replace("|", " or ").replace("~", " not ")
                                try:
                                    row[out] = int(eval(expr_eval))
                                except:
                                    row[out] = "?"
                            else:
                                row[out] = "?"

                        rows.append(row)

                        # Fill waveforms
                        for sig in inputs:
                            waveforms[sig].append(row[sig])
                        for sig in outputs:
                            waveforms[sig].append(row[sig])

                    df = pd.DataFrame(rows)
                    st.dataframe(df)

                    # Plot waveforms
                    fig, ax = plt.subplots(figsize=(8, len(waveforms)))
                    y_offset = 0
                    for sig, vals in waveforms.items():
                        ax.step(range(len(vals)), [v + y_offset for v in vals], where="post", label=sig)
                        y_offset += 2
                    ax.set_yticks([])
                    ax.legend(loc="upper right")
                    st.pyplot(fig)

                else:
                    st.warning("⚠️ Could not parse inputs/outputs properly.")
            else:
                st.warning("⚠️ No valid module header found.")

            # ---------------- BLOCK DIAGRAM ----------------
            st.subheader("📐 Block Diagram (Basic)")
            try:
                G = nx.DiGraph()
                for out in outputs:
                    for inp in inputs:
                        G.add_edge(inp, out)

                fig, ax = plt.subplots()
                pos = nx.spring_layout(G)
                nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2000, arrows=True, ax=ax)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate block diagram: {e}")
