import streamlit as st
import google.generativeai as genai
import matplotlib.pyplot as plt
import itertools
import re
import os
import pandas as pd
import graphviz
from io import BytesIO

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
        # 3. Block Diagram (LOCAL)
        # -----------------------------
        st.subheader("📦 Block Diagram")
        inputs = re.findall(r"input\s+(?:\[\d+:\d+\]\s*)?(\w+)", code_input)
        outputs = re.findall(r"output\s+(?:\[\d+:\d+\]\s*)?(\w+)", code_input)
        module_name = re.search(r"module\s+(\w+)", code_input)

        dot = graphviz.Digraph()
        if module_name:
            dot.node("M", module_name.group(1), shape="box", style="filled", color="lightblue")
            for inp in inputs:
                dot.node(inp, inp, shape="circle", color="green")
                dot.edge(inp, "M")
            for outp in outputs:
                dot.node(outp, outp, shape="circle", color="red")
                dot.edge("M", outp)

        st.graphviz_chart(dot)

        # Export block diagram as SVG
        try:
            svg_data = dot.pipe(format="svg")
            st.download_button("⬇️ Download Block Diagram (SVG)", svg_data, file_name="block_diagram.svg")
        except Exception as e:
            st.warning(f"Could not export block diagram: {e}")

        # -----------------------------
        # 4. Simulation / Truth Table
        # -----------------------------
        if not is_sequential:
            st.subheader("📊 Truth Table & Waveforms")

            # Extract inputs and outputs
            inputs = list(dict.fromkeys(inputs))  # remove duplicates
            outputs = list(dict.fromkeys(outputs))

            if inputs and outputs:
                # Build all input combinations
                combinations = list(itertools.product([0, 1], repeat=len(inputs)))
                truth_data = []

                for combo in combinations:
                    combo_dict = dict(zip(inputs, combo))

                    # Ask Gemini only to evaluate outputs (short JSON)
                    try:
                        sim = model.generate_content(
                            f"Given this combinational Verilog module:\n{code_input}\n"
                            f"Inputs = {combo_dict}\n"
                            f"Respond ONLY with output values in JSON, like {{'y':0}}"
                        )
                        out_dict = {}
                        try:
                            out_dict = eval(sim.text.strip())
                        except:
                            for o in outputs:
                                out_dict[o] = "?"
                    except:
                        out_dict = {o: "?" for o in outputs}

                    truth_data.append({**combo_dict, **out_dict})

                df = pd.DataFrame(truth_data)
                st.dataframe(df)

                # Export truth table as CSV
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Truth Table (CSV)", csv_data, file_name="truth_table.csv")

                # Plot full waveform sequence
                fig, ax = plt.subplots(figsize=(10, 4))
                time = list(range(len(combinations)))

                # Plot inputs
                for i, inp in enumerate(inputs):
                    values = [row[inp] for row in truth_data]
                    ax.step(time, [v + 2*i for v in values], where="post", label=inp)

                # Plot outputs shifted higher
                for j, outp in enumerate(outputs):
                    values = [int(row[outp]) if str(row[outp]).isdigit() else 0 for row in truth_data]
                    ax.step(time, [v + 2*(len(inputs)+j) for v in values], where="post", label=outp)

                ax.set_yticks(range(0, 2*(len(inputs)+len(outputs)), 2))
                ax.set_yticklabels(inputs + outputs)
                ax.set_xlabel("Time (input sequence steps)")
                ax.legend(loc="upper right")
                st.pyplot(fig)

                # Export waveform as PNG
                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.download_button("⬇️ Download Waveform (PNG)", buf.getvalue(), file_name="waveform.png")
            else:
                st.warning("⚠️ Could not detect input/output signals for truth table.")
        else:
            st.subheader("⏱️ Sequential Timing Diagram")
            try:
                timing = model.generate_content(
                    f"Generate a timing diagram (clock, inputs, outputs) in ASCII table format for:\n{code_input[:2000]}"
                )
                st.text(timing.text)
            except Exception as e:
                st.error(f"Error generating timing diagram: {e}")
