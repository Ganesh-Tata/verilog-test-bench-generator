import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import graphviz
import os
import google.generativeai as genai

# ----------------------------
# Set your Gemini API Key here or as environment variable
# ----------------------------
API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=API_KEY)

st.title("Verilog Testbench Generator & Waveform Viewer")

# ----------------------------
# User input
# ----------------------------
code_input = st.text_area("Enter your Verilog Code:", height=250)

generate_tb = st.button("Generate Testbench & Analysis")

if generate_tb and code_input.strip():
    st.info("Processing...")

    # ----------------------------
    # Generate explanation (fallback if API not available)
    # ----------------------------
    try:
        explanation = genai.models.generate_content(
            model="models/text-bison-001",
            prompt=f"Explain the following Verilog code in simple terms:\n{code_input}"
        ).content
    except:
        explanation = "⚠️ Gemini API not available, using fallback explanation."
    st.subheader("Explanation")
    st.code(explanation)

    # ----------------------------
    # Parse inputs, outputs, assigns
    # ----------------------------
    import re

    inputs = re.findall(r'input\s+([a-zA-Z0-9_]+)', code_input)
    outputs = re.findall(r'output\s+([a-zA-Z0-9_]+)', code_input)
    assigns = re.findall(r'assign\s+([a-zA-Z0-9_]+)\s*=\s*([^;]+);', code_input)

    if not inputs or not outputs:
        st.error("⚠️ Could not parse inputs/outputs properly.")
    else:
        st.subheader("Inputs & Outputs")
        st.write("Inputs:", inputs)
        st.write("Outputs:", outputs)

        # ----------------------------
        # Generate Truth Table
        # ----------------------------
        from itertools import product

        input_combinations = list(product([0, 1], repeat=len(inputs)))
        truth_table = []
        waveform_dict = {sig: [] for sig in inputs + outputs}

        for comb in input_combinations:
            env = dict(zip(inputs, comb))
            row = list(comb)
            for out_sig, expr in assigns:
                try:
                    # safe eval with env
                    val = eval(expr, {}, env)
                    row.append(val)
                    env[out_sig] = val
                except:
                    row.append("?")
                    env[out_sig] = "?"
            truth_table.append(row)
            for i, val in enumerate(row):
                waveform_dict[list(waveform_dict.keys())[i]].append(val)

        df = pd.DataFrame(truth_table, columns=inputs + [o for o, _ in assigns])
        st.subheader("Truth Table")
        st.dataframe(df)

        # ----------------------------
        # Waveform Plot (Step Digital)
        # ----------------------------
        def plot_waveforms_plotly(waveform_df):
            fig = go.Figure()
            n_signals = len(waveform_df.columns)

            for idx, sig in enumerate(waveform_df.columns):
                y_vals = waveform_df[sig].replace("?", 0).tolist()
                x_vals = []
                y_step = []
                for i, val in enumerate(y_vals):
                    x_vals.extend([i, i+1])
                    y_step.extend([val + idx*2, val + idx*2])
                fig.add_trace(go.Scatter(x=x_vals, y=y_step,
                                         mode='lines', name=sig))

            fig.update_layout(
                height=150 + 50*n_signals,
                yaxis=dict(
                    tickvals=[i*2 + 0.5 for i in range(n_signals)],
                    ticktext=list(waveform_df.columns)
                ),
                xaxis_title="Input Combination Index",
                title="Digital Waveforms (Step-wise)",
                showlegend=True
            )
            return fig

        waveform_df = pd.DataFrame(waveform_dict)
        st.subheader("Waveform Viewer")
        fig = plot_waveforms_plotly(waveform_df)
        st.plotly_chart(fig)

        # ----------------------------
        # Block Diagram Generation
        # ----------------------------
        try:
            dot = graphviz.Digraph(comment='Block Diagram')
            for inp in inputs:
                dot.node(inp, shape="box")
            for outp in [o for o, _ in assigns]:
                dot.node(outp, shape="ellipse")
            for outp, expr in assigns:
                expr_inputs = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr)
                for ei in expr_inputs:
                    if ei in inputs:
                        dot.edge(ei, outp)
            st.subheader("Block Diagram")
            st.graphviz_chart(dot)
        except:
            st.error("⚠️ Could not generate block diagram. Ensure Graphviz executables are installed.")

        # ----------------------------
        # Testbench Generation (Simple)
        # ----------------------------
        tb_lines = ["`timescale 1ns / 1ps", f"module tb;"]
        for inp in inputs:
            tb_lines.append(f"reg {inp};")
        for outp in [o for o, _ in assigns]:
            tb_lines.append(f"wire {outp};")
        tb_lines.append(f"{re.findall(r'module\s+([a-zA-Z0-9_]+)', code_input)[0]} uut (.*);")
        tb_lines.append("initial begin")
        for i, comb in enumerate(input_combinations):
            assign_str = " ".join([f"{inp}={v};" for inp, v in zip(inputs, comb)])
            tb_lines.append(f"#10 {assign_str}")
        tb_lines.append("end")
        tb_lines.append("endmodule")
        st.subheader("Generated Testbench")
        st.code("\n".join(tb_lines))

