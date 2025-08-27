import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from graphviz import Digraph
import re
import google.generativeai as genai
import os

# -----------------------------
# Gemini API setup
# -----------------------------
genai.api_key = os.getenv("GEMINI_API_KEY")

# -----------------------------
# Fallback explanation function
# -----------------------------
def fallback_explanation(verilog_code):
    lines = verilog_code.splitlines()
    explanation_lines = []

    module_match = re.search(r'module\s+([a-zA-Z0-9_]+)', verilog_code)
    if module_match:
        explanation_lines.append(f"This Verilog module is named '{module_match.group(1)}'.")

    inputs = re.findall(r'input\s+([a-zA-Z0-9_]+)', verilog_code)
    if inputs:
        explanation_lines.append(f"It has {len(inputs)} input(s): {', '.join(inputs)}.")

    outputs = re.findall(r'output\s+([a-zA-Z0-9_]+)', verilog_code)
    if outputs:
        explanation_lines.append(f"It has {len(outputs)} output(s): {', '.join(outputs)}.")

    assigns = re.findall(r'assign\s+([a-zA-Z0-9_]+)\s*=\s*([^;]+);', verilog_code)
    if assigns:
        logic_desc = []
        for out_sig, expr in assigns:
            logic_desc.append(f"{out_sig} = {expr.strip()}")
        explanation_lines.append("Logic assignments:\n- " + "\n- ".join(logic_desc))

    if not explanation_lines:
        return "This is a Verilog module, but its inputs/outputs or logic could not be parsed."
    return "\n".join(explanation_lines)

# -----------------------------
# Truth table generator
# -----------------------------
def generate_truth_table(inputs, outputs, assigns):
    import itertools

    waveform = {inp: [] for inp in inputs}
    waveform_y = {outp: [] for outp in outputs}
    table_rows = []

    for combo in itertools.product([0,1], repeat=len(inputs)):
        row = dict(zip(inputs, combo))
        local_dict = row.copy()
        for out_sig, expr in assigns:
            try:
                expr_eval = expr
                for var in local_dict:
                    expr_eval = expr_eval.replace(var, str(local_dict[var]))
                local_dict[out_sig] = eval(expr_eval)
            except:
                local_dict[out_sig] = '?'

        for inp in inputs:
            waveform[inp].append(row[inp])
        for outp in outputs:
            waveform_y[outp].append(local_dict[outp])

        table_rows.append(local_dict)

    df = pd.DataFrame(table_rows)
    waveform_df = pd.DataFrame({**waveform, **waveform_y})
    return df, waveform, waveform_df

# -----------------------------
# Block diagram generator
# -----------------------------
def generate_block_diagram(inputs, outputs):
    dot = Digraph()
    dot.node("Module", shape="box")
    for inp in inputs:
        dot.node(inp, shape="circle")
        dot.edge(inp, "Module")
    for outp in outputs:
        dot.node(outp, shape="circle")
        dot.edge("Module", outp)
    return dot

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Verilog Combinational Testbench Generator")

code_input = st.text_area("Enter Verilog Code", height=250)

if st.button("Generate Testbench & Analysis"):
    inputs = re.findall(r'input\s+([a-zA-Z0-9_]+)', code_input)
    outputs = re.findall(r'output\s+([a-zA-Z0-9_]+)', code_input)
    assigns = re.findall(r'assign\s+([a-zA-Z0-9_]+)\s*=\s*([^;]+);', code_input)

    if not inputs or not outputs or not assigns:
        st.warning("⚠️ Could not parse inputs/outputs properly.")
    else:
        # Truth table and waveform
        df, waveform, waveform_df = generate_truth_table(inputs, outputs, assigns)
        st.subheader("Truth Table")
        st.dataframe(df)

        st.subheader("Waveform Plot")
        fig = go.Figure()
        for sig in waveform_df.columns:
            fig.add_trace(go.Scatter(
                y=waveform_df[sig],
                name=sig,
                mode='lines+markers'
            ))
        fig.update_layout(yaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

        # Block diagram
        st.subheader("Block Diagram")
        dot = generate_block_diagram(inputs, outputs)
        st.graphviz_chart(dot)

        # Testbench generation
        st.subheader("Generated Testbench")
        testbench_lines = ["`timescale 1ns / 1ps", "module tb;"]
        for inp in inputs:
            testbench_lines.append(f"reg {inp};")
        for outp in outputs:
            testbench_lines.append(f"wire {outp};")
        module_match = re.search(r'module\s+([a-zA-Z0-9_]+)', code_input)
        if module_match:
            module_name = module_match.group(1)
            testbench_lines.append(f"{module_name} uut (.*);")
        testbench_lines.append("initial begin")
        import itertools
        for combo in itertools.product([0,1], repeat=len(inputs)):
            assign_lines = " ".join([f"{inp}={val};" for inp, val in zip(inputs, combo)])
            testbench_lines.append(f"#10 {assign_lines}")
        testbench_lines.append("end")
        testbench_lines.append("endmodule")
        st.code("\n".join(testbench_lines), language='verilog')

        # Explanation (fallback only, suppress Gemini warning)
        try:
            explanation = genai.models.generate_content(
                model="models/text-bison-001",
                prompt=f"Explain the following Verilog code in simple terms:\n{code_input}"
            ).content
        except:
            explanation = fallback_explanation(code_input)

        st.subheader("Explanation")
        st.text(explanation)
