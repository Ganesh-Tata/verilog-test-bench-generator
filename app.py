import streamlit as st
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import re
from graphviz import Digraph

# -------- Utility: parse Verilog combinational code --------
def parse_verilog(code):
    # Find module inputs and outputs
    module_match = re.search(r"module\s+\w+\s*\((.*?)\);", code, re.S)
    ports = module_match.group(1) if module_match else ""
    inputs = re.findall(r"input\s+(\w+)", code)
    outputs = re.findall(r"output\s+(\w+)", code)
    assigns = re.findall(r"assign\s+(\w+)\s*=\s*(.*?);", code)

    return inputs, outputs, assigns

# -------- Utility: convert Verilog expr to Python expr --------
def verilog_to_python(expr):
    expr = expr.replace("&", " and ")
    expr = expr.replace("|", " or ")
    expr = expr.replace("^", " != ")
    expr = re.sub(r"~(\w+)", r"(not \1)", expr)
    return expr

# -------- Truth table + waveform generation --------
def generate_truth_table(inputs, outputs, assigns):
    table = []
    waveform = {sig: [] for sig in inputs + outputs}

    # Build Python eval environment
    assign_map = {out: verilog_to_python(expr) for out, expr in assigns}

    for combo in itertools.product([0,1], repeat=len(inputs)):
        env = dict(zip(inputs, combo))
        # evaluate each assign
        for out, expr in assign_map.items():
            try:
                env[out] = int(eval(expr, {}, env))
            except Exception:
                env[out] = "?"
        row = [env[i] for i in inputs] + [env[o] for o in outputs]
        table.append(row)

        # fill waveform
        for sig in inputs + outputs:
            waveform[sig].append(env.get(sig, 0))

    df = pd.DataFrame(table, columns=inputs+outputs)
    return df, waveform

# -------- Waveform plotting --------
def plot_waveform(waveform):
    plt.figure(figsize=(10, 5))
    y_offset = 0
    for sig, values in waveform.items():
        shifted = [v + y_offset for v in values]
        plt.step(range(len(values)), shifted, where="post", label=sig)
        y_offset += 2
    plt.yticks([])
    plt.xlabel("Time steps")
    plt.title("Input/Output Waveforms")
    plt.legend(loc="upper right")
    st.pyplot(plt)

# -------- Block diagram with Graphviz --------
def generate_block_diagram(inputs, outputs, assigns):
    dot = Digraph()
    dot.attr(rankdir="LR")

    for inp in inputs:
        dot.node(inp, shape="circle", style="filled", color="lightblue")

    for out, expr in assigns:
        gate = f"{out}_gate"
        dot.node(gate, label=expr, shape="box", style="filled", color="lightgray")
        for var in re.findall(r"\b\w+\b", expr):
            if var in inputs or var in outputs:
                dot.edge(var, gate)
        dot.edge(gate, out)

    for out in outputs:
        dot.node(out, shape="doublecircle", style="filled", color="lightgreen")

    st.graphviz_chart(dot)

# -------- Streamlit UI --------
st.title("🔧 Verilog Testbench + Logic Visualizer")

code_input = st.text_area("Paste your Verilog code:", height=200)

if st.button("Generate Testbench & Visuals"):
    if code_input.strip():
        inputs, outputs, assigns = parse_verilog(code_input)

        st.subheader("📌 Detected Ports")
        st.write("Inputs:", inputs)
        st.write("Outputs:", outputs)

        if not assigns:
            st.error("No assign statements found! Only combinational logic is supported.")
        else:
            st.subheader("📊 Truth Table")
            df, waveform = generate_truth_table(inputs, outputs, assigns)
            st.dataframe(df)

            st.subheader("📈 Waveforms")
            plot_waveform(waveform)

            st.subheader("📐 Block Diagram")
            generate_block_diagram(inputs, outputs, assigns)

    else:
        st.warning("Please enter some Verilog code.")
