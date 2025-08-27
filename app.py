import streamlit as st
import google.generativeai as genai
import re
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Configure Gemini API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")


# ----------------- PARSE VERILOG -----------------
def parse_verilog(verilog_code):
    """Extract inputs, outputs, and assigns from combinational Verilog code."""

    # 1. Find inputs
    inputs = []
    input_matches = re.findall(r'\binput\s+([^;]+);', verilog_code)
    for match in input_matches:
        signals = [sig.strip() for sig in match.replace("input", "").split(",")]
        inputs.extend(signals)

    # 2. Find outputs
    outputs = []
    output_matches = re.findall(r'\boutput\s+([^;]+);', verilog_code)
    for match in output_matches:
        signals = [sig.strip() for sig in match.replace("output", "").split(",")]
        outputs.extend(signals)

    # 3. Assign statements
    assigns = re.findall(r'assign\s+(\w+)\s*=\s*(.+?);', verilog_code)

    # Deduplicate
    inputs = list(dict.fromkeys(inputs))
    outputs = list(dict.fromkeys(outputs))

    return inputs, outputs, assigns


# ----------------- EVALUATE EXPRESSION -----------------
def evaluate_expression(expr, values):
    expr = expr.replace("&", " and ")
    expr = expr.replace("|", " or ")
    expr = expr.replace("~", " not ")
    expr = expr.replace("^", " != ")
    try:
        return int(eval(expr, {}, values))
    except Exception:
        return "?"


# ----------------- TRUTH TABLE & WAVEFORMS -----------------
def generate_truth_table(inputs, outputs, assigns):
    table = []
    waveform = {sig: [] for sig in inputs + outputs}

    for combo in itertools.product([0, 1], repeat=len(inputs)):
        values = dict(zip(inputs, combo))

        # compute each assign
        for out, expr in assigns:
            values[out] = evaluate_expression(expr, values)

        row = {**values}
        table.append(row)

        for sig in waveform:
            waveform[sig].append(values.get(sig, 0))

    df = pd.DataFrame(table)
    waveform_df = pd.DataFrame(waveform)

    return df, waveform, waveform_df


# ----------------- BLOCK DIAGRAM -----------------
def generate_block_diagram(inputs, outputs, assigns):
    G = nx.DiGraph()

    for inp in inputs:
        G.add_node(inp, color="lightgreen")

    for out in outputs:
        G.add_node(out, color="lightblue")

    for out, expr in assigns:
        gate_node = f"{out}_logic"
        G.add_node(gate_node, color="orange", label=expr)
        for inp in inputs:
            if re.search(rf"\b{inp}\b", expr):
                G.add_edge(inp, gate_node)
        G.add_edge(gate_node, out)

    return G


def plot_block_diagram(G):
    pos = nx.spring_layout(G, seed=42)
    colors = [G.nodes[n].get("color", "gray") for n in G.nodes]

    labels = {n: G.nodes[n].get("label", n) for n in G.nodes}

    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, labels=labels, node_color=colors,
            node_size=1500, font_size=10, font_weight="bold", arrows=True)
    st.pyplot(plt)


# ----------------- STREAMLIT APP -----------------
st.title("🔧 Verilog Testbench & Truth Table Generator")

code_input = st.text_area("Enter your Verilog code:")

if st.button("Generate"):
    if not code_input.strip():
        st.error("Please enter Verilog code.")
    else:
        inputs, outputs, assigns = parse_verilog(code_input)

        if not inputs or not outputs or not assigns:
            st.error("Could not parse inputs/outputs properly. Please check your Verilog code.")
        else:
            st.subheader("📥 Parsed Signals")
            st.write("**Inputs:**", inputs)
            st.write("**Outputs:**", outputs)
            st.write("**Assigns:**", assigns)

            # Truth Table
            df, waveform, waveform_df = generate_truth_table(inputs, outputs, assigns)
            st.subheader("📊 Truth Table")
            st.dataframe(df)

            # Waveforms
            st.subheader("📈 Waveforms (Inputs & Outputs)")
            fig, ax = plt.subplots(figsize=(8, 4))
            for i, sig in enumerate(waveform_df.columns):
                ax.step(range(len(waveform_df)), waveform_df[sig] + 2 * i, where="post", label=sig)
            ax.set_yticks([2 * i for i in range(len(waveform_df.columns))])
            ax.set_yticklabels(waveform_df.columns)
            ax.legend(loc="upper right")
            st.pyplot(fig)

            # Block Diagram
            st.subheader("🔲 Block Diagram")
            G = generate_block_diagram(inputs, outputs, assigns)
            plot_block_diagram(G)

            # Explanation using Gemini
            st.subheader("📖 Code Explanation")
            try:
                explanation = model.generate_content(
                    f"Explain the following Verilog code in simple terms:\n{code_input}"
                )
                st.write(explanation.text)
            except Exception as e:
                st.error(f"Explanation could not be generated: {e}")
