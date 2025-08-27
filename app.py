import streamlit as st
import re
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import google.generativeai as genai

# ----------------------------
# Gemini API setup
# ----------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-pro")

# ----------------------------
# Helper: Parse inputs/outputs
# ----------------------------
def parse_ports(verilog_code):
    """Parse inputs and outputs from Verilog code."""
    inputs, outputs = [], []

    # --- Step 1: Capture port names inside module(...)
    module_match = re.search(r"module\s+\w+\s*\((.*?)\);", verilog_code, re.S)
    port_list = []
    if module_match:
        raw_ports = module_match.group(1).replace("\n", " ").split(",")
        port_list = [p.strip() for p in raw_ports if p.strip()]

    # --- Step 2: Capture declarations like 'input a, b;' or 'output y;'
    input_matches = re.findall(r"\binput\s+(?:wire|reg\s+)?([\w,\s]+);", verilog_code)
    output_matches = re.findall(r"\boutput\s+(?:wire|reg\s+)?([\w,\s]+);", verilog_code)

    for match in input_matches:
        for sig in match.split(","):
            sig = sig.strip()
            if sig:
                inputs.append(sig)

    for match in output_matches:
        for sig in match.split(","):
            sig = sig.strip()
            if sig:
                outputs.append(sig)

    # --- Step 3: If inline style was used (input a, output b in port list)
    for port in port_list:
        if port.startswith("input"):
            inputs.append(port.split()[-1])
        elif port.startswith("output"):
            outputs.append(port.split()[-1])

    return list(set(inputs)), list(set(outputs))

# ----------------------------
# Truth Table + Waveform Generator
# ----------------------------
def generate_truth_table(inputs, outputs, assigns):
    table = []
    waveform = {sig: [] for sig in inputs + outputs}

    for bits in itertools.product([0, 1], repeat=len(inputs)):
        env = dict(zip(inputs, bits))
        row = dict(env)

        for out in outputs:
            expr = assigns.get(out, "0")
            expr_eval = expr
            for k, v in env.items():
                expr_eval = expr_eval.replace(k, str(v))
            expr_eval = expr_eval.replace("&", " and ").replace("|", " or ").replace("~", " not ")
            try:
                row[out] = int(eval(expr_eval))
            except Exception:
                row[out] = "?"
        table.append(row)

        for k, v in row.items():
            waveform[k].append(v)

    df = pd.DataFrame(table)
    waveform_df = pd.DataFrame(waveform)
    return df, waveform, waveform_df

# ----------------------------
# Draw waveforms
# ----------------------------
def plot_waveforms(waveform_df):
    fig, ax = plt.subplots(figsize=(10, 4))
    time = range(len(waveform_df))
    for idx, col in enumerate(waveform_df.columns):
        ax.step(time, [v + 2 * idx for v in waveform_df[col]], where="post", label=col)
    ax.set_yticks([2 * i for i in range(len(waveform_df.columns))])
    ax.set_yticklabels(waveform_df.columns)
    ax.set_xlabel("Time")
    ax.set_title("Waveforms")
    ax.legend(loc="upper right")
    st.pyplot(fig)

# ----------------------------
# Block diagram generator
# ----------------------------
def generate_block_diagram(inputs, outputs):
    G = nx.DiGraph()
    for i in inputs:
        G.add_node(i, shape="box", color="lightblue")
        G.add_edge(i, "Logic")
    for o in outputs:
        G.add_node(o, shape="box", color="lightgreen")
        G.add_edge("Logic", o)

    pos = nx.spring_layout(G)
    colors = ["lightblue" if n in inputs else "lightgreen" if n in outputs else "lightgray" for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=2000, font_size=10, arrows=True)
    st.pyplot(plt)

# ----------------------------
# Testbench Generator
# ----------------------------
def generate_testbench(module_name, inputs, outputs):
    tb = f"module tb_{module_name};\n"
    for i in inputs:
        tb += f"  reg {i};\n"
    for o in outputs:
        tb += f"  wire {o};\n"
    tb += f"\n  {module_name} uut ({', '.join(inputs + outputs)});\n\n"
    tb += "  initial begin\n"
    tb += "    $dumpfile(\"waveform.vcd\");\n"
    tb += "    $dumpvars(0, tb_{module_name});\n"
    tb += "    // Apply test vectors\n"
    tb += "    #10;\n"
    for bits in itertools.product([0, 1], repeat=len(inputs)):
        assigns = " ".join([f"{i}={b};" for i, b in zip(inputs, bits)])
        tb += f"    {assigns} #10;\n"
    tb += "    $finish;\n"
    tb += "  end\nendmodule\n"
    return tb

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🔧 Verilog Testbench & Simulation Tool")

code_input = st.text_area("Enter your Verilog code:", height=250)

if st.button("Generate"):
    if not code_input.strip():
        st.warning("⚠️ Please paste some Verilog code.")
    else:
        # Parse inputs/outputs
        inputs, outputs = parse_ports(code_input)

        if not inputs or not outputs:
            st.error("⚠️ Could not parse inputs/outputs properly. Please check your Verilog code.")
        else:
            st.success(f"✅ Detected Inputs: {inputs}, Outputs: {outputs}")

            # Extract assigns
            assigns = {}
            for out in outputs:
                m = re.search(rf"assign\s+{out}\s*=\s*(.*?);", code_input)
                if m:
                    assigns[out] = m.group(1).strip()

            # Truth table + waveform
            df, waveform, waveform_df = generate_truth_table(inputs, outputs, assigns)
            st.subheader("Truth Table")
            st.dataframe(df)

            st.subheader("Waveforms")
            plot_waveforms(waveform_df)

            st.subheader("Block Diagram")
            generate_block_diagram(inputs, outputs)

            # Testbench
            module_name_match = re.search(r"module\s+(\w+)", code_input)
            module_name = module_name_match.group(1) if module_name_match else "my_module"
            tb_code = generate_testbench(module_name, inputs, outputs)
            st.subheader("Generated Testbench")
            st.code(tb_code, language="verilog")

            # Explanation (Gemini API)
            try:
                with st.spinner("Generating explanation..."):
                    explanation = model.generate_content(
                        f"Explain the following Verilog code in simple terms:\n{code_input}"
                    )
                st.subheader("Explanation")
                st.write(explanation.text)
            except Exception as e:
                st.error("⚠️ Explanation could not be generated due to API limits.")
