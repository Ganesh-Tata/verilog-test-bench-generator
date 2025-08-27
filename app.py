import streamlit as st
import google.generativeai as genai
import re
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import graphviz

# ---------------- GEMINI SETUP ----------------
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-pro")
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# ---------------- PARSER ----------------
def parse_verilog(verilog_code):
    inputs = re.findall(r'input\s+([\w\s,]+);', verilog_code)
    outputs = re.findall(r'output\s+([\w\s,]+);', verilog_code)
    assigns = re.findall(r'assign\s+(.*?);', verilog_code)

    input_signals = [s.strip() for line in inputs for s in line.split(",")]
    output_signals = [s.strip() for line in outputs for s in line.split(",")]

    return input_signals, output_signals, assigns

# ---------------- TRUTH TABLE + WAVEFORMS ----------------
def generate_truth_table(inputs, outputs, assigns):
    table = []
    waveform = {sig: [] for sig in inputs + outputs}

    for bits in itertools.product([0, 1], repeat=len(inputs)):
        row = dict(zip(inputs, bits))
        expr_context = row.copy()

        for assign in assigns:
            lhs, rhs = assign.split("=")
            lhs, rhs = lhs.strip(), rhs.strip()
            rhs_eval = rhs.replace("&", " and ").replace("|", " or ").replace("~", " not ")

            try:
                expr_context[lhs] = int(eval(rhs_eval, {}, expr_context))
            except:
                expr_context[lhs] = "?"

        row.update({out: expr_context.get(out, "?") for out in outputs})
        table.append(row)

        for sig in inputs + outputs:
            waveform[sig].append(row.get(sig, "?"))

    df = pd.DataFrame(table)
    waveform_df = pd.DataFrame(waveform)
    return df, waveform_df

# ---------------- BLOCK DIAGRAM ----------------
def generate_block_diagram(inputs, outputs, assigns):
    dot = graphviz.Digraph()
    for inp in inputs:
        dot.node(inp, shape="circle")
    for out in outputs:
        dot.node(out, shape="doublecircle")
    for i, assign in enumerate(assigns):
        lhs, rhs = assign.split("=")
        lhs, rhs = lhs.strip(), rhs.strip()
        node_name = f"gate{i}"
        dot.node(node_name, label=rhs, shape="box")
        for inp in inputs:
            if inp in rhs:
                dot.edge(inp, node_name)
        dot.edge(node_name, lhs)
    return dot

# ---------------- FALLBACK EXPLANATION ----------------
def local_explanation(inputs, outputs, assigns):
    explanation = "This Verilog module describes a combinational circuit.\n\n"
    explanation += f"- Inputs: {', '.join(inputs)}\n"
    explanation += f"- Outputs: {', '.join(outputs)}\n"
    explanation += "- Logic:\n"
    for assign in assigns:
        lhs, rhs = assign.split("=")
        explanation += f"  • {lhs.strip()} = {rhs.strip()}\n"
    return explanation

def local_testbench(inputs, outputs):
    tb = "module testbench;\n"
    tb += "  reg " + ", ".join(inputs) + ";\n"
    tb += "  wire " + ", ".join(outputs) + ";\n\n"
    tb += "  my_module uut (" + ", ".join(inputs + outputs) + ");\n\n"
    tb += "  initial begin\n"
    tb += "    $monitor($time, "
    tb += ", ".join([f'\" {i}=%b\" , {i}' for i in inputs + outputs])
    tb += ");\n"
    tb += "    " + "; ".join([f"{i}=0" for i in inputs]) + ";\n"
    tb += "    #10;\n"
    tb += "    // Add more stimulus here\n"
    tb += "    $finish;\n"
    tb += "  end\nendmodule\n"
    return tb

# ---------------- STREAMLIT APP ----------------
st.title("🔧 Verilog Testbench Generator")

code_input = st.text_area("Paste your Verilog code here:")

if st.button("Generate"):
    if code_input.strip():
        inputs, outputs, assigns = parse_verilog(code_input)

        if not inputs or not outputs:
            st.error("⚠️ Could not parse inputs/outputs properly.")
        else:
            # Truth table + waveforms
            df, waveform_df = generate_truth_table(inputs, outputs, assigns)
            st.subheader("📑 Truth Table")
            st.dataframe(df)

            st.subheader("📈 Waveforms")
            fig, ax = plt.subplots(figsize=(8, 4))
            for idx, sig in enumerate(waveform_df.columns):
                ax.step(range(len(waveform_df)), waveform_df[sig].replace("?", 0) + 2 * idx, where="mid", label=sig)
            ax.set_yticks([2 * i for i in range(len(waveform_df.columns))])
            ax.set_yticklabels(waveform_df.columns)
            ax.set_xlabel("Input Combination Index")
            ax.set_title("Digital Waveforms")
            ax.legend()
            st.pyplot(fig)

            # Block diagram
            st.subheader("📦 Block Diagram")
            try:
                st.graphviz_chart(generate_block_diagram(inputs, outputs, assigns))
            except Exception as e:
                st.warning(f"Block diagram could not be generated: {e}")

            # Explanation
            st.subheader("📖 Explanation")
            if GEMINI_AVAILABLE:
                try:
                    with st.spinner("Generating explanation..."):
                        explanation = model.generate_content(
                            f"Explain the following Verilog code in simple terms:\n{code_input}"
                        )
                    st.write(explanation.text)
                except:
                    st.info("⚠️ Gemini API not available, using fallback explanation.")
                    st.write(local_explanation(inputs, outputs, assigns))
            else:
                st.write(local_explanation(inputs, outputs, assigns))

            # Testbench
            st.subheader("🧪 Testbench Code")
            if GEMINI_AVAILABLE:
                try:
                    with st.spinner("Generating testbench..."):
                        tb_code = model.generate_content(
                            f"Generate a Verilog testbench for the following code:\n{code_input}"
                        )
                    st.code(tb_code.text, language="verilog")
                except:
                    st.info("⚠️ Gemini API not available, using fallback testbench.")
                    st.code(local_testbench(inputs, outputs), language="verilog")
            else:
                st.code(local_testbench(inputs, outputs), language="verilog")
