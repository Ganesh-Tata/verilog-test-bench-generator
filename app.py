import streamlit as st
import google.generativeai as genai
import re
import itertools
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- GEMINI SETUP ----------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# ---------------- HELPERS ----------------
def parse_verilog(verilog_code):
    """Extract inputs, outputs, and assigns from simple combinational Verilog code."""
    inputs = re.findall(r'input\s+([\w, ]+);', verilog_code)
    outputs = re.findall(r'output\s+([\w, ]+);', verilog_code)
    assigns = re.findall(r'assign\s+(\w+)\s*=\s*(.+);', verilog_code)

    inputs = [i.strip() for group in inputs for i in group.split(",")]
    outputs = [o.strip() for group in outputs for o in group.split(",")]
    return inputs, outputs, assigns


def verilog_to_python(expr):
    """Convert simple Verilog operators to Python equivalents."""
    expr = expr.replace("~", "1-")   # NOT
    expr = expr.replace("&", " and ")
    expr = expr.replace("|", " or ")
    expr = expr.replace("^", " ^ ")
    return expr


def generate_truth_table(inputs, outputs, assigns):
    """Generate truth table and waveform data for all input combinations."""
    table = []
    waveform = {sig: [] for sig in inputs + outputs}
    assign_map = {out: verilog_to_python(expr) for out, expr in assigns}

    for idx, combo in enumerate(itertools.product([0,1], repeat=len(inputs)), start=1):
        env = dict(zip(inputs, combo))

        # compute all outputs
        for out in outputs:
            expr = assign_map.get(out, "0")   # default = 0
            try:
                env[out] = int(eval(expr, {}, env))
            except Exception:
                env[out] = 0

        row = [idx] + [env[i] for i in inputs] + ["|"] + [env[o] for o in outputs]
        table.append(row)

        # update waveforms
        for sig in inputs + outputs:
            waveform[sig].append(env.get(sig, 0))

    columns = ["#"] + inputs + ["|"] + outputs
    df = pd.DataFrame(table, columns=columns)
    return df, waveform


def plot_waveforms(waveform, inputs, outputs):
    """Draw digital waveforms with inputs on top, outputs below."""
    total_signals = len(inputs) + len(outputs)
    fig, axes = plt.subplots(total_signals, 1, figsize=(8, 1.2*total_signals), sharex=True)

    if total_signals == 1:
        axes = [axes]

    time = list(range(len(next(iter(waveform.values())))))

    # Plot inputs first
    idx = 0
    for sig in inputs:
        axes[idx].step(time, waveform[sig], where="post", label=sig)
        axes[idx].set_ylim(-0.5, 1.5)
        axes[idx].set_yticks([0,1])
        axes[idx].legend(loc="upper right")
        idx += 1

    # Then outputs
    for sig in outputs:
        axes[idx].step(time, waveform[sig], where="post", label=sig, color="r")
        axes[idx].set_ylim(-0.5, 1.5)
        axes[idx].set_yticks([0,1])
        axes[idx].legend(loc="upper right")
        idx += 1

    plt.xlabel("Time steps")
    plt.tight_layout()
    return fig

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="Verilog Testbench Generator", layout="wide")
st.title("🔧 Verilog Testbench & Analyzer (Combinational)")

code_input = st.text_area("Enter your Verilog code here:", height=200)

if st.button("Generate Testbench & Analysis"):
    if not code_input.strip():
        st.error("Please enter some Verilog code.")
    else:
        # --- 1. Explain Verilog code ---
        with st.spinner("Explaining code..."):
            explanation = model.generate_content(
                f"Explain the following Verilog code in simple terms:\n{code_input}"
            ).text
        st.subheader("📘 Code Explanation")
        st.write(explanation)

        # --- 2. Parse Verilog ---
        inputs, outputs, assigns = parse_verilog(code_input)

        # --- 3. Generate Testbench (AI) ---
        with st.spinner("Generating testbench..."):
            tb_code = model.generate_content(
                f"Write a Verilog testbench for this code:\n{code_input}"
            ).text
        st.subheader("📝 Generated Testbench")
        st.code(tb_code, language="verilog")

        # --- 4. Truth Table & Waveforms ---
        if inputs and outputs:
            st.subheader("📊 Truth Table")
            df, waveform = generate_truth_table(inputs, outputs, assigns)
            st.dataframe(df.style.set_properties(**{'text-align': 'center'}))

            st.subheader("📈 Waveform Visualization")
            fig = plot_waveforms(waveform, inputs, outputs)
            st.pyplot(fig)
        else:
            st.warning("⚠️ Could not parse inputs/outputs properly.")
