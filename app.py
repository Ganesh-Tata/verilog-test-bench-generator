import streamlit as st
import google.generativeai as genai
import os
import re
import itertools
import pandas as pd
import plotly.graph_objects as go
from graphviz import Digraph

# ------------------- Gemini API Setup -------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
try:
    model = genai.GenerativeModel("gemini-pro")
except Exception:
    model = None

# ------------------- Verilog Parsing -------------------
def parse_verilog(code):
    inputs = re.findall(r'input\s+([\w, ]+);', code)
    outputs = re.findall(r'output\s+([\w, ]+);', code)
    assigns = re.findall(r'assign\s+(\w+)\s*=\s*(.*);', code)

    input_list = []
    for i in inputs:
        input_list.extend([x.strip() for x in i.split(",")])

    output_list = []
    for o in outputs:
        output_list.extend([x.strip() for x in o.split(",")])

    return input_list, output_list, assigns

# ------------------- Truth Table & Waveforms -------------------
def generate_truth_table(inputs, outputs, assigns):
    rows = []
    waveform = {sig: [] for sig in inputs + outputs}
    for values in itertools.product([0, 1], repeat=len(inputs)):
        env = dict(zip(inputs, values))
        for out, expr in assigns:
            expr_eval = expr
            for var in inputs:
                expr_eval = expr_eval.replace(var, str(env[var]))
            expr_eval = expr_eval.replace("~", "1-")
            expr_eval = expr_eval.replace("&", " and ")
            expr_eval = expr_eval.replace("|", " or ")
            try:
                env[out] = int(eval(expr_eval))
            except Exception:
                env[out] = "?"
        row = [env[i] for i in inputs] + [env[o] for o in outputs]
        rows.append(row)
        for sig in inputs + outputs:
            waveform[sig].append(env[sig])
    df = pd.DataFrame(rows, columns=inputs + outputs)
    waveform_df = pd.DataFrame(waveform)
    return df, waveform_df

# ------------------- Block Diagram -------------------
def generate_block_diagram(inputs, outputs, assigns):
    dot = Digraph(comment='Verilog Block Diagram')
    dot.attr(rankdir='LR', size='10')

    with dot.subgraph() as s:
        s.attr(rank='source')
        for i in inputs:
            s.node(i, shape='circle', style='filled', color='lightblue')

    with dot.subgraph() as s:
        s.attr(rank='sink')
        for o in outputs:
            s.node(o, shape='doublecircle', style='filled', color='lightgreen')

    for out, expr in assigns:
        logic_name = out + "_logic"
        dot.node(logic_name, label=expr, shape='box', style='filled', color='orange')
        for var in inputs:
            if var in expr:
                dot.edge(var, logic_name)
        dot.edge(logic_name, out)

    return dot

# ------------------- Step-by-step Explanation -------------------
def fallback_explanation(code, inputs, outputs, assigns):
    explanation = "This is a combinational Verilog module.\n\n"
    explanation += f"**Inputs:** {', '.join(inputs)}\n"
    explanation += f"**Outputs:** {', '.join(outputs)}\n\n"
    explanation += "**Step-by-step logic:**\n"
    for idx, (out, expr) in enumerate(assigns, 1):
        explanation += f"{idx}. Output `{out}` is calculated using `{expr}`.\n"
        expr_desc = expr.replace("&", " AND ").replace("|", " OR ").replace("~", " NOT ")
        explanation += f"   - Interpreted as: {expr_desc}\n"
        involved_inputs = [i for i in inputs if i in expr]
        if involved_inputs:
            explanation += f"   - Depends on inputs: {', '.join(involved_inputs)}\n"
        explanation += "\n"
    explanation += "Outputs are evaluated for all possible input combinations. Truth table and waveform graphs show complete behavior."
    return explanation

# ------------------- Testbench Fallback -------------------
def fallback_testbench(inputs, outputs, assigns):
    tb = "module tb;\n"
    for i in inputs:
        tb += f"  reg {i};\n"
    for o in outputs:
        tb += f"  wire {o};\n"
    tb += f"  my_module uut ({', '.join(inputs + outputs)});\n\n"
    tb += "  initial begin\n"
    tb += "    $monitor($time, " + ", ".join([f'\" {i}=%b\" , {i}' for i in inputs + outputs]) + ");\n"
    tb += "    " + "; ".join([f"{i}=0" for i in inputs]) + ";\n"
    tb += "    #10;\n    $finish;\n  end\nendmodule\n"
    return tb

# ------------------- Plotly Waveforms -------------------
def plot_waveforms_plotly(waveform_df):
    fig = go.Figure()
    n_signals = len(waveform_df.columns)
    for idx, sig in enumerate(waveform_df.columns):
        y = waveform_df[sig].replace("?", 0) + idx*2
        fig.add_trace(go.Scatter(y=y, x=list(range(len(y))),
                                 mode='lines', step='post', name=sig))
    fig.update_layout(
        height=200+50*n_signals,
        yaxis=dict(tickvals=[i*2+0.5 for i in range(n_signals)],
                   ticktext=list(waveform_df.columns)),
        xaxis_title="Input Combination Index",
        title="Digital Waveforms",
        showlegend=True
    )
    return fig

# ------------------- Streamlit App -------------------
st.title("🔧 Verilog Testbench Generator with Interactive Simulation")

code_input = st.text_area("Paste your Verilog code here:")

if st.button("Generate"):
    try:
        inputs, outputs, assigns = parse_verilog(code_input)
        if not inputs or not outputs or not assigns:
            st.error("⚠️ Could not parse inputs/outputs properly.")
        else:
            df, waveform_df = generate_truth_table(inputs, outputs, assigns)

            # Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Truth Table", "Waveforms", "Block Diagram", "Testbench", "Explanation"])

            with tab1:
                st.subheader("📋 Truth Table")
                st.dataframe(df)
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", data=csv, file_name="truth_table.csv", mime="text/csv")

            with tab2:
                st.subheader("📉 Input/Output Waveforms")
                fig = plot_waveforms_plotly(waveform_df)
                st.plotly_chart(fig)

            with tab3:
                st.subheader("📐 Block Diagram")
                try:
                    dot = generate_block_diagram(inputs, outputs, assigns)
                    st.graphviz_chart(dot.source)
                except Exception:
                    st.warning("⚠️ Block diagram could not be generated.")

            with tab4:
                st.subheader("🧪 Testbench Code")
                tb_code = None
                if model:
                    try:
                        tb_code = model.generate_content(f"Write a Verilog testbench for:\n{code_input}").text
                    except Exception:
                        tb_code = None
                if not tb_code:
                    tb_code = fallback_testbench(inputs, outputs, assigns)
                st.code(tb_code, language="verilog")

            with tab5:
                st.subheader("📝 Explanation")
                explanation = None
                if model:
                    try:
                        explanation = model.generate_content(f"Explain this Verilog code:\n{code_input}").text
                    except Exception:
                        explanation = None
                if not explanation:
                    explanation = fallback_explanation(code_input, inputs, outputs, assigns)
                st.write(explanation)

    except Exception as e:
        st.error(f"❌ Error: {e}")
