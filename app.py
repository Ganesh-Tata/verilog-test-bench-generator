import streamlit as st
import google.generativeai as genai
import os
import re
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from graphviz import Digraph

# ===============================
# Gemini API setup
# ===============================
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
try:
    model = genai.GenerativeModel("gemini-pro")
except Exception:
    model = None

# ===============================
# Helper: Parse Verilog
# ===============================
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

# ===============================
# Generate truth table + waveforms
# ===============================
def generate_truth_table(inputs, outputs, assigns):
    rows = []
    waveform = {sig: [] for sig in inputs + outputs}
    for values in itertools.product([0, 1], repeat=len(inputs)):
        env = dict(zip(inputs, values))
        # Evaluate outputs from assigns
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

# ===============================
# Generate block diagram
# ===============================
def generate_block_diagram(inputs, outputs, assigns):
    dot = Digraph()
    dot.attr(rankdir='LR')

    for i in inputs:
        dot.node(i, shape="circle", color="lightblue", style="filled")

    for o in outputs:
        dot.node(o, shape="doublecircle", color="lightgreen", style="filled")

    for out, expr in assigns:
        dot.node(out + "_logic", expr, shape="box", color="orange", style="filled")
        for var in inputs:
            if var in expr:
                dot.edge(var, out + "_logic")
        dot.edge(out + "_logic", out)

    return dot

# ===============================
# Fallback explanation with step-by-step logic
# ===============================
def fallback_explanation(code, inputs, outputs, assigns):
    explanation = "This is a combinational Verilog module.\n\n"
    explanation += f"**Inputs:** {', '.join(inputs)}\n"
    explanation += f"**Outputs:** {', '.join(outputs)}\n\n"
    explanation += "**Step-by-step logic:**\n"

    for idx, (out, expr) in enumerate(assigns, 1):
        explanation += f"{idx}. Output `{out}` is calculated using the expression `{expr}`.\n"
        expr_desc = expr.replace("&", " AND ").replace("|", " OR ").replace("~", " NOT ")
        explanation += f"   - Interpreted as: {expr_desc}\n"
        involved_inputs = [i for i in inputs if i in expr]
        if involved_inputs:
            explanation += f"   - Depends on inputs: {', '.join(involved_inputs)}\n"
        explanation += "\n"

    explanation += ("The outputs are evaluated for all possible combinations of inputs. "
                    "The truth table and waveform graphs visualize the complete input-output behavior.")
    return explanation

# ===============================
# Fallback testbench
# ===============================
def fallback_testbench(inputs, outputs, assigns):
    tb = "module tb;\n"
    for i in inputs:
        tb += f"  reg {i};\n"
    for o in outputs:
        tb += f"  wire {o};\n"
    tb += "\n  // DUT instance\n"
    tb += f"  my_module uut ({', '.join(inputs + outputs)});\n\n"
    tb += "  initial begin\n"
    tb += "    $monitor($time, "
    tb += ", ".join([f'\" {i}=%b\" , {i}' for i in inputs + outputs])
    tb += ");\n"
    tb += "    " + "; ".join([f"{i}=0" for i in inputs]) + ";\n"
    tb += "    // Apply input combinations here\n"
    tb += "    #10;\n"
    tb += "    $finish;\n"
    tb += "  end\nendmodule\n"
    return tb

# ===============================
# Streamlit App
# ===============================
st.title("🔧 Verilog Testbench Generator with Simulation & Step-by-Step Explanation")

code_input = st.text_area("Paste your Verilog code here:")

if st.button("Generate"):
    try:
        inputs, outputs, assigns = parse_verilog(code_input)
        if not inputs or not outputs or not assigns:
            st.error("⚠️ Could not parse inputs/outputs properly.")
        else:
            # Truth table & waveforms
            df, waveform_df = generate_truth_table(inputs, outputs, assigns)
            st.subheader("📋 Truth Table")
            st.dataframe(df)

            st.subheader("📉 Input/Output Waveforms")
            fig, ax = plt.subplots(figsize=(10, 4))
            for sig in waveform_df.columns:
                ax.step(range(len(waveform_df)), waveform_df[sig], label=sig, where='mid')
            ax.set_yticks([0, 1])
            ax.legend()
            st.pyplot(fig)

            # Block diagram
            st.subheader("📐 Block Diagram")
            try:
                dot = generate_block_diagram(inputs, outputs, assigns)
                st.graphviz_chart(dot.source)
            except Exception as e:
                st.warning("⚠️ Block diagram could not be generated.")

            # Explanation (Gemini or fallback)
            st.subheader("📝 Explanation")
            explanation = None
            if model:
                try:
                    response = model.generate_content(f"Explain this Verilog code:\n{code_input}")
                    explanation = response.text
                except Exception:
                    explanation = None
            if not explanation:
                explanation = fallback_explanation(code_input, inputs, outputs, assigns)
            st.write(explanation)

            # Testbench (Gemini or fallback)
            st.subheader("🧪 Testbench Code")
            tb_code = None
            if model:
                try:
                    tb_code = model.generate_content(
                        f"Write a Verilog testbench for the following code:\n{code_input}"
                    ).text
                except Exception:
                    tb_code = None
            if not tb_code:
                tb_code = fallback_testbench(inputs, outputs, assigns)
            st.code(tb_code, language="verilog")

    except Exception as e:
        st.error(f"❌ Error: {e}")
