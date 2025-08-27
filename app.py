import streamlit as st
import google.generativeai as genai
import graphviz
import matplotlib.pyplot as plt
import os
import re
from fpdf import FPDF

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Verilog Testbench Generator", layout="wide")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def extract_ports(verilog_code):
    """Extract input/output ports from Verilog code"""
    ports = []
    pattern = r"(input|output)\s+(?:\[\d+:\d+\]\s*)?(\w+)"
    for match in re.finditer(pattern, verilog_code):
        ports.append({"Direction": match.group(1), "Name": match.group(2)})
    return ports

def extract_parameters(verilog_code):
    """Extract parameter values"""
    params = []
    pattern = r"parameter\s+(\w+)\s*=\s*(\d+)"
    for match in re.finditer(pattern, verilog_code):
        params.append({"Parameter": match.group(1), "Value": match.group(2)})
    return params

def detect_fsm(verilog_code):
    """Detect FSM from case statements"""
    states = re.findall(r"(\w+):", verilog_code)
    if not states:
        return None
    dot = graphviz.Digraph()
    for i, state in enumerate(states):
        dot.node(state, state, shape="circle")
        if i < len(states) - 1:
            dot.edge(state, states[i+1])
    return dot

def plot_mock_waveform():
    """Simple waveform for clk & reset"""
    t = list(range(20))
    clk = [i % 2 for i in t]
    rst = [1 if i < 3 else 0 for i in t]

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.step(t, clk, where="mid", label="clk")
    ax.step(t, rst, where="mid", label="reset")
    ax.set_ylim(-0.5, 1.5)
    ax.legend()
    ax.set_title("Mock Waveform")
    return fig

def generate_testbench(verilog_code, style="Basic"):
    """Use Gemini to generate testbench"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Generate a {style} testbench for the following Verilog code and explain it:\n\n{verilog_code}"
    response = model.generate_content(prompt)
    return response.text

def save_to_pdf(verilog_code, explanation, testbench_code):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.multi_cell(0, 10, "Verilog Testbench Generator Report\n\n")
    pdf.multi_cell(0, 10, "Original Code:\n" + verilog_code + "\n\n")
    pdf.multi_cell(0, 10, "Explanation:\n" + explanation + "\n\n")
    pdf.multi_cell(0, 10, "Generated Testbench:\n" + testbench_code + "\n\n")

    pdf.output("verilog_report.pdf")
    return "verilog_report.pdf"

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("⚡ Verilog Testbench Generator with AI")
st.write("Upload or paste Verilog code to generate testbenches, diagrams, and explanations.")

verilog_code = st.text_area("✍️ Enter your Verilog Code:", height=200)

testbench_style = st.selectbox(
    "Select Testbench Style",
    ["Basic", "Self-checking", "Random stimulus"]
)

if st.button("🚀 Generate Testbench"):
    if verilog_code.strip() == "":
        st.warning("Please enter Verilog code first!")
    else:
        # Port table
        ports = extract_ports(verilog_code)
        if ports:
            st.subheader("🔌 Port Summary")
            st.table(ports)

        # Parameters
        params = extract_parameters(verilog_code)
        if params:
            st.subheader("⚙️ Parameters")
            st.table(params)

        # Testbench generation
        tb_output = generate_testbench(verilog_code, testbench_style)
        st.subheader("📝 Generated Testbench")
        st.code(tb_output, language="verilog")

        # FSM diagram
        fsm = detect_fsm(verilog_code)
        if fsm:
            st.subheader("🔄 FSM Diagram")
            st.graphviz_chart(fsm)

        # Mock waveform
        st.subheader("📉 Mock Waveform Example")
        st.pyplot(plot_mock_waveform())

        # Explanation (from AI output)
        st.subheader("📖 Explanation")
        st.write(tb_output)

        # Export option
        if st.button("💾 Export Report (PDF)"):
            pdf_file = save_to_pdf(verilog_code, tb_output, tb_output)
            with open(pdf_file, "rb") as f:
                st.download_button("Download PDF", f, file_name="verilog_report.pdf")

