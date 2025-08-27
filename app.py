import streamlit as st
import google.generativeai as genai
import graphviz
import os

# ----------------------------
# Setup Gemini API
# ----------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ----------------------------
# Helper: Generate Block Diagram
# ----------------------------
def generate_block_diagram(verilog_code: str):
    """Very basic parser for ports & Graphviz diagram"""
    lines = verilog_code.splitlines()
    module_name = "unknown_module"
    inputs, outputs = [], []

    for line in lines:
        line = line.strip()
        if line.startswith("module"):
            module_name = line.split()[1].split("(")[0]
        elif line.startswith("input"):
            parts = line.replace(",", " ").replace(";", " ").split()
            inputs += [p for p in parts[1:] if p not in ["input"]]
        elif line.startswith("output"):
            parts = line.replace(",", " ").replace(";", " ").split()
            outputs += [p for p in parts[1:] if p not in ["output"]]

    dot = graphviz.Digraph()
    dot.attr(rankdir="LR", size="8")

    # Add module block
    dot.node("module", module_name, shape="box", style="filled", color="lightblue")

    # Add inputs
    for inp in inputs:
        dot.node(inp, inp, shape="ellipse", color="green")
        dot.edge(inp, "module")

    # Add outputs
    for out in outputs:
        dot.node(out, out, shape="ellipse", color="red")
        dot.edge("module", out)

    return dot


# ----------------------------
# Helper: Gemini Prompt
# ----------------------------
def ask_gemini(prompt: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error: {e}"


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Verilog Testbench Generator", layout="wide")
st.title("🖥️ Verilog Testbench Generator with Explanation")

st.markdown("Upload or paste your **Verilog module**, and this app will:")
st.markdown("""
- ✅ Explain the input Verilog code  
- ✅ Auto-generate a testbench  
- ✅ Explain the testbench  
- ✅ Draw a simple block diagram  
- ✅ Allow downloading the testbench  
""")

# Input
verilog_code = st.text_area("Paste your Verilog code here:", height=250)

if st.button("Generate Testbench & Explanation"):
    if not verilog_code.strip():
        st.warning("⚠️ Please provide Verilog code first.")
    else:
        # --- Explanation of Input Code ---
        with st.spinner("Explaining your Verilog code..."):
            explanation = ask_gemini(
                f"Explain the following Verilog code in detail:\n\n{verilog_code}"
            )
        st.subheader("📘 Explanation of Input Code")
        st.write(explanation)

        # --- Generate Testbench ---
        with st.spinner("Generating testbench..."):
            testbench = ask_gemini(
                f"Write a Verilog testbench for the following module:\n\n{verilog_code}"
            )
        st.subheader("🧪 Generated Testbench")
        st.code(testbench, language="verilog")

        # --- Explanation of Testbench ---
        with st.spinner("Explaining the testbench..."):
            tb_explanation = ask_gemini(
                f"Explain the following Verilog testbench step by step:\n\n{testbench}"
            )
        st.subheader("📖 Testbench Explanation")
        st.write(tb_explanation)

        # --- Block Diagram ---
        st.subheader("🔲 Block Diagram of Module")
        try:
            diagram = generate_block_diagram(verilog_code)
            st.graphviz_chart(diagram.source)
        except Exception as e:
            st.error(f"Could not generate diagram: {e}")

        # --- Download Button ---
        st.subheader("⬇️ Download Testbench")
        st.download_button(
            label="Download Testbench as .v file",
            data=testbench,
            file_name="generated_testbench.v",
            mime="text/plain",
        )
