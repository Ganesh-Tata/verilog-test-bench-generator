import streamlit as st
import google.generativeai as genai
import os

# -----------------------------
# Configure Gemini API
# -----------------------------
api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
if not api_key:
    st.error("❌ Gemini API Key not found. Please set it in Streamlit secrets or as an environment variable.")
else:
    genai.configure(api_key=api_key)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Verilog Testbench Generator", page_icon="🔧", layout="wide")
st.title("🔧 Verilog Testbench Generator")

st.markdown("Paste your **Verilog module code** below and get an auto-generated **testbench** along with an **explanation**.")

# Input box for Verilog code
verilog_code = st.text_area("Enter Verilog Code:", height=250, placeholder="module and_gate(...); ... endmodule")

if st.button("Generate Testbench"):
    if not verilog_code.strip():
        st.warning("⚠️ Please enter some Verilog code first.")
    else:
        with st.spinner("Generating testbench... ⏳"):
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = f"""
                You are an expert in Verilog.
                Generate a complete Verilog testbench for the following module:

                {verilog_code}

                Requirements:
                - Include `timescale` directive.
                - Instantiate the module with correct port mapping.
                - Provide stimulus for inputs.
                - Add $monitor/$display for outputs.
                - Ensure compatibility with Icarus Verilog.
                - Also explain the testbench step by step.
                """

                response = model.generate_content(prompt)

                result = response.text

                # Try splitting into code + explanation
                if "```" in result:
                    parts = result.split("```")
                    tb_code = parts[1].replace("verilog", "").strip()
                    explanation = parts[-1].strip()
                else:
                    tb_code = result
                    explanation = "Explanation not clearly separated. Full output shown above."

                st.subheader("📜 Generated Testbench Code")
                st.code(tb_code, language="verilog")

                st.subheader("📝 Explanation")
                st.markdown(explanation)

                st.download_button("⬇️ Download Testbench", tb_code, file_name="tb_generated.v")

            except Exception as e:
                st.error(f"❌ Error: {e}")
