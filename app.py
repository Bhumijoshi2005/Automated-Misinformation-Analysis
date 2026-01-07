import streamlit as st
import joblib
import wikipedia

# Load model and vectorizer
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# --- Streamlit UI Layout ---
st.set_page_config(page_title="Automated Misinformation Analysis", layout="centered")

st.markdown("<h1 style='color:#2e86de;'> Fake News Generator & Detector</h1>", unsafe_allow_html=True)
st.markdown("Type or paste a news article below to check if it's real or fake.")

# Input Area
inputn = st.text_area("📰 Enter News Article", height=200)

# Function to show Wikipedia summary
def wiki_verify(text):
    try:
        # Use only the first 300 characters for Wikipedia search
        short_query = text[:300]
        result = wikipedia.search(short_query)
        if result:
            summary = wikipedia.summary(result[0], sentences=2)
            page_url = wikipedia.page(result[0]).url
            return summary, page_url
        else:
            return "No relevant Wikipedia page found.", None
    except Exception as e:
        return f"Error during Wikipedia search: {e}", None


# Check button
if st.button("🔍 Check News"):
    if inputn.strip():
        transform_input = vectorizer.transform([inputn])
        prediction = model.predict(transform_input)
        proba = model.predict_proba(transform_input)[0]

        st.markdown("---")
        if prediction[0] == 1:
            st.success("✅ This news is likely **Real**.")
            st.progress(int(proba[1] * 100))
        else:
            st.error("🚫 This news is likely **Fake**.")
            st.progress(int(proba[0] * 100))

            # 🔒 Prevention Panel
            st.markdown("### 🛡️ Prevention Tips")
            st.markdown("""
                - ❗ Double-check headlines before sharing.
                - 🔍 Search the same news on trusted sources:
                  - [BBC](https://www.bbc.com/news)
                  - [Reuters](https://www.reuters.com)
                  - [FactCheck.org](https://www.factcheck.org)
                - 📚 Learn to recognize clickbait and biased language.
            """)

            # Wikipedia Summary
            st.markdown("### 📘 Wikipedia Fact Check")
            summary, url = wiki_verify(inputn)
            st.write(summary)
            if url:
                st.markdown(f"[🔗 Read more on Wikipedia]({url})")
    else:
        st.warning("⚠️ Please enter some text first.")

