import streamlit as st
import tempfile
import os

# Import our existing RAG functions
from app import (
    process_pdf,
    create_vector_store,
    multimodal_rag
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multimodal PDF RAG",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📄 Multimodal PDF RAG")

st.write(
    "Upload a PDF and ask questions about its "
    "text, images, charts and other content."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    process_button = st.button(
        "Process PDF",
        use_container_width=True
    )


# ============================================================
# SESSION STATE
# ============================================================

if "vector_store" not in st.session_state:

    st.session_state.vector_store = None


if "image_data_store" not in st.session_state:

    st.session_state.image_data_store = {}


if "processed" not in st.session_state:

    st.session_state.processed = False


# ============================================================
# PROCESS PDF
# ============================================================

if process_button:

    if uploaded_file is None:

        st.warning(
            "Please upload a PDF first."
        )

    else:

        with st.spinner(
            "Processing PDF..."
        ):

            # ------------------------------------------------
            # Save uploaded PDF temporarily
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_pdf_path = (
                    temp_file.name
                )

            try:

                # ------------------------------------------------
                # EXISTING RAG PIPELINE
                # ------------------------------------------------

                (
                    all_docs,
                    embeddings_array,
                    image_data_store
                ) = process_pdf(
                    temp_pdf_path
                )

                # ------------------------------------------------
                # CREATE FAISS INDEX
                # ------------------------------------------------

                vector_store = (
                    create_vector_store(
                        all_docs,
                        embeddings_array
                    )
                )

                # ------------------------------------------------
                # STORE IN STREAMLIT SESSION
                # ------------------------------------------------

                st.session_state.vector_store = (
                    vector_store
                )

                st.session_state.image_data_store = (
                    image_data_store
                )

                st.session_state.processed = True

                st.success(
                    "PDF processed successfully!"
                )

            finally:

                # ------------------------------------------------
                # Delete temporary PDF
                # ------------------------------------------------

                if os.path.exists(
                    temp_pdf_path
                ):

                    os.remove(
                        temp_pdf_path
                    )


# ============================================================
# SHOW PROCESSING STATUS
# ============================================================

if st.session_state.processed:

    st.success(
        "✅ Document ready for questions."
    )


# ============================================================
# QUESTION INPUT
# ============================================================

if st.session_state.processed:

    st.divider()

    st.subheader(
        "💬 Ask a question"
    )

    query = st.text_input(
        "Question",
        placeholder=(
            "e.g. What does the chart on page 1 show?"
        )
    )

    ask_button = st.button(
        "Ask",
        type="primary"
    )

    # ========================================================
    # ASK QUESTION
    # ========================================================

    if ask_button:

        if not query.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching document and generating answer..."
            ):

                try:

                    answer = multimodal_rag(
                        query,
                        st.session_state.vector_store,
                        st.session_state.image_data_store
                    )

                    st.subheader(
                        "🤖 Answer"
                    )

                    st.write(
                        answer
                    )

                except Exception as e:

                    st.error(
                        f"Error: {e}"
                    )


# ============================================================
# INITIAL MESSAGE
# ============================================================

else:

    st.info(
        "👈 Upload a PDF from the sidebar "
        "and click **Process PDF** to begin."
    )