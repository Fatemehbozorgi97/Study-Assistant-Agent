import streamlit as st
with st.sidebar:

    st.header("📄 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Choose PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            filename = uploaded_file.name

            save_path = os.path.join(
                "data/raw",
                filename
            )

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Processing {filename}..."):

                num_chunks = ingest_pdf(
                    save_path,
                    filename
                )

            st.success(
                f"{filename} processed ({num_chunks} chunks)"
            )