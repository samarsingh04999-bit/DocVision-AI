# ============================================================
# MULTIMODAL PDF RAG - BASELINE
#
# Based on the original multimodal RAG notebook
#
# Pipeline:
#
# PDF
#   ↓
# Extract Text + Images
#   ↓
# CLIP Text/Image Embeddings
#   ↓
# FAISS
#   ↓
# Retrieve Text + Images
#   ↓
# Gemini
#   ↓
# Answer
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import io
import base64

import pymupdf
import numpy as np
import torch

from PIL import Image
from dotenv import load_dotenv

from transformers import CLIPProcessor, CLIPModel

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file."
    )


# ============================================================
# 3. CONFIGURATION
# ============================================================

PDF_PATH = "data/sample.pdf"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K = 5

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# CHANGED FROM ORIGINAL NOTEBOOK:
# Original notebook used GPT-4.1.
# We are using Gemini because you don't have OpenAI API access.
GEMINI_MODEL_NAME = "gemini-3.5-flash-lite"


# ============================================================
# 4. LOAD CLIP MODEL
# ============================================================

print("\nLoading CLIP model...")

clip_model = CLIPModel.from_pretrained(
    CLIP_MODEL_NAME
)

clip_processor = CLIPProcessor.from_pretrained(
    CLIP_MODEL_NAME
)

clip_model.eval()

print("CLIP model loaded.")


# ============================================================
# 5. TEXT EMBEDDING
# ============================================================

def embed_text(text):
    """
    Generate the CLIP text embedding.

    CLIP text embeddings are projected into
    the shared 512-dimensional space.
    """

    inputs = clip_processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )

    with torch.no_grad():

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Instead of using get_text_features(), we explicitly
        # run the CLIP text encoder and then apply the
        # text projection.
        #
        # This avoids the Transformers-version issue you were
        # getting where get_text_features() returned a
        # BaseModelOutputWithPooling object.
        # ----------------------------------------------------

        text_outputs = clip_model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        pooled_output = text_outputs.pooler_output

        features = clip_model.text_projection(
            pooled_output
        )

        # Normalize CLIP embedding
        features = features / features.norm(
            dim=-1,
            keepdim=True
        )

    return features.squeeze(0).cpu().numpy().astype(
        np.float32
    )


# ============================================================
# 6. IMAGE EMBEDDING
# ============================================================

def embed_image(image_data):
    """
    Generate the CLIP image embedding.

    Image embeddings are projected into the same
    512-dimensional space as text embeddings.
    """

    if isinstance(image_data, str):

        image = Image.open(
            image_data
        ).convert("RGB")

    else:

        image = image_data

    inputs = clip_processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        # ----------------------------------------------------
        # Run CLIP vision encoder
        # ----------------------------------------------------

        vision_outputs = clip_model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        pooled_output = vision_outputs.pooler_output

        # Apply CLIP visual projection
        features = clip_model.visual_projection(
            pooled_output
        )

        # Normalize CLIP embedding
        features = features / features.norm(
            dim=-1,
            keepdim=True
        )

    return features.squeeze(0).cpu().numpy().astype(
        np.float32
    )


# ============================================================
# 7. PROCESS PDF
# ============================================================

def process_pdf(pdf_path):

    print(
        f"\nProcessing PDF: {pdf_path}"
    )

    if not os.path.exists(pdf_path):

        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    pdf = pymupdf.open(pdf_path)

    all_docs = []
    all_embeddings = []

    # Stores the actual images.
    # We will later send retrieved images to Gemini.
    image_data_store = {}

    # Same chunking idea as original notebook
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # ========================================================
    # LOOP THROUGH PDF PAGES
    # ========================================================

    for page_number, page in enumerate(pdf):

        print(
            f"Processing page {page_number + 1}..."
        )

        # ====================================================
        # TEXT EXTRACTION
        # ====================================================

        text = page.get_text()

        if text.strip():

            text_doc = Document(
                page_content=text,
                metadata={
                    "page": page_number + 1,
                    "type": "text"
                }
            )

            # Split page text into chunks
            text_chunks = splitter.split_documents(
                [text_doc]
            )

            for chunk in text_chunks:

                # Create CLIP text embedding
                embedding = embed_text(
                    chunk.page_content
                )

                all_embeddings.append(
                    embedding
                )

                all_docs.append(
                    chunk
                )

        # ====================================================
        # IMAGE EXTRACTION
        # ====================================================

        images = page.get_images(
            full=True
        )

        for image_index, image in enumerate(
            images
        ):

            try:

                xref = image[0]

                base_image = pdf.extract_image(
                    xref
                )

                image_bytes = base_image[
                    "image"
                ]

                # Convert image bytes to PIL
                pil_image = Image.open(
                    io.BytesIO(image_bytes)
                ).convert("RGB")

                # Create unique image ID
                image_id = (
                    f"page_{page_number + 1}"
                    f"_image_{image_index + 1}"
                )

                # ------------------------------------------------
                # Store image as base64
                # ------------------------------------------------

                buffered = io.BytesIO()

                pil_image.save(
                    buffered,
                    format="PNG"
                )

                image_base64 = base64.b64encode(
                    buffered.getvalue()
                ).decode("utf-8")

                image_data_store[
                    image_id
                ] = image_base64

                # ------------------------------------------------
                # Create CLIP image embedding
                # ------------------------------------------------

                embedding = embed_image(
                    pil_image
                )

                all_embeddings.append(
                    embedding
                )

                # ------------------------------------------------
                # Create document representing image
                # ------------------------------------------------

                image_doc = Document(
                    page_content=(
                        f"[Image: {image_id}]"
                    ),
                    metadata={
                        "page": page_number + 1,
                        "type": "image",
                        "image_id": image_id
                    }
                )

                all_docs.append(
                    image_doc
                )

            except Exception as e:

                print(
                    f"Could not process image "
                    f"{image_index + 1} "
                    f"on page "
                    f"{page_number + 1}: {e}"
                )

    pdf.close()

    # ========================================================
    # CONVERT EMBEDDINGS TO NUMPY MATRIX
    # ========================================================

    embeddings_array = np.stack(
        all_embeddings
    ).astype(np.float32)

    print(
        f"\nTotal documents: "
        f"{len(all_docs)}"
    )

    print(
        f"Total embeddings: "
        f"{len(embeddings_array)}"
    )

    print(
        f"Embedding dimension: "
        f"{embeddings_array.shape[1]}"
    )

    print(
        f"Total images: "
        f"{len(image_data_store)}"
    )

    return (
        all_docs,
        embeddings_array,
        image_data_store
    )


# ============================================================
# 8. CREATE FAISS VECTOR STORE
# ============================================================

def create_vector_store(
    all_docs,
    embeddings_array
):

    print(
        "\nCreating FAISS vector store..."
    )

    # FAISS receives:
    #
    # (text, embedding)
    #
    # for every document.

    text_embeddings = [
        (
            doc.page_content,
            embedding
        )
        for doc, embedding
        in zip(
            all_docs,
            embeddings_array
        )
    ]

    metadatas = [
        doc.metadata
        for doc in all_docs
    ]

    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=None,
        metadatas=metadatas
    )

    print(
        "FAISS vector store created."
    )

    return vector_store


# ============================================================
# 9. INITIALIZE GEMINI
# ============================================================

print(
    "\nLoading Gemini..."
)

# CHANGED FROM ORIGINAL NOTEBOOK:
# GPT-4.1 → Gemini

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    temperature=0
)

print(
    "Gemini loaded."
)


# ============================================================
# 10. RETRIEVAL
# ============================================================

def retrieve_multimodal(
    query,
    vector_store,
    k=TOP_K
):

    # Convert user's query into CLIP embedding

    query_embedding = embed_text(
        query
    )

    # Search FAISS

    results = (
        vector_store
        .similarity_search_by_vector(
            query_embedding,
            k=k
        )
    )

    return results


# ============================================================
# 11. CREATE GEMINI MULTIMODAL MESSAGE
# ============================================================

def create_multimodal_message(
    query,
    retrieved_docs,
    image_data_store
):

    content = []

    # ========================================================
    # USER QUESTION
    # ========================================================

    content.append({
        "type": "text",
        "text": (
            f"Question:\n{query}\n\n"
            "Use the retrieved information "
            "from the PDF to answer the question."
        )
    })

    # ========================================================
    # SEPARATE TEXT AND IMAGE RESULTS
    # ========================================================

    text_docs = [
        doc
        for doc in retrieved_docs
        if doc.metadata.get("type") == "text"
    ]

    image_docs = [
        doc
        for doc in retrieved_docs
        if doc.metadata.get("type") == "image"
    ]

    # ========================================================
    # ADD TEXT CONTEXT
    # ========================================================

    if text_docs:

        content.append({
            "type": "text",
            "text": "\nRetrieved text:\n"
        })

        for doc in text_docs:

            page = doc.metadata.get(
                "page",
                "unknown"
            )

            content.append({
                "type": "text",
                "text": (
                    f"\n[Page {page}]\n"
                    f"{doc.page_content}\n"
                )
            })

    # ========================================================
    # ADD RETRIEVED IMAGES
    # ========================================================

    for doc in image_docs:

        image_id = doc.metadata.get(
            "image_id"
        )

        page = doc.metadata.get(
            "page",
            "unknown"
        )

        if (
            image_id
            and image_id in image_data_store
        ):

            content.append({
                "type": "text",
                "text": (
                    f"\n[Image from page {page}]\n"
                )
            })

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": (
                        "data:image/png;base64,"
                        f"{image_data_store[image_id]}"
                    )
                }
            })

    # ========================================================
    # INSTRUCTION
    # ========================================================

    content.append({
        "type": "text",
        "text": (
            "\n\n"
            "Answer using only the retrieved "
            "information.\n"
            "If the retrieved information is "
            "not sufficient to answer the "
            "question, say so.\n"
            "Do not invent facts."
        )
    })

    return HumanMessage(
        content=content
    )


# ============================================================
# 12. RUN MULTIMODAL RAG
# ============================================================

def multimodal_rag(
    query,
    vector_store,
    image_data_store
):

    # ========================================================
    # RETRIEVE
    # ========================================================

    retrieved_docs = retrieve_multimodal(
        query,
        vector_store,
        TOP_K
    )

    # ========================================================
    # SHOW RETRIEVED RESULTS
    # ========================================================

    print(
        "\nRetrieved documents:"
    )

    for index, doc in enumerate(
        retrieved_docs,
        start=1
    ):

        doc_type = doc.metadata.get(
            "type",
            "unknown"
        )

        page = doc.metadata.get(
            "page",
            "unknown"
        )

        if doc_type == "text":

            preview = (
                doc.page_content
                .replace("\n", " ")
            )

            if len(preview) > 120:

                preview = (
                    preview[:120]
                    + "..."
                )

            print(
                f"{index}. "
                f"TEXT | "
                f"Page {page} | "
                f"{preview}"
            )

        elif doc_type == "image":

            image_id = doc.metadata.get(
                "image_id"
            )

            print(
                f"{index}. "
                f"IMAGE | "
                f"Page {page} | "
                f"{image_id}"
            )

    # ========================================================
    # CREATE MULTIMODAL MESSAGE
    # ========================================================

    message = create_multimodal_message(
        query,
        retrieved_docs,
        image_data_store
    )

    # ========================================================
    # SEND TO GEMINI
    # ========================================================

    response = llm.invoke(
    [message]
)

# CHANGED:
# Gemini may return the response as a list of content blocks.
# Extract only the actual text.

    if isinstance(response.content, list):

      text_parts = []

      for block in response.content:

        if isinstance(block, dict):

            if block.get("type") == "text":

                text_parts.append(
                    block.get("text", "")
                )

        elif isinstance(block, str):

            text_parts.append(block)

      return "\n".join(text_parts)

    return response.content


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("MULTIMODAL PDF RAG")
    print("=" * 70)

    # ========================================================
    # PROCESS PDF
    # ========================================================

    (
        all_docs,
        embeddings_array,
        image_data_store
    ) = process_pdf(
        PDF_PATH
    )

    # ========================================================
    # CREATE VECTOR STORE
    # ========================================================

    vector_store = create_vector_store(
        all_docs,
        embeddings_array
    )

    # ========================================================
    # SYSTEM READY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("RAG SYSTEM READY")
    print("=" * 70)

    print(
        "\nAsk questions about your PDF."
    )

    print(
        "Type 'exit' to stop."
    )

    # ========================================================
    # INTERACTIVE LOOP
    # ========================================================

    while True:

        query = input(
            "\nYou: "
        ).strip()

        if query.lower() == "exit":

            print(
                "\nExiting..."
            )

            break

        if not query:

            continue

        try:

            answer = multimodal_rag(
                query,
                vector_store,
                image_data_store
            )

            print(
                "\nGemini:"
            )

            print(answer)

        except Exception as e:

            print(
                "\nError:"
            )

            print(e)


# ============================================================
# 14. START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()