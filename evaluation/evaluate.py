import json
import os
import sys
import time


# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT V2 RAG
# ============================================================

from app import (
    process_pdf,
    create_vector_stores,
    retrieve_multimodal,
    multimodal_rag
)


# ============================================================
# PATHS
# ============================================================

QUESTIONS_PATH = os.path.join(
    os.path.dirname(__file__),
    "questions.json"
)

PDF_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "RS.pdf"
)

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__),
    "results_v2.json"
)


# ============================================================
# LOAD QUESTIONS
# ============================================================

with open(
    QUESTIONS_PATH,
    "r",
    encoding="utf-8"
) as file:
    questions = json.load(file)


print("=" * 70)
print("MULTIMODAL RAG - V2 EVALUATION")
print("=" * 70)

print(f"\nQuestions loaded: {len(questions)}")
print(f"PDF: {PDF_PATH}")


# ============================================================
# PROCESS PDF ONCE
# ============================================================

print("\n" + "=" * 70)
print("PROCESSING PDF")
print("=" * 70)

processing_start = time.time()

(
    all_docs,
    embeddings_array,
    image_data_store
) = process_pdf(PDF_PATH)

processing_time = time.time() - processing_start

print(
    f"\nPDF processing time: "
    f"{processing_time:.2f} seconds"
)


# ============================================================
# CREATE SEPARATE V2 FAISS STORES
# ============================================================

print("\n" + "=" * 70)
print("CREATING SEPARATE FAISS STORES")
print("=" * 70)

vector_store_start = time.time()

(
    text_vector_store,
    image_vector_store
) = create_vector_stores(
    all_docs,
    embeddings_array
)

vector_store_time = time.time() - vector_store_start

print(
    f"\nVector store creation time: "
    f"{vector_store_time:.2f} seconds"
)


# ============================================================
# EVALUATE QUESTIONS
# ============================================================

results = []

print("\n" + "=" * 70)
print("RUNNING V2 QUESTIONS")
print("=" * 70)


for index, item in enumerate(
    questions,
    start=1
):

    question = item["question"]
    question_type = item["type"]

    print("\n" + "-" * 70)

    print(
        f"Question {index}/{len(questions)}"
    )

    print(
        f"Type: {question_type}"
    )

    print(
        f"Question: {question}"
    )


    # ========================================================
    # RETRIEVAL
    # ========================================================

    retrieval_start = time.time()

    retrieved_docs = retrieve_multimodal(
        question,
        text_vector_store,
        image_vector_store,
        text_k=5,
        image_k=5
    )

    retrieval_time = (
        time.time() - retrieval_start
    )


    # ========================================================
    # RECORD RETRIEVED SOURCES
    # ========================================================

    retrieved_sources = []

    for doc in retrieved_docs:

        metadata = doc.metadata

        retrieved_sources.append(
            {
                "page": metadata.get(
                    "page",
                    "unknown"
                ),
                "type": metadata.get(
                    "type",
                    "unknown"
                ),
                "image_id": metadata.get(
                    "image_id"
                )
            }
        )


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    generation_start = time.time()

    try:

        answer = multimodal_rag(
            question,
            text_vector_store,
            image_vector_store,
            image_data_store
        )

        generation_time = (
            time.time() - generation_start
        )

        error = None

    except Exception as e:

        answer = ""

        generation_time = (
            time.time() - generation_start
        )

        error = str(e)


    # ========================================================
    # STORE RESULT
    # ========================================================

    result = {
        "id": item["id"],
        "type": question_type,
        "question": question,
        "answer": answer,

        "retrieved_sources": retrieved_sources,

        "retrieval_time_seconds": round(
            retrieval_time,
            4
        ),

        "generation_time_seconds": round(
            generation_time,
            4
        ),

        "total_time_seconds": round(
            retrieval_time +
            generation_time,
            4
        ),

        "error": error
    }

    results.append(result)


    # ========================================================
    # PRINT QUESTION RESULT
    # ========================================================

    print(
        f"\nRetrieval time: "
        f"{retrieval_time:.2f}s"
    )

    print(
        f"Generation time: "
        f"{generation_time:.2f}s"
    )

    print("\nRetrieved sources:")

    for source in retrieved_sources:

        print(
            f"  Page {source['page']} "
            f"| {source['type']}"
            + (
                f" | {source['image_id']}"
                if source["image_id"]
                else ""
            )
        )


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# SUMMARY
# ============================================================

total_questions = len(results)

successful_questions = sum(
    1
    for result in results
    if result["error"] is None
)

failed_questions = (
    total_questions -
    successful_questions
)


average_retrieval_time = (
    sum(
        result["retrieval_time_seconds"]
        for result in results
    )
    / total_questions
)


average_generation_time = (
    sum(
        result["generation_time_seconds"]
        for result in results
    )
    / total_questions
)


average_total_time = (
    sum(
        result["total_time_seconds"]
        for result in results
    )
    / total_questions
)


# ============================================================
# IMAGE RETRIEVAL STATISTICS
# ============================================================

image_questions = [
    result
    for result in results
    if result["type"] in [
        "image",
        "multimodal"
    ]
]

image_retrieved_questions = sum(
    1
    for result in image_questions
    if any(
        source["type"] == "image"
        for source in result["retrieved_sources"]
    )
)


if image_questions:

    image_retrieval_rate = (
        image_retrieved_questions /
        len(image_questions)
    ) * 100

else:

    image_retrieval_rate = 0


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("V2 EVALUATION COMPLETE")
print("=" * 70)

print(
    f"\nTotal questions: "
    f"{total_questions}"
)

print(
    f"Successful: "
    f"{successful_questions}"
)

print(
    f"Failed: "
    f"{failed_questions}"
)

print(
    f"\nPDF processing time: "
    f"{processing_time:.2f}s"
)

print(
    f"Vector store creation time: "
    f"{vector_store_time:.2f}s"
)

print(
    f"Average retrieval time: "
    f"{average_retrieval_time:.4f}s"
)

print(
    f"Average generation time: "
    f"{average_generation_time:.2f}s"
)

print(
    f"Average total question time: "
    f"{average_total_time:.2f}s"
)

print(
    f"\nImage retrieval rate: "
    f"{image_retrieval_rate:.1f}%"
)

print(
    f"Image questions with retrieved images: "
    f"{image_retrieved_questions}/{len(image_questions)}"
)

print(
    f"\nResults saved to:"
)

print(
    RESULTS_PATH
)

print("=" * 70)