import os

import requests
import streamlit as st
from dotenv import load_dotenv


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

FASTAPI_URL = os.getenv("FASTAPI_URL")

if not FASTAPI_URL:
    raise ValueError(
        "FASTAPI_URL is not set in .env"
    )

# Remove accidental trailing slash
FASTAPI_URL = FASTAPI_URL.rstrip("/")


# ============================================================
# 2. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Programming RAG Tutor",
    page_icon="👩‍💻",
    layout="wide"
)


# ============================================================
# 3. TITLE
# ============================================================

st.title("👩‍💻 Programming RAG Tutor")

st.write(
    "Upload a screenshot of your code, ask a question, "
    "and get an AI-powered explanation using OCR + RAG."
)


# ============================================================
# 4. SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")

    st.write("FastAPI Server")

    st.code(FASTAPI_URL)

    st.info(
        "Make sure FastAPI is running before submitting."
    )


# ============================================================
# 5. INPUT
# ============================================================

st.subheader("1. Upload your code")

uploaded_file = st.file_uploader(
    "Upload a screenshot of your code",
    type=["png", "jpg", "jpeg"]
)


st.subheader("2. Ask your question")

question = st.text_input(
    "Question",
    placeholder="e.g. Why this error?"
)


# ============================================================
# 6. SUBMIT
# ============================================================

if st.button("🚀 Ask Tutor", type="primary"):

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if uploaded_file is None:
        st.warning("Please upload an image first.")

    elif not question.strip():
        st.warning("Please enter your question.")

    else:

        with st.spinner(
            "Processing image → OCR → Retrieval → AI Tutor..."
        ):

            try:

                # ====================================================
                # Prepare image
                # ====================================================

                files = {
                    "image": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }


                # ====================================================
                # Question
                # ====================================================

               
                data = {
                    "question": question.strip()
                }

                response = requests.post(
                    FASTAPI_URL,
                    data=data,
                    files=files,
                    timeout=180
                )
              



                # ====================================================
                # Check HTTP status
                # ====================================================

                if response.status_code != 200:

                    st.error(
                        f"❌ FastAPI returned HTTP "
                        f"{response.status_code}"
                    )

                    st.code(
                        response.text,
                        language="json"
                    )

                    st.stop()


                # ====================================================
                # Parse JSON
                # ====================================================

                result = response.json()


            # ========================================================
            # Connection error
            # ========================================================

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to FastAPI.\n\n"
                    "Make sure FastAPI is running on:\n"
                    f"{FASTAPI_URL}"
                )

                st.stop()


            # ========================================================
            # Timeout
            # ========================================================

            except requests.exceptions.Timeout:

                st.error(
                    "⏳ The request took too long.\n\n"
                    "OCR + retrieval + AI generation can take "
                    "some time. Please try again."
                )

                st.stop()


            # ========================================================
            # Invalid JSON
            # ========================================================

            except requests.exceptions.JSONDecodeError:

                st.error(
                    "❌ FastAPI did not return valid JSON."
                )

                st.code(
                    response.text,
                    language="text"
                )

                st.stop()


            # ========================================================
            # Other request errors
            # ========================================================

            except requests.exceptions.RequestException as e:

                st.error(
                    f"❌ Request failed:\n\n{str(e)}"
                )

                st.stop()


            # ========================================================
            # Unexpected error
            # ========================================================

            except Exception as e:

                st.error(
                    f"❌ Unexpected error:\n\n{str(e)}"
                )

                st.stop()


        # ============================================================
        # SUCCESS
        # ============================================================

        st.success("✅ Tutor response received!")


        # ============================================================
        # TUTOR ANSWER
        # ============================================================

        st.subheader("🎓 Tutor Answer")

        answer = result.get(
            "answer",
            "No answer returned."
        )

        st.markdown(answer)


        # ============================================================
        # OCR RESULTS
        # ============================================================

        with st.expander("🔍 OCR Results", expanded=False):

            st.markdown("### Raw OCR")

            raw_ocr = result.get(
                "raw_ocr",
                ""
            )

            st.code(
                raw_ocr,
                language="text"
            )


            st.markdown("### Validated OCR")

            validated_ocr = result.get(
                "validated_ocr",
                ""
            )

            st.code(
                validated_ocr,
                language="python"
            )


            st.write(
                "OCR Validation Model:",
                result.get(
                    "ocr_validation_model",
                    "Unknown"
                )
            )


        # ============================================================
        # RETRIEVED EXAMPLES
        # ============================================================

        with st.expander(
            "📚 Retrieved Examples",
            expanded=False
        ):

            examples = result.get(
                "retrieved_examples",
                []
            )

            if not examples:

                st.info(
                    "No examples were retrieved."
                )

            else:

                for i, example in enumerate(
                    examples,
                    start=1
                ):

                    st.markdown(
                        f"### Example {i}"
                    )


                    st.write(
                        f"**ID:** "
                        f"{example.get('ID', 'Unknown')}"
                    )


                    st.write(
                        f"**Distance:** "
                        f"{example.get('DISTANCE', 'Unknown')}"
                    )


                    if example.get("TOPIC"):

                        st.write(
                            f"**Topic:** "
                            f"{example.get('TOPIC')}"
                        )


                    if example.get("LIBRARY"):

                        st.write(
                            f"**Library:** "
                            f"{example.get('LIBRARY')}"
                        )


                    st.code(
                        example.get(
                            "TEXT",
                            ""
                        ),
                        language="python"
                    )


                    st.divider()


        # ============================================================
        # MODEL INFORMATION
        # ============================================================

        with st.expander(
            "🤖 Model Information",
            expanded=False
        ):

            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "OCR Validation",
                    result.get(
                        "ocr_validation_model",
                        "Unknown"
                    )
                )


            with col2:

                st.metric(
                    "Tutor Model",
                    result.get(
                        "tutor_model",
                        "Unknown"
                    )
                )

