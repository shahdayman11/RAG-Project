import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

FASTAPI_URL = os.getenv(
    "FASTAPI_URL"
)

st.set_page_config(
    page_title="Python Documentation Assistant",
    page_icon="🐍",
    layout="wide"
)

st.title("🐍 Python Documentation Assistant")

st.write(
    "Ask questions about NumPy, Pandas, Scikit-learn, "
    "or general Python."
)
with st.sidebar:

    st.header("About")

    st.write(
        """
This assistant uses:

- FAISS for document retrieval
- BGE-small for embeddings
- Gemini for query routing
- Gemini for answer generation

The knowledge base contains:

- NumPy
- Pandas
- Scikit-learn
        """
    )

    st.divider()

    st.write(
        "Backend:"
    )

    st.code(
        FASTAPI_URL
    )

if "messages" not in st.session_state:

    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )
query = st.chat_input(
    "Ask a Python question..."
)

if query:

    # Display user message

    with st.chat_message(
        "user"
    ):

        st.markdown(
            query
        )
# Save user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )
    try:

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Thinking..."
            ):

                response = requests.post(
                    f"{FASTAPI_URL}/query",
                    json={
                        "query": query
                    },
                    timeout=120
                )

            if response.status_code != 200:

                st.error(
                    f"FastAPI returned status "
                    f"{response.status_code}"
                )

                st.code(
                    response.text
                )

            else:

                data = response.json()
                answer = data.get(
                    "answer",
                    "No answer returned."
                )
                st.markdown(
                    answer
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )
    except requests.exceptions.ConnectionError:

        with st.chat_message(
            "assistant"
        ):

            st.error(
                "Could not connect to the FastAPI backend."
            )

            st.write(
                "Make sure FastAPI is running on:"
            )

            st.code(
                FASTAPI_URL
            )
    except requests.exceptions.Timeout:

        with st.chat_message(
            "assistant"
        ):

            st.error(
                "The request took too long."
            )

            st.write(
                "The Gemini API or retrieval process "
                "did not respond within the timeout."
            )
    except Exception as e:

        with st.chat_message(
            "assistant"
        ):

            st.error(
                "An unexpected error occurred."
            )

            st.code(
                str(e)
            )
with st.sidebar:

    if st.button(
        "Clear conversation"
    ):
        st.session_state.messages = []
        st.rerun()