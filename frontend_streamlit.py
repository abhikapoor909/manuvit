import streamlit as st
import requests
import os
import base64 # Import the base64 module

# --- Configuration ---
FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://localhost:8000")
ASSISTANT_NAME = "Manu"
ASSISTANT_DEVELOPER = os.environ.get("ASSISTANT_DEVELOPER", "Xibotix Pvt Lim")

# --- Image Paths ---


from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH= BASE_DIR / "vitdata.pdf"
VIT_LOGO_PATH = BASE_DIR / "vit.png"  # Make sure this file exists in the same directory
XIBOTIX_LOGO_PATH = BASE_DIR /"xibotix.png" # Make sure this file exists in the same directory

# --- URLs for Clickable Logos ---
VIT_URL = "https://vit.ac.in/"
XIBOTIX_URL = "https://www.xibotix.com/index.html#"

# --- Helper function to encode image to base64 ---
def img_to_base64(image_path):
    """Reads an image file and returns its base64 encoded string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        st.error(f"Error: Image file not found at '{image_path}'.")
        st.stop()
        return None
    except Exception as e:
        st.error(f"Error encoding image '{image_path}': {e}")
        st.stop()
        return None


# --- Streamlit App ---

st.set_page_config(
    page_title=ASSISTANT_NAME, # Set a simple title for the browser tab
    layout="wide", # Use wide layout to utilize full page width
    initial_sidebar_state="collapsed" # Keep sidebar collapsed initially
)

# --- Check and Encode Logos ---
vit_img_base64 = img_to_base64(VIT_LOGO_PATH)
xibotix_img_base64 = img_to_base64(XIBOTIX_LOGO_PATH)

# If either image failed to load/encode, img_to_base64 would stop the app.
# If we reach here, both are available.

# --- Header Layout (Columns for left logo, center title, right logo/text) ---
# Adjust column ratios to control spacing and placement
# [Left Column Width, Center Column Width, Right Column Width]
# Experiment with these numbers (e.g., [2, 6, 3], [3, 5, 3])
col_vit, col_title, col_xibotix = st.columns([2, 5, 3]) # Adjusted ratios slightly based on common layouts

with col_vit:
    # VIT Logo on the left, made clickable using HTML markdown
    vit_html = f"""
    <a href="{VIT_URL}" target="_blank" style="display: inline-block; margin-top: 10px; margin-bottom: 10px;">
        <img src="data:image/png;base64,{vit_img_base64}" style="max-width: 120px; height: auto;">
    </a>
    """
    st.markdown(vit_html, unsafe_allow_html=True)


with col_title:
    # Main title in the central column
    # Use CSS to ensure centering within the column
    st.markdown(f"<h1 style='text-align: center;'>{ASSISTANT_NAME}</h1>", unsafe_allow_html=True)


with col_xibotix:
    # Xibotix logo and developer text in the top right column
    # Use a div with right-alignment for both elements
    xibotix_html = f"""
    <div style='text-align: right;'>
        <a href="{XIBOTIX_URL}" target="_blank" style="display: inline-block; margin-top: 10px;">
            <img src="data:image/png;base64,{xibotix_img_base64}" style="max-width: 80px; height: auto;">
        </a>
        <div style='font-size: small;'>Developed by {ASSISTANT_DEVELOPER}</div>
    </div>
    """
    st.markdown(xibotix_html, unsafe_allow_html=True)

# Add a visual separator below the header
st.divider()

# --- Chat Interface ---
# (This part remains largely the same as it's the core functionality)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Optional: Add an initial greeting message
    # st.session_state.messages.append({"role": "assistant", "content": f"Hello! I am {ASSISTANT_NAME}, ready to help you find information from the document."})


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input(f"Ask {ASSISTANT_NAME} about the document..."):
    # Add user message to chat history immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Re-run the app to display the user message immediately
    st.rerun() # Use st.rerun() to refresh the page state and show the new user message

# --- This block runs AFTER st.rerun() but before the next user input ---
# It processes the *last* message added (which should be the user's prompt)
# and adds the assistant's response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_query = st.session_state.messages[-1]["content"]

    # Get assistant response from FastAPI backend
    try:
        # Display a placeholder for the assistant's response while waiting
        with st.chat_message("assistant"):
            with st.spinner(f"Asking {ASSISTANT_NAME}..."):
                 response = requests.post(
                    f"{FASTAPI_URL}/chat",
                    json={"query": user_query}
                 )
                 response.raise_for_status() # Raise HTTPError for bad responses

                 answer_data = response.json()
                 assistant_response = answer_data.get("answer", "Error: Received empty answer from server.")

        # Add SUCCESSFUL assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        st.rerun() # Re-run again to display the new assistant message


    except requests.exceptions.ConnectionError:
        error_message = "Error: Could not connect to the FastAPI server. Is it running?"
        st.error(error_message) # Display prominent error box outside chat
        # Add a simplified error message to chat history
        st.session_state.messages.append({"role": "assistant", "content": "Error: Could not connect to the server. Please ensure the backend is running."})
        st.rerun() # Re-run to show the error in chat history


    except requests.exceptions.RequestException as e:
        error_message = f"Error from server: {e}"
        st.error(error_message) # Display prominent error box outside chat
        # Add a simplified error to chat history
        st.session_state.messages.append({"role": "assistant", "content": f"Error processing request on server: {e}"})
        st.rerun() # Re-run to show the error in chat history

    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        st.error(error_message) # Display prominent error box outside chat
        # Add a simplified error to chat history
        st.session_state.messages.append({"role": "assistant", "content": f"An unexpected error occurred: {e}"})
        st.rerun() # Re-run to show the error in chat history


# Optional: Info message about the backend
st.sidebar.info("Ensure the FastAPI server is running before using the chat.")