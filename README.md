import streamlit as st

# Session state to track button clicks
if "feedback" not in st.session_state:
    st.session_state.feedback = None  # Tracks feedback ('up' or 'down')

# Function to handle button clicks
def handle_feedback(feedback_type):
    st.session_state.feedback = feedback_type

st.write("Did you find this helpful?")

# Create thumbs up and thumbs down buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("👍 Thumbs Up", key="thumbs_up"):
        handle_feedback("up")
with col2:
    if st.button("👎 Thumbs Down", key="thumbs_down"):
        handle_feedback("down")

# Display feedback result
if st.session_state.feedback == "up":
    st.success("Thank you for the positive feedback!")
elif st.session_state.feedback == "down":
    st.error("Thank you for the feedback! We'll work on improving.")

# Prevent clicking both buttons at the same time
if st.session_state.feedback:
    st.write(f"You selected: {st.session_state.feedback}")



import streamlit as st

# Create radio buttons for thumbs up/down
feedback = st.radio(
    "Did you find this helpful?",
    ("👍 Thumbs Up", "👎 Thumbs Down", "No Feedback"),
    index=2,  # Default to "No Feedback"
)

if feedback == "👍 Thumbs Up":
    st.success("Thank you for the positive feedback!")
elif feedback == "👎 Thumbs Down":
    st.error("Thank you for the feedback! We'll work on improving.")
else:
    st.info("Please provide your feedback.")
