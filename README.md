st.markdown(
    """
    <style>
    .custom-text-input input {
        width: 400px !important; /* Set width */
        height: 40px !important; /* Set height */
        font-size: 16px; /* Adjust font size */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Apply the custom CSS class to Streamlit's text input
user_input = st.text_input(
    "Enter text:", 
    key="custom_text_input", 
    placeholder="Type something here..."
)

if user_input:
    st.write("You entered:", user_input)
