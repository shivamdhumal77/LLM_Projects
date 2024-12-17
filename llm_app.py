import streamlit as st
from langchain.llms import Ollama

# Initialize the models
gemma = Ollama(model="gemma:2b")
llama = Ollama(model="llama3.2")
phi3_model = Ollama(model="phi3")

# Function to query all models
def query_models(query):
    # Send the query to each model and store the results
    responses = {
        "phi": phi3_model(query),
        "gemma2": gemma(query),
        "llama3.2": llama(query),
    }
    
    return responses

# Streamlit interface
def main():
    st.title("Multi-Model Query App")
    st.write("Enter your query below and see the responses from different models.")
    
    # Input box for the user to enter their query
    user_query = st.text_input("Enter your query:")
    
    # If the user has entered a query, call the models
    if user_query:
        st.write("### Responses from models:")

        # Get responses from the models
        responses = query_models(user_query)

        # Display results in separate boxes
        with st.expander("Phi3 Model Response"):
            st.write(responses["phi"])

        with st.expander("Gemma2 Model Response"):
            st.write(responses["gemma2"])

        with st.expander("Llama3.2 Model Response"):
            st.write(responses["llama3.2"])

# Run the Streamlit app
if __name__ == "__main__":
    main()
