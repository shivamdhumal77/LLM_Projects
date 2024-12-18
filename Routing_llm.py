import streamlit as st
from langchain.llms import Ollama
from collections import Counter
import concurrent.futures

# Example models
models = ["llama3.2", "nemotron-mini", "phi3", "gemma:2b"]

# Function to generate responses from the models
def generate_response_from_model(query, model_name):
    llm = Ollama(model=model_name)
    response = llm(query)
    return model_name, response

# Function to remove duplicates (based on exact sentence matching or similarity)
def remove_duplicates(responses):
    unique_responses = []
    seen_responses = set()
    for response in responses:
        if response not in seen_responses:
            unique_responses.append(response)
            seen_responses.add(response)
    return unique_responses

# Function to synthesize responses from all models
def synthesize_responses(responses):
    unique_responses = remove_duplicates(responses)
    final_response = " ".join(unique_responses)
    return final_response

# Streamlit UI to take user input
st.title("LLM Query Synthesizer")

query = st.text_area("Please enter your query:", "")

# Button to generate responses
if st.button("Generate Responses"):
    if query.strip():  # Ensure there's input
        # Generate responses from all models using parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_model = {executor.submit(generate_response_from_model, query, model): model for model in models}
            responses = {}
            for future in concurrent.futures.as_completed(future_to_model):
                model_name, response = future.result()
                responses[model_name] = response

        # Show individual responses from each model in separate boxes
        for model, response in responses.items():
            st.subheader(f"Response from {model}")
            st.text_area(f"Response from {model}:", response, height=200)

        # Combine and synthesize all responses
        final_response = synthesize_responses(list(responses.values()))

        # Display the final synthesized response
        st.subheader("Final Synthesized Response:")
        st.text_area("Final Synthesized Response:", final_response, height=200)
    else:
        st.warning("Please enter a query before generating responses.")