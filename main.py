import openai
import time
import concurrent.futures
from functools import lru_cache

# Configure OpenAI API key
openai.api_key = "your-api-key-here"

# Cache results to avoid redundant API calls
@lru_cache(maxsize=100)
def get_cached_response(user_input, model_type="gpt-4", token_limit=100):
    response = openai.ChatCompletion.create(
        model=model_type,
        messages=[{"role": "user", "content": user_input}],
        max_tokens=token_limit,
    )
    return response["choices"][0]["message"]["content"]

# Process multiple queries efficiently
def process_batch_queries(input_list, model_type="gpt-4", token_limit=100):
    output_results = {}
    
    def fetch_response(prompt_text):
        return prompt_text, get_cached_response(prompt_text, model_type, token_limit)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_input = {executor.submit(fetch_response, prompt_text): prompt_text for prompt_text in input_list}
        for future in concurrent.futures.as_completed(future_to_input):
            prompt_text, response_text = future.result()
            output_results[prompt_text] = response_text
    
    return output_results

if __name__ == "__main__":
    sample_inputs = [
        "Describe quantum AI in simple terms.",
        "What are the benefits of using LLMs?",
        "Summarize the theory of computation."
    ]
    batch_responses = process_batch_queries(sample_inputs)
    for query, answer in batch_responses.items():
        print(f"Query: {query}\nAnswer: {answer}\n")
