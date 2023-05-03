import openai
import key

def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def q_with_data(question, data):
    
    prompt = f"""
    Your task is to answer with following question using the data delimited by <>.

    {question}

    <{data}>
    """

    return get_completion(prompt)


def samples():
    response = get_completion('What are the best jobs going to be in 20 years?')
    print(f'{response}\n')

    response = get_completion('Write Python code to compute the Fibonacci sequence')
    print(f'{response}\n')

    response = get_completion('Write a short essay about the book Atlas Shrugged')
    print(f'{response}\n')


if __name__ == "__main__":
    samples()

