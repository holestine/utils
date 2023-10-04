import openai
import key     # Put the code to load your key in here

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # Degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def collect_messages(prompt):
    context.append({'role':'user', 'content':f"{prompt}"})
    response = get_completion_from_messages(context) 
    context.append({'role':'assistant', 'content':f"{response}"})
    return response

context = [ {'role':'system', 'content':"""
You are a code generation tool used to create styles for the BuildBox application. \
The user will give you a description of the style and you will create JSON code. \
All the properties are the same as XAML styles but there is also a new one, special_property, \
that should be set to a color that compliments the background. \
"""} ]  # accumulate messages in this object

#context.append(
#{'role':'system', 'content':'Additional information including samples can be added here'},    
#)

human_response = input("\nStart using the BuildBox Styles Chatbot.\n\n")
while human_response != 'done':
    gpt_response   = collect_messages(human_response)
    human_response = input(f"\n{gpt_response}\n\n")
