import os
import ast
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_completion(prompt, model="gpt-4o"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content

def get_recommendations_better(games, file, n, model="gpt-4"):
    with open(file, "r") as file:
        data = file.read()
    prompt = f"""Give me the best {n} new board game recommendations from the attached csv file.
                 Here are the user's opinions: {games}. The keys of this dictionary corresponds
                 to the board game ids, and the values correspond to the user's rating of each game.
                 Here is the data: {data}. Please present the result as a Python dictionary with the BGGIds as the keys. 
                 The values for the dictionary will be in a list format, and will follow this structure: 
                 [Name, AvgRating, NumUserRatings, YearPublished, GameWeight] (and return no other output). 
                 Make the recommendations based off genre, game weight, and rating. Thanks!
                 P.S. make sure your output doesnt have ```python or any other formatting stuff.
                 Also, don't recommend the same games present in the user's opinions already.
                 DONT JUST LOOK AT THE FIRST FEW GAMES IN THE DATA, LOOK AT ALL OF IT.
                 If you have missing data, do your best based on results from the internet (return something if possible).
                 Lastly, round the average rating to two decimal places"""
    output = get_completion(prompt)
    return ast.literal_eval(output)

# def get_recommendations(games, file, n, model="gpt-4o"):
#     my_assistant = client.beta.assistants.create(
#         model=model,
#         instructions="You are an absolute gigachad who can give board game recommendations",
#         name="Board Game Recommender",
#         tools=[{"type": "file_search"}]
#     )
#     my_thread = client.beta.threads.create()

#     my_file = client.files.create(
#         file=open(file, "rb"),
#         purpose='assistants'
#     )

#     my_thread_message = client.beta.threads.messages.create(
#         thread_id=my_thread.id,
#         role="user",
#         content=f"""Give me the best {n} board gamerecommendations from the attached csv file.
#                  Here are the user's opinions: {games}. The keys of this dictionary corresponds
#                  to the board game ids, and the values correspond to the user's rating of each game.
#                  Please present the result as a Python dictionary with the BGGIds as the keys and 
#                  the names as the values""",
#         attachments=[
#             {
#                 "file_id": my_file.id,
#                 "tools": [{"type": "file_search"}] # Or "code_interpreter" if applicable
#             }
#         ]
#     )

#     my_run = client.beta.threads.runs.create(
#         thread_id=my_thread.id,
#         assistant_id=my_assistant.id,
#     )

#     keep_retrieving_run = client.beta.threads.runs.retrieve(
#         thread_id=my_thread.id,
#         run_id=my_run.id
#     )

#     all_messages = client.beta.threads.messages.list(
#         thread_id=my_thread.id
#     )

#     print(f"User: {my_thread_message.content[0].text.value}")
#     print(f"Assistant: {all_messages.data[0].content[0].text.value}")
# 207167:10
# 192291:10, 291457: 10, 237182: 10
# print(get_recommendations_better({172225:10, 192291:10, 2223:10}, "df_games_llm.txt", 10))