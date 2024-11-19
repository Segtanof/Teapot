import ollama
import pandas as pd

response_v = ollama.chat(
    model='llama3.2-vision',
    messages=[{
        'role': 'user',
        'content': 'describe what you see',
        'images': ['test.png']
    }]
)

response_c = ollama.chat(
    model='llama3.2:latest',
    messages=[{
        'role': 'user',
        'content': response_v['message']["content"] + "\n please base on the above information, produce a pandas dataframme to contain the information. no need to greet or make conclusion, directly output your code. assume this structure: data = { [your code]} \n df = pd.DataFrame(data). ONLY limit your response within the brackets. do not provide import info, or last time about df = pd...",        
    }]
)

data = response_c["message"]["content"]

print(data)

df = pd.DataFrame({data})

print(df.head())

# with open('university_records.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(response_c["message"]["content"])

