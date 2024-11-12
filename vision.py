import ollama
import csv

response_v = ollama.chat(
    model='llama3.2-vision',
    messages=[{
        'role': 'user',
        'content': 'youre a secretary of a company. extract the personal information of this card using json format',
        'images': ['ecum.jpg']
    }]
)

response_c = ollama.chat(
    model='qwen2.5-coder:14b',
    messages=[{
        'role': 'user',
        'content': response_v['message']["content"] + "\n please base on the above information, produce a table in csv format to contain the personal information. first colomn is the title, and second column is the content.",
        
    }]
)
with open('university_records.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(response_c["message"]["content"])

