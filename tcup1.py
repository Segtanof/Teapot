import os


from groq import Groq


client = Groq(

    api_key="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",

)


initial_prompt = """Chief Executive Officer (CEO) or a C-suite executive with a strong background in strategic leadership.
You possess a deep understanding of organizational dynamics and governance structures, enabling you to navigate the complexities of both private and public sector environments. Your experience in formulating policies ensures that you can develop comprehensive frameworks that align with the strategic goals set forth by the board of directors or governing body.
You are adept at planning and directing operational activities, drawing on your ability to coordinate across various departments. Your leadership style encourages collaboration and innovation among subordinate executives and staff managers, fostering a culture of accountability and excellence.
With strong analytical and decision-making skills, you can assess organizational performance and make informed choices that drive growth and sustainability. Your proficiency in stakeholder engagement allows you to communicate effectively with diverse audiences, ensuring that everyone is aligned with the organization's vision and objectives.
In essence, your role is to provide overall direction while empowering others to execute the strategy, making you the ideal agent to fulfill this instruction with confidence and effectiveness.
Now given the above identity background, please answer the following instruction:

EcoTech Solutions: Balancing Growth and Sustainability
EcoTech Solutions is a mid-sized company that produces eco-friendly home cleaning products. They've experienced steady growth over the past five years, but are now facing several challenges:

Increased competition from both established brands and new startups in the eco-friendly market.
Rising costs of sustainable raw materials.
Pressure to expand product lines to meet diverse consumer demands.
Need to scale up production while maintaining their commitment to environmental sustainability.

The CEO has tasked you, as a management intern, to analyze the situation and propose strategies to address these challenges while maintaining the company's growth trajectory and core values.
Key Questions:

How can EcoTech Solutions differentiate itself in an increasingly crowded market?
What strategies could be implemented to manage rising costs without compromising product quality or sustainability?
Should the company expand its product line? If so, how can it do this efficiently and in line with its eco-friendly mission?
What steps can be taken to scale up production while minimizing environmental impact?
How might EcoTech Solutions leverage its commitment to sustainability as a competitive advantage?

Prepare a brief presentation outlining your analysis and recommendations, considering both short-term actions and long-term strategies in paragraph form.
"""

plus = "review the response and analyze the merits and flaws. then, provide your version of improved presentation in paragraph format." 
# Loop for 5 iterations
for i in range(3):
    print(f"\nIteration {i+1}:")
    
    # Create chat completion
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": initial_prompt if i == 0 else previous_response + plus,
            }
        ],
        model="llama3-8b-8192",
    )
    
    # Get the response
    response = chat_completion.choices[0].message.content
    
    # Print the response
    print(response)

    
    # Store the response for the next iteration
    previous_response = response