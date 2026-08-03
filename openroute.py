import requests
import json

API_KEY = "..."
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
model = 'tencent/hy3:free'

# First API call with reasoning
response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": model,
    "messages": [
        {
          "role": "user",
          "content": "You are an outstanding writer. Make up a short story about cat and robot."
        }
      ],
    "reasoning": {"enabled": True}
  })
)

# Extract the assistant message with reasoning_details
response = response.json()
answer = response['choices'][0]['message']
print(response['choices'][0]['message'])
content = answer.get('content')
reasoning_details = answer.get('reasoning_details')

# Preserve the assistant message with reasoning_details
messages = [
  {"role": "user", "content": "You are an outstanding writer. Make up a short story about cat and robot."},
  {
    "role": "assistant",
    "content": content,
    "reasoning_details": reasoning_details  # Pass back unmodified
  },
  {"role": "user", "content": "Are you sure? Think carefully."}
]

# Second API call - model continues reasoning from where it left off
response2 = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": model,
    "messages": messages,  # Includes preserved reasoning_details
    "reasoning": {"enabled": True}
  })
)

answer = response2.json()
print(answer['choices'][0]['message'])
