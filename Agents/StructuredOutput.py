import os
import dotenv
from openai import OpenAI


def weather(model_name, client):

    dialog = [
        {"role": "user", "content": "Hi! I want to go to Rome in a week. How's the weather there?"},
        {"role": "assistant", "content": "Rome will be quite warm in a week. Around 25 degrees is expected."},
        {"role": "user", "content": "Is that Celsius or Fahrenheit?  And will it rain?"},
        {"role": "assistant", "content": "Celsius, of course! There won't be any rain, it'll be sunny."},
        {"role": "user", "content": "Great! Then I'll pack my summer clothes. Thanks for the info!"}
    ]
    weather_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "weather",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "temperature": {
                        "type": "number",
                        "description": "Temperature value",
                    },
                    "unit": {
                        "type": "string",
                        "description": "Temperature unit",
                        "enum": ["C", "F"]
                    },
                    "location": {
                        "type": "string",
                        "description": "City or location name",
                    },
                    "rain": {
                        "type": "string",
                        "description": "Rain",
                        "enum": ["Yes", "No"]
                    },
                },
                "required": ["temperature", "unit", "location"],
                "additionalProperties": False,
            },
        }
    }
    completion = client.chat.completions.create(
        model=model_name,
        messages=dialog,
        response_format=weather_schema,
        max_tokens=200,
        tool_choice="auto",
    )
    print(completion.choices[0].message.content)


def generate_search_queries(user_query, model_name, client, system_message, search_query_schema):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": system_message}, {"role": "user", "content": user_query}],
        response_format=search_query_schema,
        max_tokens=1000
    )
    return completion.choices[0].message.content


def search(model_name, client):
    examples = [
        {
            "role": "user",
            "content": "We're moving into a new apartment and need to buy several things to furnish it: a large refrigerator with a bottom freezer, at least 350 liters, and an A+++ energy rating. We also need a top-loading washing machine with a 7-8 kg capacity and a delayed start function. We're also looking for a 65-inch Smart TV with Netflix and YouTube support and 4K UHD resolution. For the kitchen, we need a 4-burner gas stove with an electric grill and oven. Finally, we need a 12-placeholder dishwasher with a quick wash and condenser drying function."
        },
        {
            "role": "user",
            "content": "We're getting our child ready for first grade and need to buy everything he needs: a school uniform for a boy (jacket and trousers, height 134 cm), an orthopedic backpack with a rigid back and reflective elements. School supplies: a grade school diary, 12 squared and lined notebooks, textbook covers, and a two-tier pencil case. We also need a sports uniform for physical education: size 33 sneakers with Velcro fasteners, a tracksuit, and two white T-shirts. Electronics: a children's smartwatch with a GPS tracker and phone function, and a dimmable desk lamp."
        },
        {
            "role": "user",
            "content": "We're setting up a home gym and workspace: for training, we need a power rack with a pull-up bar and parallel bars up to 2.2 meters high, an Olympic 20 kg barbell, and a set of weights from 5 to 25 kg. We're also looking for an adjustable bench press and adjustable dumbbells up to 25 kg each. For cardio, we need an electric treadmill with a minimum motor power of 2.5 hp and a folding treadmill. For the workspace, we need a gaming computer with an i7 processor, RTX 4070 graphics card, 32 GB of RAM, and a 1 TB SSD. A 27-inch monitor with a 144 Hz refresh rate and an IPS panel. A gaming chair with lumbar support and adjustable height is also needed. Wireless noise-canceling headphones for concentration."
        }
    ]

    search_query_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "search_queries",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "description": "Product category"
                                },
                                "query": {
                                    "type": "string",
                                    "description": "Search query for internal search"
                                }
                            },
                            "required": ["category", "query"],
                            "additionalProperties": False
                        },
                        "description": "List of search queries for different product categories"
                    }
                },
                "required": ["queries"],
                "additionalProperties": False,
            },
        }
    }
    system_message = "Generate search queries based on user needs. The search query format should match the search in Brave."
    for i, dialog in enumerate(examples, 1):
        print(f"=== Example {i} ===")
        user_query = dialog["content"]
        print(f"User query: {user_query}")
        print("\nGenerated search queries:")
        try:
            result = generate_search_queries(user_query, model_name, client, system_message, search_query_schema)
            print(result)
        except Exception as e:
            print(f"Error: {e}")
        print("\n" + "=" * 50 + "\n")

def main(**kwargs):

    dotenv.load_dotenv()
    API_KEY = os.getenv('OPENROUTE_API_KEY', '')
    model_name = "nvidia/nemotron-3-ultra-550b-a55b:free"
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
    if kwargs['weather']:
        weather(model_name, client)
    if kwargs['search']:
        search(model_name, client)


if __name__ == "__main__":
    params = {'weather': False, 'search': True}
    main(**params)
