import os 
import asyncio
import tiktoken
from enum import Enum
from . import async_openai_client, async_anthropic_client, async_groq_client, groq_client



enc=tiktoken.get_encoding("cl100k_base")

# Enum for model names
model_name_company_mapping={
    "gpt-4-turbo": "openai",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "ft:gpt-4o-mini-2024-07-18:personal::9oMM1Fxh": "openai",
    "gpt-3.5-turbo": "openai",

    "claude-3-5-sonnet-20240620": "anthropic",
    "claude-3-sonnet-20240229": "anthropic",
    "claude-3-opus-20240229": "anthropic",
    "claude-3-haiku-20240307": "anthropic",

    "gemini-1.5-flash": "google",

    "llama-3.1-70b-versatile": "groq",
    "llama-3.1-8b-instant": "groq",
    "llama-3.2-90b-vision-preview": "groq",
    "llama3-70b-8192": "groq",
    "llama-3.2-11b-vision-preview": "groq",
}
class ModelCompanyEnum(str, Enum):
    openai = "openai"
    anthropic = "anthropic"
    google = "google"
    groq = "groq"
class ModelNameEnum(str, Enum):
    gpt_4_turbo = "gpt-4-turbo"
    gpt_4o = "gpt-4o"
    gpt_4o_mini = "gpt-4o-mini"
    gpt_4o_mini_tune = "ft:gpt-4o-mini-2024-07-18:personal::9oMM1Fxh"
    gpt_3_5_turbo = "gpt-3.5-turbo"

    claude_3_5_sonnet = "claude-3-5-sonnet-20240620"
    claude_3_sonnet = "claude-3-sonnet-20240229"
    claude_3_opus = "claude-3-opus-20240229"
    claude_3_haiku = "claude-3-haiku-20240307"

    gemini_1_5_flash="gemini-1.5-flash"

    llama_3_1_70B_versatile="llama-3.1-70b-versatile"
    llama_3_1_8B_instant="llama-3.1-8b-instant"
    llama_3_2_90B_vision_preview="llama-3.2-90b-vision-preview"
    llama_3_70b_8192="llama3-70b-8192"
    llama_3_2_11b_vision_preview="llama-3.2-11b-vision-preview"

class ModelCostEnum(str, Enum):
    gpt_4_turbo = {"input": 10/1000000.0, "output": 30/1000000.0}
    gpt_4o = {"input": 5/1000000.0, "output": 20/1000000.0}
    gpt_4o_mini = {"input": 0.15/1000000.0, "output": 0.6/1000000.0}
    gpt_4o_mini_tune = {"input": 0.3/1000000.0, "output": 1.2/1000000.0}
    gpt_3_5_turbo = {"input": 1/1000000.0, "output": 8/1000000.0}

    claude_3_5_sonnet = {"input": 3/1000000.0, "output": 15/1000000.0}
    claude_3_sonnet = {"input": 3/1000000.0, "output": 15/1000000.0}
    claude_3_opus = {"input": 10/1000000.0, "output": 60/1000000.0}
    claude_3_haiku = {"input": 0.25/1000000.0, "output": 1.25/1000000.0}

    gemini_1_5_flash = {"input": 0.075/1000000.0, "output": 0.3/1000000.0}

    llama_3_1_70B_versatile = {"input": 0.59/1000000.0, "output": 0.79/1000000.0}
    llama_3_1_8B_instant = {"input": 0.2/1000000.0, "output": 0.3/1000000.0}
    llama_3_2_90B_vision_preview = {"input": 0.89/1000000.0, "output": 0.89/1000000.0}
    llama_3_70b_8192 = {"input": 0.59/1000000.0, "output": 0.79/1000000.0}
    llama_3_2_11b_vision_preview = {"input": 0.89/1000000.0, "output": 0.89/1000000.0}
    



# Response Handler
async def async_response_handler(
    prompt: dict,
    model_name: str,
    temp=0.0,
    frequency_penalty=0,
    presence_penalty=0,
    max_tokens=None
) -> tuple[str, float]:
    model_company=model_name_company_mapping.get(model_name,None)

    max_retries=1

    for retry in range(0, max_retries):
        try:
            if model_company==ModelCompanyEnum.openai:
                if max_tokens:
                    response = await async_openai_client.chat.completions.create(
                            model=model_name,
                            messages=prompt,
                            temperature=temp,
                            top_p=1,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty,
                        )
                else:
                    response = await async_openai_client.chat.completions.create(
                            model=model_name,
                            messages=prompt,
                            temperature=temp,
                            top_p=1,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty,
                            max_tokens=max_tokens
                        )
                
                cost=prompt_and_response_cost(prompt, response.choices[0].message.content, model_name)

                return response.choices[0].message.content, cost
            
            elif model_company==ModelCompanyEnum.anthropic:
                system_message=prompt[0]["content"]
                response = await async_anthropic_client.messages.create(
                        model=model_name,
                        max_tokens=4000,
                        temperature=temp,
                        system=system_message,
                        messages=prompt[1:]
                    )
                
                cost=prompt_and_response_cost(prompt, response.content[0].text, model_name)

                return response.content[0].text, cost
            
            elif model_company==ModelCompanyEnum.google:
                # Create the model
                pass
                # generation_config = {
                #     "temperature": 1,
                #     "top_p": 0.95,
                #     "top_k": 64,
                #     "max_output_tokens": 8192,
                #     "response_mime_type": "text/plain",
                # }
                # model = genai.GenerativeModel(
                #     model_name="gemini-1.5-flash",
                #     generation_config=generation_config,
                #     system_instruction=prompt[0]["content"],
                # )
                # response = await model.generate_content_async(prompt[1]["content"])
                # cost=prompt_and_response_cost(prompt, response._result.candidates[0].content.parts[0].text, model_name)

                # return response._result.candidates[0].content.parts[0].text, cost
            
            elif model_company==ModelCompanyEnum.groq:
                pass

                
                chat_completion = await async_groq_client.chat.completions.create(
                    messages=prompt,
                    model=model_name,
                )
                cost=prompt_and_response_cost(prompt, chat_completion.choices[0].message.content, model_name)

                return chat_completion.choices[0].message.content, cost
            break
        except Exception as e:
            if (retry+1)>=max_retries:
                raise Exception(f"Error in async_response_handler after retry {retry}: ", e)
            else:
                await asyncio.sleep(10+((retry+1)*3))
    
    raise Exception("Model not recognized")
    
# calculate cost
def calculate_cost(model_name, input="", output=""):
    # get the cost rates for the model name
    for model_enum_name in ModelNameEnum:
        if model_name==model_enum_name.value:
            cost_rate = ModelCostEnum[model_enum_name.name].value
            cost_rate=eval(cost_rate)

            input_cost = len(enc.encode(input)) * cost_rate["input"]
            output_cost = len(enc.encode(output)) * cost_rate["output"]
            # print(input_cost, output_cost, len(input), len(output))
            # return {"input": input_cost, "output": output_cost}
            cost=input_cost+output_cost
            return cost
    return 0

def prompt_and_response_cost(prompt, response, model_name):
    input_str=""
    for message in prompt:
        if "content" in message:
            input_str+=message["content"]
    output_str=response
    return calculate_cost(model_name, input_str, output_str)












import replicate

async def generate_image(prompt):

    input = {
        "prompt": prompt
    }

    output = replicate.run(
        "black-forest-labs/flux-dev",
        input=input,
    )
    return output



















# Response Handler
def streamed_response_handler(
    prompt: dict,
    model_name: str,
    temp=0.0,
    frequency_penalty=0,
    presence_penalty=0,
    max_tokens=None
) -> tuple[str, float]:
    model_company=model_name_company_mapping.get(model_name,None)

    max_retries=1

    for retry in range(0, max_retries):
        try:
            # if model_company==ModelCompanyEnum.openai:
            #     if max_tokens:
            #         response = await async_openai_client.chat.completions.create(
            #                 model=model_name,
            #                 messages=prompt,
            #                 temperature=temp,
            #                 top_p=1,
            #                 frequency_penalty=frequency_penalty,
            #                 presence_penalty=presence_penalty,
            #             )
            #     else:
            #         response = await async_openai_client.chat.completions.create(
            #                 model=model_name,
            #                 messages=prompt,
            #                 temperature=temp,
            #                 top_p=1,
            #                 frequency_penalty=frequency_penalty,
            #                 presence_penalty=presence_penalty,
            #                 max_tokens=max_tokens
            #             )
                
            #     cost=prompt_and_response_cost(prompt, response.choices[0].message.content, model_name)

            #     return response.choices[0].message.content, cost
            
            # elif model_company==ModelCompanyEnum.anthropic:
            #     system_message=prompt[0]["content"]
            #     response = await async_anthropic_client.messages.create(
            #             model=model_name,
            #             max_tokens=4000,
            #             temperature=temp,
            #             system=system_message,
            #             messages=prompt[1:]
            #         )
                
            #     cost=prompt_and_response_cost(prompt, response.content[0].text, model_name)

            #     return response.content[0].text, cost
            
            if model_company==ModelCompanyEnum.google:
                # Create the model
                pass
                # generation_config = {
                #     "temperature": 1,
                #     "top_p": 0.95,
                #     "top_k": 64,
                #     "max_output_tokens": 8192,
                #     "response_mime_type": "text/plain",
                # }
                # model = genai.GenerativeModel(
                #     model_name="gemini-1.5-flash",
                #     generation_config=generation_config,
                #     system_instruction=prompt[0]["content"],
                # )
                # response = await model.generate_content_async(prompt[1]["content"])
                # cost=prompt_and_response_cost(prompt, response._result.candidates[0].content.parts[0].text, model_name)

                # return response._result.candidates[0].content.parts[0].text, cost
            
            elif model_company==ModelCompanyEnum.groq:
                
                chat_completion = groq_client.chat.completions.create(
                    messages=prompt,
                    model=model_name,
                    stream=True
                )
                for completion in chat_completion:
                    # print(completion)
                    # check if choice object has message attribute
                    if (completion.choices[0].delta.content):
                        print(completion.choices[0].delta.content, end="")

                cost=prompt_and_response_cost(prompt, chat_completion.choices[0].message.content, model_name)

                return chat_completion.choices[0].message.content, cost
            break
        except Exception as e:
            if (retry+1)>=max_retries:
                raise Exception(f"Error in async_response_handler after retry {retry}: ", e)
    
    raise Exception("Model not recognized")