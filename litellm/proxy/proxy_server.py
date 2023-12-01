import ast
import asyncio
import os
import re
import requests
import secrets
import subprocess
import sys
import threading
import traceback
from datetime import datetime, timedelta
from typing import Optional, List
import secrets, subprocess
import warnings
messages: list = []
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path - for litellm local dev

try:
    import uvicorn
    import fastapi
    import appdirs
    import backoff
    import yaml
    import rq
    import orjson
except ImportError:
    import sys

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "uvicorn",
            "fastapi",
            "appdirs",
            "backoff",
            "pyyaml", 
            "rq",
             "orjson"
        ]
    )
    import uvicorn
    import fastapi
    import appdirs
    import backoff
    import yaml
    import orjson

    warnings.warn(
        "Installed runtime dependencies for proxy server. Specify these dependencies explicitly with `pip install litellm[proxy]`"
    )

import random

list_of_messages = [
    "'The thing I wish you improved is...'",
    "'A feature I really want is...'",
    "'The worst thing about this product is...'",
    "'This product would be better if...'",
    "'I don't like how this works...'",
    "'It would help me if you could add...'",
    "'This feature doesn't meet my needs because...'",
    "'I get frustrated when the product...'",
]


def generate_feedback_box():
    box_width = 60

    # Select a random message
    message = random.choice(list_of_messages)

    print()
    print("\033[1;37m" + "#" + "-" * box_width + "#\033[0m")
    print("\033[1;37m" + "#" + " " * box_width + "#\033[0m")
    print("\033[1;37m" + "# {:^59} #\033[0m".format(message))
    print(
        "\033[1;37m"
        + "# {:^59} #\033[0m".format("https://github.com/BerriAI/litellm/issues/new")
    )
    print("\033[1;37m" + "#" + " " * box_width + "#\033[0m")
    print("\033[1;37m" + "#" + "-" * box_width + "#\033[0m")
    print()
    print(" Thank you for using LiteLLM! - Krrish & Ishaan")
    print()
    print()
    print()
    print(
        "\033[1;31mGive Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new\033[0m"
    )
    print()
    print()


import litellm
from litellm.caching import DualCache
litellm.suppress_debug_info = True
from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.routing import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse, FileResponse, ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
import json
import logging
# from litellm.proxy.queue import start_rq_worker_in_background

app = FastAPI(docs_url="/", title="LiteLLM API")
router = APIRouter()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

user_api_base = None
user_model = None
user_debug = False
user_max_tokens = None
user_request_timeout = None
user_temperature = None
user_telemetry = True
user_config = None
user_headers = None
local_logging = True # writes logs to a local api_log.json file for debugging
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
experimental = False
#### GLOBAL VARIABLES ####
llm_router: Optional[litellm.Router] = None
llm_model_list: Optional[list] = None
general_settings: dict = {}
log_file = "api_log.json"
worker_config = None
master_key = None
prisma_client = None
user_api_key_cache = DualCache()
### REDIS QUEUE ###
async_result = None
celery_app_conn = None
celery_fn = None # Redis Queue for handling requests
#### HELPER FUNCTIONS ####
def print_verbose(print_statement):
    global user_debug
    if user_debug:
        print(print_statement)

def usage_telemetry(
    feature: str,
):  # helps us know if people are using this feature. Set `litellm --telemetry False` to your cli call to turn this off
    if user_telemetry:
        logging.info(R"Telemetry on")
        data = {"feature": feature}  # "local_proxy_server"
        threading.Thread(
            target=litellm.utils.litellm_telemetry, args=(data,), daemon=True
        ).start()

async def user_api_key_auth(request: Request):
    global master_key, prisma_client, llm_model_list
    if master_key is None:
        return
    try:
        api_key = await oauth2_scheme(request=request)
        route = request.url.path
        if api_key == master_key:
            return

        if (route == "/key/generate" or route == "/key/delete") and api_key != master_key:
            raise Exception(f"If master key is set, only master key can be used to generate new keys")

        if prisma_client:
            ## check for cache hit (In-Memory Cache)
            valid_token = user_api_key_cache.get_cache(key=api_key)
            if valid_token is None:
                ## check db
                valid_token = await prisma_client.litellm_verificationtoken.find_first(
                    where={
                        "token": api_key,
                        "expires": {"gte": datetime.utcnow()}  # Check if the token is not expired
                    }
                )
                ## save to cache for 60s
                user_api_key_cache.set_cache(key=api_key, value=valid_token, ttl=60)
            else:
                print(f"API Key Cache Hit!")
            if valid_token:
                litellm.model_alias_map = valid_token.aliases
                config = valid_token.config
                if config != {}:
                    model_list = config.get("model_list", [])
                    llm_model_list =  model_list
                    print("\n new llm router model list", llm_model_list)
                if len(valid_token.models) == 0: # assume an empty model list means all models are allowed to be called
                    return
                else:
                    data = await request.json()
                    model = data.get("model", None)
                    if model in litellm.model_alias_map:
                        model = litellm.model_alias_map[model]
                    if model and model not in valid_token.models:
                        raise Exception(f"Token not allowed to access model")
                return
            else:
                raise Exception(f"Invalid token")
    except Exception as e:
        print(f"An exception occurred - {e}")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": "invalid user key"},
    )

def prisma_setup(database_url: Optional[str]):
    global prisma_client
    if database_url:
        try:
            print("LiteLLM: DATABASE_URL Set in config, trying to 'pip install prisma'")
            os.environ["DATABASE_URL"] = database_url
            subprocess.run(['prisma', 'generate'])
            subprocess.run(['prisma', 'db', 'push', '--accept-data-loss']) # this looks like a weird edge case when prisma just wont start on render. we need to have the --accept-data-loss
            # Now you can import the Prisma Client
            from prisma import Client
            prisma_client = Client()
        except Exception as e:
            print("Error when initializing prisma, Ensure you run pip install prisma", e)

def celery_setup(use_queue: bool):
    global celery_fn, celery_app_conn, async_result
    print(f"value of use_queue: {use_queue}")
    if use_queue:
        from litellm.proxy.queue.celery_worker import start_worker
        from litellm.proxy.queue.celery_app import celery_app, process_job
        from celery.result import AsyncResult
        start_worker(os.getcwd())
        celery_fn = process_job
        async_result = AsyncResult
        celery_app_conn = celery_app

def run_ollama_serve():
        command = ['ollama', 'serve']

        with open(os.devnull, 'w') as devnull:
            process = subprocess.Popen(command, stdout=devnull, stderr=devnull)

def load_router_config(router: Optional[litellm.Router], config_file_path: str):
    global master_key
    config = {}
    try: 
        if os.path.exists(config_file_path):
            with open(config_file_path, 'r') as file:
                config = yaml.safe_load(file)
        else:
            raise Exception(f"Path to config does not exist, Current working directory: {os.getcwd()}, 'os.path.exists({config_file_path})' returned False")
    except Exception as e:
        raise Exception(f"Exception while reading Config: {e}")

    if user_debug:
       print(f"Loaded config YAML:\n{json.dumps(config, indent=2)}")

    ## ENVIRONMENT VARIABLES
    environment_variables = config.get('environment_variables', None)
    if environment_variables:
        for key, value in environment_variables.items():
            os.environ[key] = value

    ## GENERAL SERVER SETTINGS (e.g. master key,..)
    general_settings = config.get("general_settings", {})
    if general_settings is None:
        general_settings = {}
    if general_settings:
        ### MASTER KEY ###
        master_key = general_settings.get("master_key", None)
        if master_key and master_key.startswith("os.environ/"):
            master_key_env_name = master_key.replace("os.environ/", "")
            master_key = os.getenv(master_key_env_name)
        ### CONNECT TO DATABASE ###
        database_url = general_settings.get("database_url", None)
        prisma_setup(database_url=database_url)
        ### START REDIS QUEUE ###
        use_queue = general_settings.get("use_queue", False)
        celery_setup(use_queue=use_queue)

    ## LITELLM MODULE SETTINGS (e.g. litellm.drop_params=True,..)
    litellm_settings = config.get('litellm_settings', None)
    if litellm_settings:
        # ANSI escape code for blue text
        blue_color_code = "\033[94m"
        reset_color_code = "\033[0m"
        for key, value in litellm_settings.items():
            if key == "cache":
                print(f"{blue_color_code}\nSetting Cache on Proxy")
                from litellm.caching import Cache
                cache_type = value["type"]
                cache_host = os.environ.get("REDIS_HOST")
                cache_port = os.environ.get("REDIS_PORT")
                cache_password = os.environ.get("REDIS_PASSWORD")

                # Assuming cache_type, cache_host, cache_port, and cache_password are strings
                print(f"{blue_color_code}Cache Type:{reset_color_code} {cache_type}")
                print(f"{blue_color_code}Cache Host:{reset_color_code} {cache_host}")
                print(f"{blue_color_code}Cache Port:{reset_color_code} {cache_port}")
                print(f"{blue_color_code}Cache Password:{reset_color_code} {cache_password}")
                print()

                litellm.cache = Cache(
                    type=cache_type,
                    host=cache_host,
                    port=cache_port,
                    password=cache_password
                )
            else:
                setattr(litellm, key, value)

    ## MODEL LIST
    model_list = config.get('model_list', None)
    if model_list:
        router = litellm.Router(model_list=model_list, num_retries=3)
        print(f"\033[32mLiteLLM: Proxy initialized with Config, Set models:\033[0m")
        for model in model_list:
            print(f"\033[32m    {model.get('model_name', '')}\033[0m")
            litellm_model_name = model["litellm_params"]["model"]
            if "ollama" in litellm_model_name:
                run_ollama_serve()
    return router, model_list, general_settings

async def generate_key_helper_fn(duration_str: str, models: list, aliases: dict, config: dict, spend: float):
    token = f"sk-{secrets.token_urlsafe(16)}"
    def _duration_in_seconds(duration: str):
        match = re.match(r"(\d+)([smhd]?)", duration)
        if not match:
            raise ValueError("Invalid duration format")

        value, unit = match.groups()
        value = int(value)

        if unit == "s":
            return value
        elif unit == "m":
            return value * 60
        elif unit == "h":
            return value * 3600
        elif unit == "d":
            return value * 86400
        else:
            raise ValueError("Unsupported duration unit")

    duration = _duration_in_seconds(duration=duration_str)
    expires = datetime.utcnow() + timedelta(seconds=duration)
    aliases_json = json.dumps(aliases)
    config_json = json.dumps(config)
    try:
        db = prisma_client
        # Create a new verification token (you may want to enhance this logic based on your needs)
        verification_token_data = {
            "token": token,
            "expires": expires,
            "models": models,
            "aliases": aliases_json,
            "config": config_json,
            "spend": spend
        }
        print(f"verification_token_data: {verification_token_data}")
        new_verification_token = await db.litellm_verificationtoken.create( # type: ignore
           {**verification_token_data} # type: ignore
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    return {"token": new_verification_token.token, "expires": new_verification_token.expires}

async def delete_verification_token(tokens: List[str]):
    global prisma_client
    try:
        if prisma_client:
            # Assuming 'db' is your Prisma Client instance
            deleted_tokens = await prisma_client.litellm_verificationtoken.delete_many(
                where={"token": {"in": tokens}}
            )
        else:
            raise Exception
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    return deleted_tokens

async def generate_key_cli_task(duration_str):
    task = asyncio.create_task(generate_key_helper_fn(duration_str=duration_str))
    await task

def save_worker_config(**data): 
    import json
    os.environ["WORKER_CONFIG"] = json.dumps(data)

def initialize(
    model,
    alias,
    api_base,
    api_version,
    debug,
    temperature,
    max_tokens,
    request_timeout,
    max_budget,
    telemetry,
    drop_params,
    add_function_to_prompt,
    headers,
    save,
    config,
    use_queue
):
    global user_model, user_api_base, user_debug, user_max_tokens, user_request_timeout, user_temperature, user_telemetry, user_headers, experimental, llm_model_list, llm_router, general_settings
    generate_feedback_box()
    user_model = model
    user_debug = debug
    dynamic_config = {"general": {}, user_model: {}}
    if config:
        llm_router, llm_model_list, general_settings = load_router_config(router=llm_router, config_file_path=config)
    if headers:  # model-specific param
        user_headers = headers
        dynamic_config[user_model]["headers"] = headers
    if api_base:  # model-specific param
        user_api_base = api_base
        dynamic_config[user_model]["api_base"] = api_base
    if api_version:
        os.environ[
            "AZURE_API_VERSION"
        ] = api_version  # set this for azure - litellm can read this from the env
    if max_tokens:  # model-specific param
        user_max_tokens = max_tokens
        dynamic_config[user_model]["max_tokens"] = max_tokens
    if temperature:  # model-specific param
        user_temperature = temperature
        dynamic_config[user_model]["temperature"] = temperature
    if request_timeout:
        user_request_timeout = request_timeout
        dynamic_config[user_model]["request_timeout"] = request_timeout
    if alias:  # model-specific param
        dynamic_config[user_model]["alias"] = alias
    if drop_params == True:  # litellm-specific param
        litellm.drop_params = True
        dynamic_config["general"]["drop_params"] = True
    if add_function_to_prompt == True:  # litellm-specific param
        litellm.add_function_to_prompt = True
        dynamic_config["general"]["add_function_to_prompt"] = True
    if max_budget:  # litellm-specific param
        litellm.max_budget = max_budget
        dynamic_config["general"]["max_budget"] = max_budget
    if debug==True:  # litellm-specific param
        litellm.set_verbose = True
    if use_queue:
        celery_setup(use_queue=use_queue)
    if experimental:
        pass
    user_telemetry = telemetry
    usage_telemetry(feature="local_proxy_server")
    curl_command = """
    curl --location 'http://0.0.0.0:8000/chat/completions' \\
    --header 'Content-Type: application/json' \\
    --data ' {
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ]
    }'
    \n
    """
    print()
    print(f"\033[1;34mLiteLLM: Test your local proxy with: \"litellm --test\" This runs an openai.ChatCompletion request to your proxy [In a new terminal tab]\033[0m\n")
    print(f"\033[1;34mLiteLLM: Curl Command Test for your local proxy\n {curl_command} \033[0m\n")
    print("\033[1;34mDocs: https://docs.litellm.ai/docs/simple_proxy\033[0m\n")
# for streaming
def data_generator(response):
    print_verbose("inside generator")
    for chunk in response:
        print_verbose(f"returned chunk: {chunk}")
        try:
            yield f"data: {json.dumps(chunk.dict())}\n\n"
        except:
            yield f"data: {json.dumps(chunk)}\n\n"


def litellm_completion(*args, **kwargs):
    global user_temperature, user_request_timeout, user_max_tokens, user_api_base
    call_type = kwargs.pop("call_type")
    # override with user settings, these are params passed via cli
    if user_temperature: 
        kwargs["temperature"] = user_temperature
    if user_request_timeout:
        kwargs["request_timeout"] = user_request_timeout
    if user_max_tokens: 
        kwargs["max_tokens"] = user_max_tokens
    if user_api_base: 
        kwargs["api_base"] = user_api_base
    ## ROUTE TO CORRECT ENDPOINT ##
    router_model_names = [m["model_name"] for m in llm_model_list] if llm_model_list is not None else []
    try:
        if llm_router is not None and kwargs["model"] in router_model_names: # model in router model list
            if call_type == "chat_completion":
                response = llm_router.completion(*args, **kwargs)
            elif call_type == "text_completion":
                response = llm_router.text_completion(*args, **kwargs)
        else:
            if call_type == "chat_completion":
                response = litellm.completion(*args, **kwargs)
            elif call_type == "text_completion":
                response = litellm.text_completion(*args, **kwargs)
    except Exception as e:
        raise e
    if 'stream' in kwargs and kwargs['stream'] == True: # use generate_responses to stream responses
        return StreamingResponse(data_generator(response), media_type='text/event-stream')
    return response

@app.on_event("startup")
async def startup_event():
    global prisma_client
    import json
    worker_config = json.loads(os.getenv("WORKER_CONFIG"))
    initialize(**worker_config)
    if prisma_client:
        await prisma_client.connect()

@app.on_event("shutdown")
async def shutdown_event():
    global prisma_client
    if prisma_client:
        print("Disconnecting from Prisma")
        await prisma_client.disconnect()

#### API ENDPOINTS ####
@router.get("/v1/models", dependencies=[Depends(user_api_key_auth)])
@router.get("/models", dependencies=[Depends(user_api_key_auth)])  # if project requires model list
def model_list():
    global llm_model_list, general_settings
    all_models = []
    if general_settings.get("infer_model_from_keys", False):
        all_models = litellm.utils.get_valid_models()
    if llm_model_list: 
        all_models = list(set(all_models + [m["model_name"] for m in llm_model_list]))
    if user_model is not None:
        all_models += [user_model]
    print_verbose(f"all_models: {all_models}")
    ### CHECK OLLAMA MODELS ### 
    try:
        response = requests.get("http://0.0.0.0:11434/api/tags")
        models = response.json()["models"]
        ollama_models = ["ollama/" + m["name"].replace(":latest", "") for m in models]
        all_models.extend(ollama_models)
    except Exception as e: 
        pass
    return dict(
        data=[
            {
                "id": model,
                "object": "model",
                "created": 1677610602,
                "owned_by": "openai",
            }
            for model in all_models
        ],
        object="list",
    )

@router.post("/v1/completions", dependencies=[Depends(user_api_key_auth)])
@router.post("/completions", dependencies=[Depends(user_api_key_auth)])
@router.post("/engines/{model:path}/completions", dependencies=[Depends(user_api_key_auth)])
async def completion(request: Request, model: Optional[str] = None):
    try: 
        body = await request.body()
        body_str = body.decode()
        try:
            data = ast.literal_eval(body_str)
        except: 
            data = json.loads(body_str)
        data["model"] = (
            general_settings.get("completion_model", None) # server default
            or user_model # model name passed via cli args
            or model # for azure deployments
            or data["model"] # default passed in http request
        )
        if user_model:
            data["model"] = user_model
        data["call_type"] = "text_completion"
        if "metadata" in data:
            data["metadata"]["user_api_key"] = user_api_key_dict["api_key"]
        else:
            data["metadata"] = {"user_api_key": user_api_key_dict["api_key"]}

        return litellm_completion(
            **data
        )
    except Exception as e: 
        print(f"\033[1;31mAn error occurred: {e}\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`")
        error_traceback = traceback.format_exc()
        error_msg = f"{str(e)}\n\n{error_traceback}"
        try:
            status = e.status_code  # type: ignore
        except:
            status = 500
        raise HTTPException(
            status_code=status,
            detail=error_msg
        )
                              

@router.post("/v1/chat/completions", dependencies=[Depends(user_api_key_auth)])
@router.post("/chat/completions", dependencies=[Depends(user_api_key_auth)])
@router.post("/openai/deployments/{model:path}/chat/completions", dependencies=[Depends(user_api_key_auth)]) # azure compatible endpoint
async def chat_completion(request: Request, model: Optional[str] = None, user_api_key_dict: dict = Depends(user_api_key_auth)):
    global general_settings, user_debug
    try: 
        body = await request.body()
        body_str = body.decode()
        try:
            data = ast.literal_eval(body_str)
        except: 
            data = json.loads(body_str)
        print_verbose(f"receiving data: {data}")
        data["model"] = (
            general_settings.get("completion_model", None) # server default
            or user_model # model name passed via cli args
            or model # for azure deployments
            or data["model"] # default passed in http request
        )

        if "metadata" in data:
            data["metadata"]["user_api_key"] = user_api_key_dict["api_key"]
        else:
            data["metadata"] = {"user_api_key": user_api_key_dict["api_key"]}

        global user_temperature, user_request_timeout, user_max_tokens, user_api_base
        # override with user settings, these are params passed via cli
        if user_temperature: 
            data["temperature"] = user_temperature
        if user_request_timeout:
            data["request_timeout"] = user_request_timeout
        if user_max_tokens: 
            data["max_tokens"] = user_max_tokens
        if user_api_base: 
            data["api_base"] = user_api_base
        router_model_names = [m["model_name"] for m in llm_model_list] if llm_model_list is not None else []
        if llm_router is not None and data["model"] in router_model_names: # model in router model list 
                response = await llm_router.acompletion(**data)
        else: 
            response = await litellm.acompletion(**data)
        if 'stream' in data and data['stream'] == True: # use generate_responses to stream responses
            return StreamingResponse(async_data_generator(response), media_type='text/event-stream')
        return response
    except Exception as e: 
        print(f"\033[1;31mAn error occurred: {e}\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`")
        if llm_router is not None and data["model"] in router_model_names:
            print("Results from router")
            print("\nRouter stats")
            print("\nTotal Calls made")
            for key, value in llm_router.total_calls.items():
                print(f"{key}: {value}")
            print("\nSuccess Calls made")
            for key, value in llm_router.success_calls.items():
                print(f"{key}: {value}")
            print("\nFail Calls made")
            for key, value in llm_router.fail_calls.items():
                print(f"{key}: {value}")
        if user_debug:
            traceback.print_exc()
        error_traceback = traceback.format_exc()
        error_msg = f"{str(e)}\n\n{error_traceback}"
        try:
            status = e.status_code # type: ignore
        except:
            status = 500
        raise HTTPException(
            status_code=status,
            detail=error_msg
        )

@router.post("/v1/embeddings", dependencies=[Depends(user_api_key_auth)], response_class=ORJSONResponse)
@router.post("/embeddings", dependencies=[Depends(user_api_key_auth)], response_class=ORJSONResponse)
async def embeddings(request: Request, user_api_key_dict: dict = Depends(user_api_key_auth)): 
    try: 

        # Use orjson to parse JSON data, orjson speeds up requests significantly
        body = await request.body()
        data = orjson.loads(body)

        data["model"] = (
            general_settings.get("embedding_model", None) # server default
            or user_model # model name passed via cli args
            or data["model"] # default passed in http request
        )
        if user_model:
            data["model"] = user_model
        data["metadata"] = {"user_api_key": user_api_key_dict["api_key"]}
        ## ROUTE TO CORRECT ENDPOINT ##
        router_model_names = [m["model_name"] for m in llm_model_list] if llm_model_list is not None else []
        if llm_router is not None and data["model"] in router_model_names: # model in router model list
            response = await llm_router.aembedding(**data)
        else:
            response = await litellm.aembedding(**data)
        return response
    except Exception as e:
        traceback.print_exc()
        raise e
    except Exception as e:
        pass

#### KEY MANAGEMENT #### 

@router.post("/key/generate", dependencies=[Depends(user_api_key_auth)])
async def generate_key_fn(request: Request):
    data = await request.json()

    duration_str = data.get("duration", "1h")  # Default to 1 hour if duration is not provided
    models = data.get("models", []) # Default to an empty list (meaning allow token to call all models)
    aliases = data.get("aliases", {}) # Default to an empty dict (no alias mappings, on top of anything in the config.yaml model_list)
    config = data.get("config", {})
    spend = data.get("spend", 0)
    if isinstance(models, list):
        response = await generate_key_helper_fn(duration_str=duration_str, models=models, aliases=aliases, config=config, spend=spend)
        return {"key": response["token"], "expires": response["expires"]}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "models param must be a list"},
        )

@router.post("/key/delete", dependencies=[Depends(user_api_key_auth)])
async def delete_key_fn(request: Request):
    try:
        data = await request.json()

        keys = data.get("keys", [])

        if not isinstance(keys, list):
            if isinstance(keys, str):
                keys = [keys]
            else:
                raise Exception(f"keys must be an instance of either a string or a list")

        deleted_keys = await delete_verification_token(tokens=keys)
        assert len(keys) == deleted_keys
        return {"deleted_keys": keys}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )

@router.get("/test")
async def test_endpoint(request: Request):
    return {"route": request.url.path}

#### EXPERIMENTAL QUEUING #### 
@router.post("/queue/request", dependencies=[Depends(user_api_key_auth)])
async def async_queue_request(request: Request):
    global celery_fn, llm_model_list
    body = await request.body()
    body_str = body.decode()
    try:
        data = ast.literal_eval(body_str)
    except:
        data = json.loads(body_str)
    data["model"] = (
        general_settings.get("completion_model", None) # server default
        or user_model # model name passed via cli args
        or data["model"] # default passed in http request
    )
    data["llm_model_list"] = llm_model_list
    print(f"data: {data}")
    job = celery_fn.apply_async(kwargs=data)
    return {"id": job.id, "url": f"/queue/response/{job.id}", "eta": 5, "status": "queued"}
    pass

@router.get("/queue/response/{task_id}", dependencies=[Depends(user_api_key_auth)])
async def async_queue_response(request: Request, task_id: str):
    global celery_app_conn, async_result
    try:
        job = async_result(task_id, app=celery_app_conn)
        if job.ready():
            return {"status": "finished", "result": job.result}
        else:
            return {'status': 'queued'}
    except Exception as e:
        return {"status": "finished", "result": str(e)}


@router.get("/ollama_logs", dependencies=[Depends(user_api_key_auth)])
async def retrieve_server_log(request: Request):
    filepath = os.path.expanduser("~/.ollama/logs/server.log")
    return FileResponse(filepath)


#### BASIC ENDPOINTS #### 

@router.get("/test")
async def test_endpoint(request: Request): 
    return {"route": request.url.path}

@router.get("/health", description="Check the health of all the endpoints in config.yaml", tags=["health"])
async def health_endpoint(request: Request, model: Optional[str] = fastapi.Query(None, description="Specify the model name (optional)")):
    global llm_model_list
    healthy_endpoints = []
    unhealthy_endpoints = []
    if llm_model_list:
        for model_name in llm_model_list:
            try:
                if model is None or model == model_name["litellm_params"]["model"]: # if model specified, just call that one.
                    litellm_params = model_name["litellm_params"]
                    model_name = litellm.utils.remove_model_id(litellm_params["model"]) # removes, ids set by litellm.router
                    if model_name not in litellm.all_embedding_models: # filter out embedding models
                        litellm_params["messages"] = [{"role": "user", "content": "Hey, how's it going?"}]
                        litellm_params["model"] = model_name
                        litellm.completion(**litellm_params)
                        cleaned_params = {}
                        for key in litellm_params:
                            if key != "api_key" and key != "messages":
                                cleaned_params[key] = litellm_params[key]
                        healthy_endpoints.append(cleaned_params)
            except Exception as e:
                print("Got Exception", e)
                cleaned_params = {}
                for key in litellm_params:
                    if key != "api_key" and key != "messages":
                        cleaned_params[key] = litellm_params[key]
                unhealthy_endpoints.append(cleaned_params)
                pass
    return {
        "healthy_endpoints": healthy_endpoints,
        "unhealthy_endpoints": unhealthy_endpoints
    }

@router.get("/")
async def home(request: Request):
    return "LiteLLM: RUNNING"

@router.get("/routes")
async def get_routes():
    """
    Get a list of available routes in the FastAPI application.
    """
    routes = []
    for route in app.routes:
        route_info = {
            "path": route.path,
            "methods": route.methods,
            "name": route.name,
            "endpoint": route.endpoint.__name__ if route.endpoint else None,
        }
        routes.append(route_info)

    return {"routes": routes}


app.include_router(router)
