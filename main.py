import secrets
import string
from datetime import datetime

import redis
from dotenv import load_dotenv
from fastapi import FastAPI, status, Header
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import AgentExecutor
from langchain.agents import OpenAIFunctionsAgent
from langchain.chat_models.openai import ChatOpenAI
from langchain.memory import RedisChatMessageHistory
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema.messages import (
    SystemMessage,
)
from langchain.tools import StructuredTool

from io_processing import *
from logger import logger
from utils import is_url, is_base64

gpt_model = get_config_value("llm", "gpt_model", None)

system_rules = get_config_value("llm", "all_chat_prompt", None)

app = FastAPI(title="ALL BOT Service",
              #   docs_url=None,  # Swagger UI: disable it by setting docs_url=None
              redoc_url=None,  # ReDoc : disable it by setting docs_url=None
              swagger_ui_parameters={"defaultModelsExpandDepth": -1},
              description='',
              version="1.0.0"
              )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_host = get_config_value('redis', 'redis_host', None)
redis_port = get_config_value('redis', 'redis_port', None)
redis_index = get_config_value('redis', 'redis_index', None)

# Connect to Redis
redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_index)  # Adjust host and port if needed


# Define a function to store and retrieve data in Redis
def store_data(key, value):
    redis_client.set(key, value)


def retrieve_data(key):
    data_from_redis = redis_client.get(key)
    return data_from_redis.decode('utf-8') if data_from_redis is not None else None


@app.on_event("startup")
async def startup_event():
    logger.info('Invoking startup_event')
    load_dotenv()
    logger.info('startup_event : Engine created')


@app.on_event("shutdown")
async def shutdown_event():
    logger.info('Invoking shutdown_event')
    logger.info('shutdown_event : Engine closed')


class OutputResponse(BaseModel):
    audio: str = None
    text: str = None
    content_id: str = None


class ResponseForQuery(BaseModel):
    output: OutputResponse


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""
    status: str = "OK"


class QueryInputModel(BaseModel):
    session_id: str = None
    language: str = None
    text: str = None
    audio: str = None


class GetContentRequest(BaseModel):
    user_id: str = None
    language: str = None


class UserAnswerRequest(BaseModel):
    user_id: str = None
    content_id: str = None
    audio: str = None
    language: str = None
    original_text: str = None


class GetContentResponse(BaseModel):
    output: OutputResponse


class QueryModel(BaseModel):
    input: QueryInputModel


class TranscribeInput(BaseModel):
    file_url: str = Field(description="audio input")
    input_language: str = Field(description="audio language")


transcribe = StructuredTool.from_function(
    func=process_incoming_voice,
    name="Transcribe",
    description="generate transcribe of audio",
    args_schema=TranscribeInput,
    return_direct=True,
)


def _handle_error(error: ToolException) -> str:
    return (
            "The following errors occurred during tool execution:"
            + error.args[0]
            + "Please try another tool."
    )


class VoiceMakerInput(BaseModel):
    message: str = Field(description="text input")
    input_language: str = Field(description="audio language")


voice_maker = StructuredTool.from_function(
    func=process_outgoing_voice,
    name="VoiceMaker",
    description="generate audio from text in the input language",
    args_schema=VoiceMakerInput,
    return_direct=True,
    handle_tool_error=_handle_error,
)


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to ALL BOT Service"}


@app.get(
    "/health",
    tags=["Health Check"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
    include_in_schema=True
)
def get_health() -> HealthCheck:
    """
    ## Perform a Health Check
    Endpoint to perform a healthcheck on. This endpoint can primarily be used Docker
    to ensure a robust container orchestration and management is in place. Other
    services which rely on proper functioning of the API service will not deploy if this
    endpoint returns any other HTTP status code except 200 (OK).
    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    return HealthCheck(status="OK")


llm = ChatOpenAI(model=gpt_model, temperature=0)


@app.post("/v1/learn_language", include_in_schema=True)
async def query(request: QueryModel, x_request_id: str = Header(None, alias="X-Request-ID")) -> ResponseForQuery:
    load_dotenv()

    language_code_list = get_config_value('request', 'supported_lang_codes', None).split(",")
    if language_code_list is None:
        raise HTTPException(status_code=422, detail="supported_lang_codes not configured!")

    language = request.input.language.strip().lower()
    if language is None or language == "" or language not in language_code_list:
        raise HTTPException(status_code=422, detail="Unsupported language code entered!")

    audio_url = request.input.audio
    input_text = request.input.text
    session_id = request.input.session_id

    # setup Redis as a message store
    message_history = RedisChatMessageHistory(
        url="redis://" + redis_host + ":" + redis_port + "/" + redis_index, session_id=session_id
    )

    eng_text = None
    ai_assistant = None
    ai_reg_text = None

    if (input_text is None or input_text == "") and (audio_url is None or audio_url == ""):
        raise HTTPException(status_code=422, detail="Either 'text' or 'audio' should be present!")
    elif input_text is not None and audio_url is not None and input_text != "" and audio_url != "":
        raise HTTPException(status_code=422, detail="Both 'text' and 'audio' cannot be taken as input! Either 'text' "
                                                    "or 'audio' is allowed.")
    if input_text:
        eng_text, error_message = process_incoming_text(input_text, language)
    else:
        if not is_url(audio_url) and not is_base64(audio_url):
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid audio input!")
        logger.debug("audio_url:: ", audio_url)
        logger.debug("invoking audio url to text conversion")
        reg_text, eng_text, error_message = process_incoming_voice(audio_url, language)
        logger.debug("audio converted:: eng_text:: ", eng_text)

    tools = [voice_maker]

    print("system_rules:: ", system_rules)

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=SystemMessage(
            content=system_rules,
        ),
        extra_prompt_messages=[
            MessagesPlaceholder(variable_name='chat_history'),
            HumanMessagePromptTemplate.from_template("user: {input}, language: {input_language}, history: {chat_history}"),

            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )

    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    print({"input_text": input_text, "eng_text": eng_text, "chat_history": message_history.messages, "input_language": language})

    llm_response = agent_executor.invoke({"input": eng_text, "chat_history": message_history.messages, "input_language": language})

    logger.info({"llm response": llm_response})

    message_history.add_user_message(eng_text)
    try:
        ai_assistant = llm_response["output"]["eng_text"]
        message_history.add_ai_message(ai_assistant)
        audio_output_url = llm_response["output"]["audio"]
        ai_reg_text = llm_response["output"]["reg_text"]
    except:
        ai_assistant = llm_response["output"]
        message_history.add_ai_message(llm_response["output"])
        response = llm_response['output']
        audio_output_url, ai_reg_text = process_outgoing_voice_manual(response, language)

    if ai_assistant.startswith("Goodbye") and ai_assistant.endswith("See you soon!"):
        message_history.clear()

    response = ResponseForQuery(output=OutputResponse(audio=audio_output_url, text=ai_reg_text))
    return response


@app.post("/v1/get_content", include_in_schema=True)
async def fetch_content(request: GetContentRequest) -> GetContentResponse:
    language_code_list = get_config_value('request', 'supported_lang_codes', None).split(",")
    if language_code_list is None:
        raise HTTPException(status_code=422, detail="supported_lang_codes not configured!")

    language = request.language.strip().lower()
    if language is None or language == "" or language not in language_code_list:
        raise HTTPException(status_code=422, detail="Unsupported language code entered!")

    user_id = request.user_id

    # api-endpoint
    get_milestone_url = get_config_value('ALL_APIS', 'get_milestone_api', None)

    # defining a params dict for the parameters to be sent to the API
    params = {'language': language}

    # sending get request and saving the response as response object
    milestone_response = requests.get(url=get_milestone_url + user_id, params=params)

    user_milestone_level = milestone_response.json()["data"]["milestone_level"]

    # api-endpoint
    get_progress_url = get_config_value('ALL_APIS', 'get_user_progress_api', None)

    # defining a params dict for the parameters to be sent to the API
    params = {'language': language}

    # sending get request and saving the response as response object
    progress_response = requests.get(url=get_progress_url + user_id, params=params)

    progress_result = progress_response.json()["result"]

    logger.info({"progress_result": progress_result})

    if type(progress_result) is not str:
        prev_session_id = progress_result["sessionId"]

    milliseconds = round(time.time() * 1000)
    current_session_id = str(user_id) + str(milliseconds)
    print("current_session_id:: ", current_session_id)
    store_data(user_id + "_" + user_milestone_level + "_session", current_session_id)

    output_response = get_next_content(user_milestone_level, user_id, language)
    content_response = GetContentResponse(output=output_response)
    return content_response


@app.post("/v1/submit_response", include_in_schema=True)
async def submit_response(request: UserAnswerRequest) -> GetContentResponse:
    user_id = request.user_id
    audio = request.audio
    language = request.language
    content_id = request.content_id
    original_text = request.original_text

    if not is_url(audio) and not is_base64(audio):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid audio input!")
    logger.debug("audio:: ", audio)

    logger.debug("invoking audio url to text conversion")
    reg_text, eng_text, error_message = process_incoming_voice(audio, language)
    logger.debug("audio converted:: eng_text:: ", eng_text)

    # api-endpoint
    get_milestone_url = get_config_value('ALL_APIS', 'get_milestone_api', None)

    # defining a params dict for the parameters to be sent to the API
    params = {'language': language}

    # sending get request and saving the response as response object
    milestone_response = requests.get(url=get_milestone_url + user_id, params=params)

    user_milestone_level = milestone_response.json()["data"]["milestone_level"]
    logger.info({"user_milestone_level": user_milestone_level})

    current_session_id = retrieve_data(user_id + "_" + user_milestone_level + "_session")
    logger.info({"current_session_id": current_session_id})

    sub_session_id = retrieve_data(user_id + user_milestone_level + "_sub_session")
    logger.info({"sub_session_id": sub_session_id})

    in_progress_collection_category = retrieve_data(user_id + "_" + user_milestone_level + "_progress_collection_category")

    if in_progress_collection_category is None:
        in_progress_collection_category = "Sentence"

    # Get the current date
    current_date = datetime.now().date()

    # Format the date as "YYYY-MM-DD"
    formatted_date = current_date.strftime("%Y-%m-%d")

    if sub_session_id is None:
        sub_session_id = generate_sub_session_id()
        store_data(user_id + user_milestone_level + "_sub_session", sub_session_id)

    update_learner_profile = get_config_value('ALL_APIS', 'update_learner_profile', None) + language
    payload = {"audio": audio, "contentId": content_id, "contentType": in_progress_collection_category, "date": formatted_date, "language": language, "original_text": original_text, "session_id": current_session_id, "sub_session_id": sub_session_id,
               "user_id": user_id}
    headers = {
        'Content-Type': 'application/json'
    }

    logger.info({"update_learner_profile_payload": payload})
    update_learner_profile_response = requests.request("POST", update_learner_profile, headers=headers, data=json.dumps(payload))
    update_status = update_learner_profile_response.json()["status"]
    logger.info({"update_learner_profile_response": update_learner_profile_response})

    if update_status == "success":
        completed_contents = retrieve_data(user_id + "_" + user_milestone_level + "_completed_contents")
        logger.info({"completed_contents": completed_contents})
        if completed_contents and completed_contents != '[null]':
            completed_contents = json.loads(completed_contents)
            completed_contents.remove("null")
            if type(completed_contents) == list:
                completed_contents = set(completed_contents)
            logger.info({"Bef completed_contents": completed_contents})
            logger.info({"content_id": content_id})
            completed_contents.add(content_id)
            logger.info({"Aft completed_contents": completed_contents})
        else:
            completed_contents = {content_id}
        completed_contents = list(completed_contents)
        logger.info({"updated_completed_contents": completed_contents})
        store_data(user_id + "_" + user_milestone_level + "_completed_contents", json.dumps(completed_contents))
    else:
        raise HTTPException(500, "Submitted response could not be registered!")

    output_response = get_next_content(user_milestone_level, user_id, language)

    content_response = GetContentResponse(output=output_response)
    return content_response


def generate_sub_session_id(length=24):
    # Define the set of characters to choose from
    characters = string.ascii_letters + string.digits

    # Generate a random session ID
    sub_session_id = ''.join(secrets.choice(characters) for _ in range(length))

    return sub_session_id


def get_next_content(user_milestone_level, user_id, language) -> OutputResponse:
    stored_user_assessment_collections: str = retrieve_data(user_id + "_" + user_milestone_level + "_collections")

    user_assessment_collections: dict = {}
    if stored_user_assessment_collections:
        user_assessment_collections = json.loads(stored_user_assessment_collections)

    logger.info({"Redis user_assessment_collections": user_assessment_collections})

    if stored_user_assessment_collections is None:
        user_assessment_collections: dict = {}
        get_assessment_api = get_config_value('ALL_APIS', 'get_assessment_api', None)
        payload = {"tags": ["ASER"], "language": language}
        headers = {
            'Content-Type': 'application/json'
        }

        get_assessment_response = requests.request("POST", get_assessment_api, headers=headers, data=json.dumps(payload))

        logger.info({"get_assessment_response": get_assessment_response})

        assessment_data = get_assessment_response.json()["data"]

        logger.info({"assessment_data": assessment_data})
        for collections in assessment_data:
            if collections["category"] == "Sentence" or collections["category"] == "Word":
                if user_assessment_collections is None or collections["category"] not in user_assessment_collections.keys():
                    user_assessment_collections = {collections["category"]: collections}
                elif collections["category"] in user_assessment_collections.keys() and user_milestone_level in collections["tags"]:
                    user_assessment_collections[collections["category"]] = collections

        logger.info({"user_assessment_collections": json.dumps(user_assessment_collections)})
        store_data(user_id + "_" + user_milestone_level + "_collections", json.dumps(user_assessment_collections))

    completed_collections = retrieve_data(user_id + "_" + user_milestone_level + "_completed_collections")
    in_progress_collection = retrieve_data(user_id + "_" + user_milestone_level + "_progress_collection")

    if completed_collections:
        completed_collections = json.loads(completed_collections)
        for completed_collection in completed_collections:
            user_assessment_collections = {key: val for key, val in user_assessment_collections.items() if val.get("collectionId") != completed_collection}
    else:
        completed_collections = []

    current_collection = None

    if in_progress_collection:
        for collection_value in user_assessment_collections.values():
            if collection_value.get("collectionId") == in_progress_collection:
                current_collection = collection_value
    else:
        current_collection = list(user_assessment_collections.values())[0]
        logger.info({"current_collection": current_collection})
        store_data(user_id + "_" + user_milestone_level + "_progress_collection", current_collection.get("collectionId"))
        store_data(user_id + "_" + user_milestone_level + "_progress_collection_category", current_collection.get("category"))

    logger.info({"current_collection": current_collection})

    completed_contents = retrieve_data(user_id + "_" + user_milestone_level + "_completed_contents")
    logger.info({"completed_contents": completed_contents})
    if completed_contents:
        completed_contents = json.loads(completed_contents)
        for content_id in completed_contents:
            print("\ncontent_id:: ", content_id)
            for content in current_collection.get("content"):
                print("\ncontent:: ", content)
                if content.get("contentId") == content_id:
                    print("\n MATCH FOUND!! Removing content from collection")
                    current_collection.get("content").remove(content)

    logger.info({"updated_current_collection": current_collection})

    if "content" in current_collection.keys() and len(current_collection.get("content")) == 0:
        store_data(user_id + "_" + user_milestone_level + "_completed_collections", completed_collections.append(current_collection.get("collectionId")))
        user_assessment_collections = {key: val for key, val in user_assessment_collections.items() if val.get("collectionId") != current_collection.get("collectionId")}
        if len(user_assessment_collections) != 0:
            current_collection = user_assessment_collections.get(0)
            logger.info({"current_collection": current_collection})
            store_data(user_id + "_" + user_milestone_level + "_progress_collection", current_collection.get("collectionId"))
        else:
            output = OutputResponse(audio="", text="Congratulations! You have completed the assessment")
            return output

    content_source_data = current_collection.get("content")[0].get("contentSourceData")[0]
    logger.info({"content_source_data": content_source_data})

    output = OutputResponse(audio=content_source_data.get("audioUrl"), text=content_source_data.get("text"), content_id=current_collection.get("content")[0].get("contentId"))
    return output
