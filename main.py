import json
import secrets
import string
from datetime import datetime
from typing import Optional

import openai
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


class ConversationResponse(BaseModel):
    audio: str = None
    text: str = None


class ResponseForQuery(BaseModel):
    output: ConversationResponse


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""
    status: str = "OK"


class QueryInputModel(BaseModel):
    user_id: str = None
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
    conversation: Optional[ConversationResponse]
    content: Optional[OutputResponse]


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
async def query(request: QueryInputModel, x_request_id: str = Header(None, alias="X-Request-ID")) -> GetContentResponse:
    load_dotenv()

    language_code_list = get_config_value('request', 'supported_lang_codes', None).split(",")
    if language_code_list is None:
        raise HTTPException(status_code=422, detail="supported_lang_codes not configured!")

    language = request.language.strip().lower()
    if language is None or language == "" or language not in language_code_list:
        raise HTTPException(status_code=422, detail="Unsupported language code entered!")

    audio_url = request.audio
    input_text = request.text
    user_id = request.user_id

    current_session_id = retrieve_data(user_id + "_" + language + "_session")

    if current_session_id is None:
        milliseconds = round(time.time() * 1000)
        current_session_id = str(user_id) + str(milliseconds)
        store_data(user_id + "_" + language + "_session", current_session_id)
    logger.info({"user_id": user_id, "current_session_id": current_session_id})

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

    content_response = invoke_llm(user_id, language, current_session_id, eng_text)
    return content_response


@app.post("/v1/get_content", include_in_schema=True)
async def fetch_content(request: GetContentRequest) -> GetContentResponse:
    language_code_list = get_config_value('request', 'supported_lang_codes', None).split(",")
    if language_code_list is None:
        raise HTTPException(status_code=422, detail="supported_lang_codes not configured!")

    language = request.language
    if language is None or language == "" or language not in language_code_list:
        raise HTTPException(status_code=422, detail="Unsupported language code entered!")

    user_id = request.user_id
    learning_language = get_config_value('request', 'learn_language', None)

    content_response = func_get_content(user_id, learning_language)
    return content_response


def func_get_content(user_id, language) -> GetContentResponse:
    # api-endpoint
    get_milestone_url = get_config_value('ALL_APIS', 'get_milestone_api', None)

    # defining a params dict for the parameters to be sent to the API
    params = {'language': language}

    # sending get request and saving the response as response object
    milestone_response = requests.get(url=get_milestone_url + user_id, params=params)

    user_milestone_level = milestone_response.json()["data"]["milestone_level"]

    # # api-endpoint
    # get_progress_url = get_config_value('ALL_APIS', 'get_user_progress_api', None)
    #
    # # defining a params dict for the parameters to be sent to the API
    # params = {'language': language}
    #
    # # sending get request and saving the response as response object
    # progress_response = requests.get(url=get_progress_url + user_id, params=params)
    #
    # progress_result = progress_response.json()["result"]
    #
    # logger.info({"user_id": user_id, "progress_result": progress_result})
    #
    # if type(progress_result) is not str:
    #     prev_session_id = progress_result["sessionId"]

    mode = get_config_value('request', 'mode', None)
    current_session_id = retrieve_data(user_id + "_" + language + "_session")

    if current_session_id is None:
        milliseconds = round(time.time() * 1000)
        current_session_id = str(user_id) + str(milliseconds)
    logger.info({"user_id": user_id, "current_session_id": current_session_id})

    if mode == "discovery":
        store_data(user_id + "_" + language + "_" + user_milestone_level + "_session", current_session_id)
        output_response = get_discovery_content(user_milestone_level, user_id, language, current_session_id)
    else:
        store_data(user_id + "_" + language + "_session", current_session_id)
        output_response = get_showcase_content(user_id, language)

    match language:
        case "kn":
            conversation_text = "ನಿಮಗೆ ಸ್ವಾಗತ! ನೀವು ಪ್ರಯತ್ನಿಸಬೇಕಾದ ಪದ ಇಲ್ಲಿದೆ: "
            conversation_audio = "https://ax2cel5zyviy.compat.objectstorage.ap-hyderabad-1.oraclecloud.com/sbdjb-kathaasaagara/audio-output-20240228-070142.mp3"
        case _:
            conversation_text = "Welcome! Here is the word to try:"
            conversation_audio = "https://ax2cel5zyviy.compat.objectstorage.ap-hyderabad-1.oraclecloud.com/sbdjb-kathaasaagara/audio-output-20240228-065640.mp3"

    conversation_response = ConversationResponse(audio=conversation_audio, text=conversation_text)

    content_response = GetContentResponse(conversation=conversation_response, content=output_response)
    return content_response


@app.post("/v1/submit_response", include_in_schema=True)
async def submit_response(request: UserAnswerRequest) -> GetContentResponse:
    user_id = request.user_id
    audio = request.audio
    content_id = request.content_id
    original_text = request.original_text

    language_code_list = get_config_value('request', 'supported_lang_codes', None).split(",")
    if language_code_list is None:
        raise HTTPException(status_code=422, detail="supported_lang_codes not configured!")

    language = request.language
    if language is None or language == "" or language not in language_code_list:
        raise HTTPException(status_code=422, detail="Unsupported language code entered!")

    if user_id is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid user_id input!")

    if original_text is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid original_text input!")

    if content_id is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid content_id input!")

    if not is_url(audio) and not is_base64(audio):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid audio input!")
    logger.debug("audio:: ", audio)

    learning_language = get_config_value('request', 'learn_language', None)

    logger.debug("invoking audio url to text conversion")
    reg_text, eng_text, error_message = process_incoming_voice(audio, language)
    logger.info({"user_id": user_id, "audio_converted_eng_text:": eng_text})

    mode = get_config_value('request', 'mode', None)
    user_milestone_level = None
    if mode == "discovery":
        # api-endpoint
        get_milestone_url = get_config_value('ALL_APIS', 'get_milestone_api', None)

        # defining a params dict for the parameters to be sent to the API
        params = {'language': learning_language}

        # sending get request and saving the response as response object
        milestone_response = requests.get(url=get_milestone_url + user_id, params=params)

        user_milestone_level = milestone_response.json()["data"]["milestone_level"]
        logger.info({"user_id": user_id, "user_milestone_level": user_milestone_level})
        current_session_id = retrieve_data(user_id + "_" + language + "_" + user_milestone_level + "_session")
        sub_session_id = retrieve_data(user_id + "_" + language + "_" + user_milestone_level + "_sub_session")
        in_progress_collection_category = retrieve_data(user_id + "_" + language + "_" + user_milestone_level + "_progress_collection_category")
        if in_progress_collection_category is None:
            in_progress_collection_category = "Word"
            store_data(user_id + "_" + language + "_" + user_milestone_level + "_progress_collection_category", in_progress_collection_category)
    else:
        current_session_id = retrieve_data(user_id + "_" + language + "_session")
        sub_session_id = retrieve_data(user_id + "_" + language + "_sub_session")

        in_progress_collection_category = retrieve_data(user_id + "_" + language + "_progress_collection_category")
        if in_progress_collection_category is None:
            in_progress_collection_category = "Word"
            store_data(user_id + "_" + language + "_progress_collection_category", in_progress_collection_category)

    logger.info({"user_id": user_id, "current_session_id": current_session_id})
    logger.info({"user_id": user_id, "sub_session_id": sub_session_id})

    if current_session_id is None:
        return func_get_content(user_id=user_id, language=learning_language)

    # Get the current date
    current_date = datetime.now().date()

    # Format the date as "YYYY-MM-DD"
    formatted_date = current_date.strftime("%Y-%m-%d")

    if sub_session_id is None:
        sub_session_id = generate_sub_session_id()
        if mode == "discovery":
            store_data(user_id + "_" + language + "_" + user_milestone_level + "_sub_session", sub_session_id)
        else:
            store_data(user_id + "_" + language + "_sub_session", sub_session_id)

    update_learner_profile = get_config_value('ALL_APIS', 'update_learner_profile', None) + learning_language
    payload = {"audio": audio, "contentId": content_id, "contentType": in_progress_collection_category, "date": formatted_date, "language": learning_language, "original_text": original_text, "session_id": current_session_id,
               "sub_session_id": sub_session_id,
               "user_id": user_id}
    headers = {
        'Content-Type': 'application/json'
    }

    logger.debug({"user_id": user_id, "update_learner_profile_payload": payload})
    update_learner_profile_response = requests.request("POST", update_learner_profile, headers=headers, data=json.dumps(payload))
    update_status = update_learner_profile_response.json()["status"]
    logger.debug({"user_id": user_id, "update_learner_profile_response": update_learner_profile_response})

    if update_status == "success":
        if mode == "discovery":
            completed_contents = retrieve_data(user_id + "_" + language + "_" + user_milestone_level + "_completed_contents")
        else:
            completed_contents = retrieve_data(user_id + "_" + language + "_completed_contents")
        logger.debug({"user_id": user_id, "completed_contents": completed_contents})
        if completed_contents:
            completed_contents = json.loads(completed_contents)
            if type(completed_contents) == list:
                completed_contents = set(completed_contents)
            completed_contents.add(content_id)
        else:
            completed_contents = {content_id}
        completed_contents = list(completed_contents)
        logger.debug({"user_id": user_id, "updated_completed_contents": completed_contents})
        if mode == "discovery":
            store_data(user_id + "_" + language + "_" + user_milestone_level + "_completed_contents", json.dumps(completed_contents))
        else:
            store_data(user_id + "_" + language + "_completed_contents", json.dumps(completed_contents))
    else:
        raise HTTPException(500, "Submitted response could not be registered!")

    if mode == "discovery":
        output_response = get_discovery_content(user_milestone_level, user_id, language, current_session_id)
    else:
        output_response = get_showcase_content(user_id, language)

    conversation_text = None
    conversation_audio = None

    logger.info({"user_id": user_id, "output_response": output_response})

    if output_response.content_id is not None:
        # match language:
        #     case "kn":
        #         conversation_text = "ಉತ್ತಮ ಪ್ರಯತ್ನ! ಮುಂದಿನ ಪದ ಇಲ್ಲಿದೆ: "
        #         conversation_audio = "https://ax2cel5zyviy.compat.objectstorage.ap-hyderabad-1.oraclecloud.com/sbdjb-kathaasaagara/audio-output-20240228-112329.mp3"
        #     case _:
        #         conversation_text = "Well tried! Here is the next word:"
        #         conversation_audio = "https://ax2cel5zyviy.compat.objectstorage.ap-hyderabad-1.oraclecloud.com/sbdjb-kathaasaagara/audio-output-20240228-062956.mp3"

        llm_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        nudge_user_prompt = get_config_value("llm", "nudge_user_prompt", None)
        logger.debug(f"nudge_user_prompt: {nudge_user_prompt}")
        nudge_message = None
        if nudge_user_prompt:
            nudge_res = llm_client.chat.completions.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": nudge_user_prompt}
                ],
            )
            nudge_res_message = nudge_res.choices[0].message.model_dump()
            nudge_message = nudge_res_message["content"]
            logger.info({"user_id": user_id, "user_language": language, "nudge_message": nudge_message})

        conversation_audio, conversation_text = process_outgoing_voice_manual(nudge_message, language)

    else:
        content_response = invoke_llm(user_id, language, current_session_id, 'user_completed')
        return content_response
        # match language:
        #     case "kn":
        #         conversation_text = "ನೀವು ಮೌಲ್ಯಮಾಪನವನ್ನು ಪೂರ್ಣಗೊಳಿಸಿದ್ದೀರಿ! ಹೊಸದಾಗಿ ಪ್ರಾರಂಭಿಸಲು ಮರು-ಲಾಗಿನ್ ಮಾಡಿ."
        #         conversation_audio = "https://ax2cel5zyviy.compat.objectstorage.ap-hyderabad-1.oraclecloud.com/sbdjb-kathaasaagara/audio-output-20240228-064932.mp3"
        #     case _:
        #         conversation_text = "You have completed the assessment! Re-login to start fresh."
        #         conversation_audio = "https://ax2cel5zyviy.compat.objectstorage.ap-hyderabad-1.oraclecloud.com/sbdjb-kathaasaagara/audio-output-20240228-065008.mp3"

    conversation_response = ConversationResponse(audio=conversation_audio, text=conversation_text)

    if output_response.audio is not None:
        content_response = GetContentResponse(conversation=conversation_response, content=output_response)
    else:
        content_response = GetContentResponse(conversation=conversation_response)
    return content_response


user_learning_emotions = json.loads(get_config_value('request', 'user_emotions_response', None))


def invoke_llm(user_id, language, current_session_id, user_input) -> GetContentResponse:
    # setup Redis as a message store
    message_history = RedisChatMessageHistory(
        url="redis://" + redis_host + ":" + redis_port + "/" + redis_index, session_id=current_session_id
    )

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

    llm_response = agent_executor.invoke({"input": user_input, "chat_history": message_history.messages, "input_language": language})

    logger.info({"llm response": llm_response})

    message_history.add_user_message(user_input)

    try:
        ai_assistant = llm_response["output"]["eng_text"]
        if ai_assistant in user_learning_emotions.keys():
            ai_assistant = user_learning_emotions.get(ai_assistant)
            message_history.add_ai_message(ai_assistant)
            audio_output_url, ai_reg_text = process_outgoing_voice_manual(ai_assistant, language)
        else:
            message_history.add_ai_message(ai_assistant)
            audio_output_url = llm_response["output"]["audio"]
            ai_reg_text = llm_response["output"]["reg_text"]
    except:
        ai_assistant = llm_response["output"]
        if ai_assistant in user_learning_emotions.keys():
            ai_assistant = user_learning_emotions.get(ai_assistant)
        message_history.add_ai_message(ai_assistant)
        audio_output_url, ai_reg_text = process_outgoing_voice_manual(ai_assistant, language)

    if ai_assistant.startswith("Goodbye") and ai_assistant.endswith("See you soon!"):
        message_history.clear()

    if ai_assistant == "user_agreed" or ai_assistant == "'user_agreed'" :
        content_response = func_get_content(user_id, language)
    else:
        conversation_text = ai_reg_text
        conversation_audio = audio_output_url
        conversation_response = ConversationResponse(audio=conversation_audio, text=conversation_text)
        content_response = GetContentResponse(conversation=conversation_response)

    logger.info({"content_response": content_response})
    return content_response


def generate_sub_session_id(length=24):
    # Define the set of characters to choose from
    characters = string.ascii_letters + string.digits

    # Generate a random session ID
    sub_session_id = ''.join(secrets.choice(characters) for _ in range(length))

    return sub_session_id


def get_discovery_content(user_milestone_level, user_id, language, session_id) -> OutputResponse:
    stored_user_assessment_collections: str = retrieve_data(user_id + "_" + language + "_" + user_milestone_level + "_collections")
    headers = {
        'Content-Type': 'application/json'
    }
    user_assessment_collections: dict = {}
    if stored_user_assessment_collections:
        user_assessment_collections = json.loads(stored_user_assessment_collections)

    logger.info({"user_id": user_id, "Redis user_assessment_collections": user_assessment_collections})

    learning_language = get_config_value('request', 'learn_language', None)

    if stored_user_assessment_collections is None:
        user_assessment_collections: dict = {}
        get_assessment_api = get_config_value('ALL_APIS', 'get_assessment_api', None)
        payload = {"tags": ["ASER"], "language": learning_language}

        get_assessment_response = requests.request("POST", get_assessment_api, headers=headers, data=json.dumps(payload))
        logger.info({"user_id": user_id, "get_assessment_response": get_assessment_response})

        assessment_data = get_assessment_response.json()["data"]
        logger.info({"user_id": user_id, "assessment_data": assessment_data})
        for collection in assessment_data:
            if collection["category"] == "Sentence" or collection["category"] == "Word":
                if user_assessment_collections is None:
                    user_assessment_collections = {collection["category"]: collection}
                elif collection["category"] not in user_assessment_collections.keys():
                    user_assessment_collections.update({collection["category"]: collection})
                elif collection["category"] in user_assessment_collections.keys() and user_milestone_level in collection["tags"]:
                    user_assessment_collections.update({collection["category"]: collection})

        logger.info({"user_id": user_id, "user_assessment_collections": json.dumps(user_assessment_collections)})
        store_data(user_id + "_" + language + "_" + user_milestone_level + "_collections", json.dumps(user_assessment_collections))

    completed_collections = retrieve_data(user_id + "_" + language + "_" + user_milestone_level + "_completed_collections")
    logger.info({"user_id": user_id, "completed_collections": completed_collections})
    in_progress_collection = retrieve_data(user_id + "_" + language + "_" + user_milestone_level + "_progress_collection")
    logger.info({"user_id": user_id, "in_progress_collection": in_progress_collection})

    if completed_collections and in_progress_collection and in_progress_collection in json.loads(completed_collections):
        in_progress_collection = None

    if completed_collections:
        completed_collections = json.loads(completed_collections)
        for completed_collection in completed_collections:
            user_assessment_collections = {key: val for key, val in user_assessment_collections.items() if val.get("collectionId") != completed_collection}

    current_collection = None

    if in_progress_collection:
        for collection_value in user_assessment_collections.values():
            if collection_value.get("collectionId") == in_progress_collection:
                logger.debug({"user_id": user_id, "setting_current_collection_using_in_progress_collection": collection_value})
                current_collection = collection_value
    elif len(user_assessment_collections.values()) > 0:
        current_collection = list(user_assessment_collections.values())[0]
        logger.debug({"user_id": user_id, "setting_current_collection_using_assessment_collections": current_collection})
        store_data(user_id + "_" + language + "_" + user_milestone_level + "_progress_collection", current_collection.get("collectionId"))
        store_data(user_id + "_" + language + "_" + user_milestone_level + "_progress_collection_category", current_collection.get("category"))
    else:
        # redis_client.delete(user_id + "_" + language + "_" + user_milestone_level + "_collections")
        # redis_client.delete(user_id + "_" + language + "_" + user_milestone_level + "_completed_collections")
        # redis_client.delete(user_id + "_" + language + "_" + user_milestone_level + "_progress_collection")
        # redis_client.delete(user_id + "_" + language + "_" + user_milestone_level + "_progress_collection_category")
        # redis_client.delete(user_id + "_" + language + "_" + user_milestone_level + "_completed_contents")
        # redis_client.delete(user_id + "_" + language + "_" + user_milestone_level + "_session")
        # redis_client.delete(user_id + "_" + language + "_" + user_milestone_level + "_sub_session")
        output = OutputResponse(audio="completed", text="completed")
        return output

    logger.info({"user_id": user_id, "current_collection": current_collection})

    completed_contents = retrieve_data(user_id + "_" + language + "_" + user_milestone_level + "_completed_contents")
    logger.debug({"user_id": user_id, "completed_contents": completed_contents})
    if completed_contents:
        completed_contents = json.loads(completed_contents)
        for content_id in completed_contents:
            for content in current_collection.get("content"):
                if content.get("contentId") == content_id:
                    current_collection.get("content").remove(content)

    logger.info({"user_id": user_id, "updated_current_collection": current_collection})

    if "content" not in current_collection.keys() or len(current_collection.get("content")) == 0:
        if completed_collections:
            completed_collections.append(current_collection.get("collectionId"))
        else:
            completed_collections = [current_collection.get("collectionId")]
        store_data(user_id + "_" + language + "_" + user_milestone_level + "_completed_collections", json.dumps(completed_collections))
        user_assessment_collections = {key: val for key, val in user_assessment_collections.items() if val.get("collectionId") != current_collection.get("collectionId")}

        logger.info({"user_id": user_id, "completed_collection_id": current_collection.get("collectionId"), "after_removing_completed_collection_user_assessment_collections": user_assessment_collections})

        add_lesson_api = get_config_value('ALL_APIS', 'add_lesson_api', None)
        add_lesson_payload = {"userId": user_id, "sessionId": session_id, "milestone": "discoverylist/discovery/" + current_collection.get("collectionId"), "lesson": current_collection.get("name"), "progress": 100,
                              "milestoneLevel": user_milestone_level, "language": learning_language}
        add_lesson_response = requests.request("POST", add_lesson_api, headers=headers, data=json.dumps(add_lesson_payload))
        logger.info({"user_id": user_id, "add_lesson_response": add_lesson_response})

        if len(user_assessment_collections) != 0:
            current_collection = list(user_assessment_collections.values())[0]
            logger.info({"user_id": user_id, "current_collection": current_collection})
            store_data(user_id + "_" + language + "_" + user_milestone_level + "_progress_collection", current_collection.get("collectionId"))
        else:
            # get_result_api = get_config_value('ALL_APIS', 'get_result_api', None)
            # get_result_payload = {"sub_session_id": sub_session_id, "contentType": current_collection.get("category"), "session_id": session_id, "user_id": user_id, "collectionId": current_collection.get("collectionId"), "language": language}
            # get_result_response = requests.request("POST", get_result_api, headers=headers, data=json.dumps(get_result_payload))
            # logger.info({"user_id": user_id, "get_result_response": get_result_response})
            # percentage = get_result_response.json()["data"]["percentage"]
            #
            # redis_client.delete(user_id + "_" + language + "_" + user_milestone_level + "_collections")
            # redis_client.delete(user_id + "_" + language + "_" + user_milestone_level + "_completed_collections")
            # redis_client.delete(user_id + "_" + language + "_" + user_milestone_level + "_progress_collection")
            # redis_client.delete(user_id + "_" + language + "_" + user_milestone_level + "_progress_collection_category")
            # redis_client.delete(user_id + "_" + language + "_" + user_milestone_level + "_completed_contents")
            # redis_client.delete(user_id + "_" + language + "_" + user_milestone_level + "_session")
            # redis_client.delete(user_id + "_" + language + "_" + user_milestone_level + "_sub_session")
            output = OutputResponse(audio="Completed", text="Completed")
            return output

    content_source_data = current_collection.get("content")[0].get("contentSourceData")[0]
    logger.debug({"user_id": user_id, "content_source_data": content_source_data})
    content_id = current_collection.get("content")[0].get("contentId")
    audio_url = "https://all-dev-content-service.s3.ap-south-1.amazonaws.com/Audio/" + content_id + ".wav"

    output = OutputResponse(audio=audio_url, text=content_source_data.get("text"), content_id=content_id)
    return output


def get_showcase_content(user_id, language) -> OutputResponse:
    current_content = None
    stored_user_showcase_contents: str = retrieve_data(user_id + "_" + language + "_showcase_contents")
    user_showcase_contents = []
    if stored_user_showcase_contents:
        user_showcase_contents = json.loads(stored_user_showcase_contents)

    logger.info({"user_id": user_id, "Redis stored_user_showcase_contents": stored_user_showcase_contents})

    learning_language = get_config_value('request', 'learn_language', None)

    if stored_user_showcase_contents is None:
        get_showcase_contents_api = get_config_value('ALL_APIS', 'get_showcase_contents_api', None) + user_id
        content_limit = int(get_config_value('request', 'content_limit', None))
        target_limit = int(get_config_value('request', 'target_limit', None))
        # defining a params dict for the parameters to be sent to the API
        params = {'language': learning_language, 'contentlimit': content_limit, 'gettargetlimit': target_limit}
        # sending get request and saving the response as response object
        showcase_contents_response = requests.get(url=get_showcase_contents_api + user_id, params=params)
        user_showcase_contents = showcase_contents_response.json()["content"]
        logger.info({"user_id": user_id, "user_showcase_contents": user_showcase_contents})
        store_data(user_id + "_" + language + "_showcase_contents", json.dumps(user_showcase_contents))

    completed_contents = retrieve_data(user_id + "_" + language + "_completed_contents")
    logger.info({"user_id": user_id, "completed_contents": completed_contents})
    in_progress_content = retrieve_data(user_id + "_" + language + "_progress_content")
    logger.info({"user_id": user_id, "progress_content": in_progress_content})

    if completed_contents and in_progress_content and in_progress_content in json.loads(completed_contents):
        in_progress_content = None

    if completed_contents:
        completed_contents = json.loads(completed_contents)
        for completed_content in completed_contents:
            for showcase_content in user_showcase_contents:
                if showcase_content.get("contentId") == completed_content:
                    user_showcase_contents.remove(showcase_content)

    if in_progress_content is None and len(user_showcase_contents) > 0:
        current_content = user_showcase_contents[0]
        store_data(user_id + "_" + language + "_progress_content", current_content.get("contentId"))
    elif in_progress_content is not None and len(user_showcase_contents) > 0:
        for showcase_content in user_showcase_contents:
            if showcase_content.get("contentId") == in_progress_content:
                current_content = showcase_content
    else:
        # redis_client.delete(user_id + "_" + language + "_contents")
        # redis_client.delete(user_id + "_" + language + "_progress_content")
        # redis_client.delete(user_id + "_" + language + "_completed_contents")
        # redis_client.delete(user_id + "_" + language + "_session")
        # redis_client.delete(user_id + "_" + language + "_sub_session")

        output = OutputResponse(audio="Completed", text="Completed")
        return output

    logger.info({"user_id": user_id, "current_content": current_content})
    content_source_data = current_content.get("contentSourceData")[0]
    logger.debug({"user_id": user_id, "content_source_data": content_source_data})
    content_id = current_content.get("contentId")
    audio_url = "https://all-dev-content-service.s3.ap-south-1.amazonaws.com/Audio/" + content_id + ".wav"

    output = OutputResponse(audio=audio_url, text=content_source_data.get("text"), content_id=content_id)
    return output
