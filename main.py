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

    redis_host = get_config_value('redis', 'redis_host', None)
    redis_port = get_config_value('redis', 'redis_port', None)
    redis_index = get_config_value('redis', 'redis_index', None)

    # setup Redis as a message store
    message_history = RedisChatMessageHistory(
        url="redis://"+redis_host+":"+redis_port+"/"+redis_index, session_id=session_id
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
