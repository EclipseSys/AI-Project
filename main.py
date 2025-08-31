# main.py
import os
import uuid
import json
import asyncio
import datetime
from typing import List, Optional, Dict

import aiohttp
import requests
from fastapi import (
    FastAPI, HTTPException, Depends, Header, BackgroundTasks,
    Form, File, UploadFile, Body
)
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field

# --------------------------------------------------------------------------- #
#  CONFIGURATION
# --------------------------------------------------------------------------- #
KEYS_FILE = "keys.json"
HISTORIES_FILE = "histories.json"
MAX_TOKENS = 130_000
IMAGE_CONTENT_TYPE = "image/jpeg"

AI_MODELS = {
    "openai/gpt-oss-20b": [ # User-facing model name
        {
            "name": "openai/gpt-oss-20b CLUSTER ONE", # Internal identifier
            "endpoint": "",
            "key": "",
            "model": "openai/gpt-oss-20b",
        },
    ],
    "openai/gpt-oss-120b": [ # User-facing model name
        {
            "name": "openai/gpt-oss-120b CLUSTER ONE", # Internal identifier
            "endpoint": "",
            "key": "",
            "model": "openai/gpt-oss-120b",
        }
    ]
}

VISION_MODELS = {
    "meta-llama/llama-4-maverick-17b-128e-instruct": [
        {
            "name": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "endpoint": "https://api.groq.com/openai/v1/chat/completions",
            "key":  "",
            "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
        }
    ]
}

# --------------------------------------------------------------------------- #
#  PERSISTENCE HELPERS
# --------------------------------------------------------------------------- #

def load_json(path: str, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# Global in‑memory copies – kept sync with disk by background tasks
api_keys: Dict[str, str] = load_json(KEYS_FILE, {})                # email -> api_key
keys_by_api: Dict[str, str] = {v: k for k, v in api_keys.items()}  # api_key -> email
thread_histories: Dict[str, Dict] = load_json(HISTORIES_FILE, {})  # api_key -> {thread_id: [messages]}

# --------------------------------------------------------------------------- #
#  SECURITY DEPENDENCY
# --------------------------------------------------------------------------- #

http_bearer = HTTPBearer(auto_error=False)

async def get_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer)):
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    key = credentials.credentials
    if key not in keys_by_api:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return key

# --------------------------------------------------------------------------- #
#  AI HELPERS
# --------------------------------------------------------------------------- #

def token_count(messages: List[dict]) -> int:
    """Character‑based token counter – replace with tiktoken if you want exact counts."""
    return sum(len(msg.get("content", "")) for msg in messages)


def call_ai_model(messages: List[dict], username: str, user_system_prompt: str, modelname: Optional[str] = None) -> str:
    """
    Tries the preferred model first, then falls back to the list.
    Returns a string – the assistant’s reply.
    """
    system_prompt_content = user_system_prompt.replace("{username}", username)

    system_prompt_message = {"role": "system", "content": system_prompt_content}
    full_messages = [system_prompt_message] + messages

    headers = {"Content-Type": "application/json"}

    # Helper to attempt a single model configuration
    def try_model_config(model_config):
        payload = {"model": model_config["model"], "messages": full_messages}
        headers["Authorization"] = f"Bearer {model_config['key']}"
        try:
            r = requests.post(model_config["endpoint"], headers=headers, json=payload, timeout=30)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            print(f"[{model_config['name']}] error: {exc}")
        return None

    if modelname and modelname in AI_MODELS:
        for model_config in AI_MODELS[modelname]:
            ans = try_model_config(model_config)
            if ans:
                return ans
        return "⚠️ All configurations for the requested AI model are currently unavailable. Please try again later."

    return "⚠️ No suitable AI model found or all AI models are currently unavailable. Please try again later."

async def analyze_image(messages: List[dict], username: str, user_system_prompt: str, modelname: Optional[str] = None, image_bytes: Optional[bytes] = None) -> Optional[str]:
    """Feed an image to the vision model – returns the textual description."""
    data_url = f"data:{IMAGE_CONTENT_TYPE};base64,{image_bytes.decode('utf-8')}" if image_bytes else None

    system_prompt_content = user_system_prompt.replace("{username}", username)
    system_prompt_message = {"role": "system", "content": system_prompt_content}

    full_messages = [system_prompt_message] + list(messages)
    if image_bytes and full_messages and full_messages[-1]["role"] == "user":
        last_user_message = full_messages[-1]
        if isinstance(last_user_message["content"], str):
            last_user_message["content"] = [
                {"type": "text", "text": last_user_message["content"]}
            ]
        last_user_message["content"].append({"type": "image_url", "image_url": {"url": data_url}})
    elif image_bytes and (not full_messages or full_messages[-1]["role"] != "user"):
        full_messages.append(
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}
        )
    elif not image_bytes and modelname not in VISION_MODELS:
        return None

    async def try_vision_model_config(model_config):
        headers = {
            "Authorization": f"Bearer {model_config['key']}",
            "Content-Type": "application/json",
        }
        payload = {"model": model_config["model"], "messages": full_messages}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(model_config["endpoint"], headers=headers, json=payload, timeout=30) as r:
                    if r.status == 200:
                        return (await r.json())["choices"][0]["message"]["content"]
        except Exception as exc:
            print(f"[Vision - {model_config['name']}] {exc}")
        return None

    if modelname and modelname in VISION_MODELS:
        for model_config in VISION_MODELS[modelname]:
            ans = await try_vision_model_config(model_config)
            if ans:
                return ans
        return None 

    return None

# --------------------------------------------------------------------------- #
#  FASTAPI SETUP
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Eclipse Systems AI Project",
    description="Eclipse Systems AI Project supports text, image, and file inputs; per‑key histories; AI model selector; key registration & revocation.",
    version="0.0.1-ALPHA",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
#  Pydantic MODELS
# --------------------------------------------------------------------------- #

class RegisterRequest(BaseModel):
    email: EmailStr = Field(..., description="Email address for registration.")

class RegisterResponse(BaseModel):
    api_key: str = Field(..., description="Newly generated API key.")

class ChatRequest(BaseModel):
    thread_id: Optional[str] = Field(None, description="Use a previously created thread ID; otherwise a new one is created.")
    username: str = Field(..., description="The user's display name for context in the system prompt.")
    content: str = Field(..., description="The textual message from the user.")
    modelname: Optional[str] = Field(None, description="User-facing AI model name (e.g., 'llama', 'deepseek', 'groq-vision'). If an image is provided, a vision model is required. If no modelname is specified and no image is provided, a default AI model will be used. Fallback to the list if unavailable.")
    system_prompt: str = Field(..., description="System prompt to guide the AI model's behavior. Use {username} as a placeholder for the user's display name.")

class DeleteKeyRequest(BaseModel):
    api_key: str = Field(..., description="API key to be revoked.")

class DeleteKeyResponse(BaseModel):
    detail: str = Field(..., description="Confirmation message of key deletion.")

# --------------------------------------------------------------------------- #
#  ENDPOINTS
# --------------------------------------------------------------------------- #

@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


@app.get("/models", summary="List available AI and Vision models", description="Retrieves a list of all currently configured AI and Vision models available for use.")
async def list_models():
    return {"ai_models": list(AI_MODELS.keys()), "vision_models": list(VISION_MODELS.keys())}


@app.post("/register", response_model=RegisterResponse, summary="Create a new API key", description="Register a new email address to receive a unique API key. Each email can only be registered once.")
async def register(req: RegisterRequest):
    """Register a new email → receive a unique API key. One key per e‑mail."""
    if req.email.lower() in api_keys:
        raise HTTPException(status_code=409, detail="Email already registered.")
    key = str(uuid.uuid4())
    api_keys[req.email.lower()] = key
    keys_by_api[key] = req.email.lower()
    save_json(KEYS_FILE, api_keys)
    return RegisterResponse(api_key=key)


@app.delete("/apikey", response_model=DeleteKeyResponse, summary="Revoke a key and delete all its data", description="Delete an existing API key and all associated conversation histories.")
async def delete_key(req: DeleteKeyRequest):
    """Delete a key and all histories tied to it."""
    key = req.api_key
    if key not in keys_by_api:
        raise HTTPException(status_code=404, detail="API key not found.")
    email = keys_by_api.pop(key)
    api_keys.pop(email, None)
    thread_histories.pop(key, None)
    # Persist changes
    save_json(KEYS_FILE, api_keys)
    save_json(HISTORIES_FILE, thread_histories)
    return DeleteKeyResponse(detail=f"Key and all data for {email} deleted.")


@app.post("/chat", summary="Send a message (text / image / file) and get a reply", description="Send a user message, optionally with images or files, to an AI or Vision model and receive a response. Supports conversation threading and model selection.")
async def chat(
    chat_request_body: Optional[str] = Form(None, description="JSON string for chat data in multipart/form-data requests."),
    chat_request_form: Optional[str] = Form(None, description="JSON string representation of the ChatRequest data for multipart/form-data requests."),
    files: List[UploadFile] = File(None, description="Optional multipart files (images or supported text files). Max one image per message."),
    api_key: str = Depends(get_api_key),
    background_tasks: BackgroundTasks = None,
):
    if chat_request_body and chat_request_form:
        raise HTTPException(status_code=400, detail="Cannot provide both chat_request_body and chat_request_form.")

    if not chat_request_body and not chat_request_form and not files:
        raise HTTPException(status_code=400, detail="No chat request data or files provided.")

    chat_request = None
    if chat_request_body:
        try:
            chat_request = ChatRequest.model_validate_json(chat_request_body)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid chat_request_body JSON: {e}")
    elif chat_request_form:
        try:
            chat_request = ChatRequest.model_validate_json(chat_request_form)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid chat_request_form JSON: {e}")
    else:
        chat_request = ChatRequest(
            username="anonymous", 
            content="", 
            system_prompt="You are a helpful assistant."
        )


    histories = thread_histories.setdefault(api_key, {})

    if not chat_request.thread_id:
        chat_request.thread_id = str(uuid.uuid4())

    if chat_request.thread_id not in histories:
        histories[chat_request.thread_id] = []

    image_bytes = None
    text_file_contents = []

    for f in files or []:
        if f.content_type.startswith("image/"):
            image_bytes = await f.read()
        else:
            filename = f.filename.lower()
            if any(filename.endswith(ext) for ext in {".txt", ".js", ".py", ".log", ".json"}):
                text = (await f.read()).decode("utf-8", errors="replace")
                text_file_contents.append(f"[File: {f.filename}]\n{text}")

    histories[chat_request.thread_id].append({"role": "user", "content": chat_request.content})
    for file_content in text_file_contents:
        histories[chat_request.thread_id].append({"role": "user", "content": file_content})

    is_vision_model_requested = chat_request.modelname in VISION_MODELS
    is_ai_model_requested = chat_request.modelname in AI_MODELS or not chat_request.modelname

    reply = ""

    if image_bytes:
        if not is_vision_model_requested:
            raise HTTPException(
                status_code=400,
                detail="Image provided, but a vision model was not specified or an incompatible AI model was requested."
            )

        desc = await analyze_image(
            messages=histories[chat_request.thread_id],
            username=chat_request.username,
            user_system_prompt=chat_request.system_prompt,
            modelname=chat_request.modelname,
            image_bytes=image_bytes
        )
        if desc:
            reply = desc
        else:
            raise HTTPException(status_code=502, detail="Vision model could not process the request with the provided image.")

    else:
        if is_vision_model_requested:
            desc = await analyze_image(
                messages=histories[chat_request.thread_id],
                username=chat_request.username,
                user_system_prompt=chat_request.system_prompt,
                modelname=chat_request.modelname,
                image_bytes=None
            )
            if desc:
                reply = desc
            else:
                raise HTTPException(status_code=502, detail="Vision model could not process the text-only request.")
        elif is_ai_model_requested:
            ai_reply = call_ai_model(
                messages=histories[chat_request.thread_id],
                username=chat_request.username,
                user_system_prompt=chat_request.system_prompt,
                modelname=chat_request.modelname
            )
            if ai_reply:
                reply = ai_reply
        else:
            raise HTTPException(status_code=400, detail=f"Invalid modelname: {chat_request.modelname}. Model not found in AI or Vision models.")

    total_tokens = token_count(histories[chat_request.thread_id])
    if total_tokens > MAX_TOKENS:
        raise HTTPException(
            status_code=403,
            detail="Token limit exceeded; please start a new thread."
        )

    if not reply:
        raise HTTPException(status_code=502, detail="Failed to get a response from any AI model.")

    histories[chat_request.thread_id].append({"role": "assistant", "content": reply})

    background_tasks.add_task(save_json, HISTORIES_FILE, thread_histories)

    return {
        "thread_id": chat_request.thread_id,
        "reply": reply,
        "tokens_used": total_tokens + len(reply),
    }


@app.get("/privacy-policy", summary="Privacy Policy", description="Details on data collection, usage, and user rights, compliant with German legal standards.")
async def privacy_policy():
    policy_text = """
    ## Privacy Policy

    **Operator:** This Eclipse Systems FREE AI Project service is operated by students as part of an educational project. It has no concrete commercial operator.

    **Data Collection:**
    - **API Keys:** We collect and store API keys linked to email addresses for authentication and to manage conversation histories.
    - **Conversation Histories:** We store your conversation threads to maintain context for ongoing interactions. This data is associated with your API key.
    - **Usage Data:** We may collect anonymous usage data (e.g., token counts) for internal analysis and to improve service quality.

    **Data Usage:**
    - Your API key and email are used solely for authentication and account management.
    - Conversation histories are used to provide conversational context and improve the AI's responses within your threads.
    - Anonymous usage data helps us optimize the Eclipse Systems FREE AI Project service.

    **Data Storage and Security:**
    - Data is stored on servers located within Germany and is subject to German data protection laws.
    - We implement reasonable technical and organizational measures to protect your data from unauthorized access, loss, or alteration.

    **Your Rights (under GDPR, applicable in Germany):**
    - **Right to Access:** You have the right to request copies of your personal data.
    - **Right to Rectification:** You have the right to request that we correct any information you believe is inaccurate or complete incomplete information.
    - **Right to Erasure:** You have the right to request that we erase your personal data, under certain conditions.
    - **Right to Restrict Processing:** You have the right to request that we restrict the processing of your personal data, under certain conditions.
    - **Right to Object to Processing:** You have the right to object to our processing of your personal data, under certain conditions.
    - **Right to Data Portability:** You have the right to request that we transfer the data that we have collected to another organization, or directly to you, under certain conditions.

    **Contact for Data Deletion or Queries:**
    If you wish to have your data deleted, or have any questions regarding your data or this policy, please contact the operator at: **plurryt@gmail.com**

    **Changes to this Policy:**
    We may update our Privacy Policy from time to time. We will notify you of any changes by posting the new Privacy Policy on this page.

    **Last Updated:** {{DATE}}
    """.replace("{{DATE}}", datetime.date.today().isoformat())
    return JSONResponse(content={"policy": policy_text})


@app.get("/tos", summary="Terms of Service", description="Outlines the terms and conditions for using the API, compliant with German legal standards.")
async def tos():
    tos_text = """
    ## Terms of Service (ToS)

    **Operator:** This Eclipse Systems FREE AI Project service is operated by students as part of an educational project. It has no concrete commercial operator.

    **Acceptance of Terms:**
    By accessing or using the OVM Support API, you agree to be bound by these Terms of Service. If you disagree with any part of the terms, then you may not access the service.

    **Service Description:**
    The OVM Support API provides access to AI and Vision models for troubleshooting server-side technical errors. The service is provided "as is" and "as available" without any warranties, express or implied.

    **User Conduct:**
    You agree not to use the service for any unlawful purpose or in any way that interrupts, damages, or impairs the service.

    **Disclaimer of Warranties:**
    The service is provided without any representations or warranties, express or implied. We do not warrant that the service will be uninterrupted, error-free, or free of viruses or other harmful components.

    **Limitation of Liability:**
    In no event shall the operators be liable for any direct, indirect, incidental, special, consequential, or punitive damages, including without limitation, loss of profits, data, use, goodwill, or other intangible losses, resulting from (i) your access to or use of or inability to access or use the service; (ii) any conduct or content of any third party on the service; (iii) any content obtained from the service; and (iv) unauthorized access, use, or alteration of your transmissions or content, whether based on warranty, contract, tort (including negligence), or any other legal theory, whether or not we have been informed of the possibility of such damage.

    **Governing Law:**
    These Terms shall be governed and construed in accordance with the laws of Germany, without regard to its conflict of law provisions.

    **Contact for Data Deletion or Queries:**
    If you wish to have your data deleted, or have any questions regarding your data or these terms, please contact the operator at: **plurryt@gmail.com**

    **Changes to Terms:**
    We reserve the right, at our sole discretion, to modify or replace these Terms at any time. If a revision is material, we will try to provide at least 30 days' notice prior to any new terms taking effect. What constitutes a material change will be determined at our sole discretion.

    **Last Updated:** {{DATE}}
    """.replace("{{DATE}}", datetime.date.today().isoformat())
    return JSONResponse(content={"terms_of_service": tos_text})


@app.get("/history/{thread_id}", summary="Get the full conversation for a thread", description="Retrieve the complete conversation history for a specific thread ID.")
async def get_history(thread_id: str, api_key: str = Depends(get_api_key)):
    histories = thread_histories.get(api_key, {})
    if thread_id not in histories:
        raise HTTPException(status_code=404, detail="Thread not found.")
    return {"thread_id": thread_id, "history": histories[thread_id]}


@app.delete("/forget/{thread_id}", summary="Purge a thread’s history", description="Delete a specific conversation thread and all its messages.")
async def forget_thread(thread_id: str, api_key: str = Depends(get_api_key)):
    histories = thread_histories.get(api_key, {})
    if thread_id not in histories:
        raise HTTPException(status_code=404, detail="Thread not found.")
    histories.pop(thread_id)
    # Persist
    save_json(HISTORIES_FILE, thread_histories)
    return {"detail": f"Thread {thread_id} history purged."}


# --------------------------------------------------------------------------- #
#  HEALTH CHECK & STARTUP
# --------------------------------------------------------------------------- #

@app.post("/chat/json", summary="Send a text-only message via JSON body", description="Simplified chat endpoint that accepts only JSON body requests for text-only conversations.")
async def chat_json(
    chat_request: ChatRequest = Body(..., description="Chat request data in JSON format"),
    api_key: str = Depends(get_api_key),
    background_tasks: BackgroundTasks = None,
):
    """Simplified chat endpoint for JSON-only requests (no file uploads)."""
    
    histories = thread_histories.setdefault(api_key, {})
    
    if not chat_request.thread_id:
        chat_request.thread_id = str(uuid.uuid4())
    
    if chat_request.thread_id not in histories:
        histories[chat_request.thread_id] = []
    
    histories[chat_request.thread_id].append({"role": "user", "content": chat_request.content})
    
    is_vision_model_requested = chat_request.modelname in VISION_MODELS
    is_ai_model_requested = chat_request.modelname in AI_MODELS or not chat_request.modelname
    
    reply = ""
    
    if is_vision_model_requested:
        desc = await analyze_image(
            messages=histories[chat_request.thread_id],
            username=chat_request.username,
            user_system_prompt=chat_request.system_prompt,
            modelname=chat_request.modelname,
            image_bytes=None
        )
        if desc:
            reply = desc
        else:
            raise HTTPException(status_code=502, detail="Vision model could not process the text-only request.")
    elif is_ai_model_requested:
        ai_reply = call_ai_model(
            messages=histories[chat_request.thread_id],
            username=chat_request.username,
            user_system_prompt=chat_request.system_prompt,
            modelname=chat_request.modelname
        )
        if ai_reply:
            reply = ai_reply
    else:
        raise HTTPException(status_code=400, detail=f"Invalid modelname: {chat_request.modelname}. Model not found in AI or Vision models.")
    
    total_tokens = token_count(histories[chat_request.thread_id])
    if total_tokens > MAX_TOKENS:
        raise HTTPException(
            status_code=403,
            detail="Token limit exceeded; please start a new thread."
        )
    
    if not reply:
        raise HTTPException(status_code=502, detail="Failed to get a response from any AI model.")
    
    histories[chat_request.thread_id].append({"role": "assistant", "content": reply})
    
    background_tasks.add_task(save_json, HISTORIES_FILE, thread_histories)
    
    return {
        "thread_id": chat_request.thread_id,
        "reply": reply,
        "tokens_used": total_tokens + len(reply),
    }


@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "time": datetime.datetime.utcnow().isoformat() + "Z"}


@app.on_event("startup")
async def startup_event():
    pass


@app.on_event("shutdown")
async def shutdown_event():
    save_json(KEYS_FILE, api_keys)
    save_json(HISTORIES_FILE, thread_histories)
