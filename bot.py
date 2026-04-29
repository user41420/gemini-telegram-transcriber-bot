import os
import asyncio
import tempfile
import requests
import logging
import subprocess
from pyrogram import Client, filters, compose
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from google import genai
import motor.motor_asyncio

API_ID = int(os.environ.get("API_ID", "0"))
API_HASH = os.environ.get("API_HASH", "")

_tokens_env = os.environ.get("BOT_TOKENS", "")
BOT_TOKENS = [t.strip() for t in _tokens_env.split(",") if t.strip()]

MONGO_URL = os.environ.get("MONGO_URL", "")
db_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL) if MONGO_URL else None
db = db_client["bot_db"] if db_client is not None else None
users_collection = db["users"] if db is not None else None

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "99"))
MAX_UPLOAD_SIZE = MAX_UPLOAD_MB * 1024 * 1024
MAX_MESSAGE_CHUNK = 4095
DOWNLOADS_DIR = os.environ.get("DOWNLOADS_DIR", "./downloads")
REQUEST_TIMEOUT = 240

API_KEY_REQ_MSG = """**Send me Gemini API key first.**

Please follow these easy steps:
1️⃣ [Click Here to Create an API Key](https://aistudio.google.com/api-keys?project=gen-lang-client-0087827115)
2️⃣ Copy the key you just created.
3️⃣ Paste it here in this chat.

📺 Need a video tutorial? [Watch the guide here](https://t.me/+WnUuqifcQKg5MzA8)

💡 *A little reminder: This bot is the most accurate one available on Telegram and it's 100% free.*
📂 *The bot code is open source:* [View on GitHub](https://github.com/user41420)"""

os.makedirs(DOWNLOADS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

LANGS = [
("🇬🇧 English", "en"), ("🇸🇦 العربية", "ar"), ("🇪🇸 Español", "es"), ("🇫🇷 Français", "fr"),
("🇷🇺 Русский", "ru"), ("🇩🇪 Deutsch", "de"), ("🇮🇳 हिन्दी", "hi"), ("🇮🇷 فارسی", "fa"),
("🇮🇩 Indonesia", "id"), ("🇺🇦 Українська", "uk"), ("🇮🇹 Italiano", "it"),
("🇹🇷 Türkçe", "tr"), ("🇧🇬 Български", "bg"), ("🇷🇸 Srpski", "sr"), ("🇵🇰 اردو", "ur"),
("🇹🇭 ไทย", "th"), ("🇻🇳 Tiếng Việt", "vi"), ("🇯🇵 日本語", "ja"), ("🇰🇷 한국어", "ko"),
("🇨🇳 中文", "zh"),  ("🇸🇪 Svenska", "sv"), ("🇳🇴 Norsk", "no"),
("🇮🇱 עברית", "he"), ("🇩🇰 Dansk", "da"), ("🇪🇹 አማርኛ", "am"), ("🇫🇮 Suomi", "fi"),
("🇧🇩 বাংলা", "bn"), ("🇰🇪 Kiswahili", "sw"), ("🇳🇵 नेपाली", "ne"),
("🇵🇱 Polski", "pl"), ("🇬🇷 Ελληνικά", "el"), ("🇨🇿 Čeština", "cs"), ("🇮🇸 Íslenska", "is"),
("🇱🇹 Lietuvių", "lt"), ("🇱🇻 Latviešu", "lv"), ("🇭🇷 Hrvatski", "hr"),
("🇭🇺 Magyar", "hu"), ("🇷🇴 Română", "ro"), ("🇸🇴 Somali", "so"), ("🇲🇾 Melayu", "ms"),
("🇺🇿 O'zbekcha", "uz"), ("🇵🇭 Tagalog", "tl"), ("🇵🇹 Português", "pt"), ("Auto Detect ⭐️", "auto")
]

user_transcriptions = {}

async def get_db_data(uid):
    if users_collection is None:
        return {}
    data = await users_collection.find_one({"_id": uid})
    return data if data else {}

async def update_db_data(uid, data):
    if users_collection is not None:
        await users_collection.update_one({"_id": uid}, {"$set": data}, upsert=True)

async def get_user_mode(uid):
    data = await get_db_data(uid)
    return data.get("mode", "Split messages")

async def safe_execute(coro):
    try:
        return await coro
    except Exception:
        return None

def convert_to_gemini_audio(input_path):
    output_path = input_path + ".mp3"
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-b:a", "128k",
            output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except Exception:
        return None

def _sync_process_audio_gemini(processed_path, lang_code, api_key):
    client = genai.Client(api_key=api_key)
    uploaded_file = client.files.upload(file=processed_path)
    
    base_prompt = (
        "Convert this audio into a full written transcript with these rules:\n\n"
        "- Do not summarize anything; include everything that is said.\n"
        "- If the audio is long or covers multiple topics, split it into clear, well-organized paragraphs.\n"
        "- Correct grammar and spelling mistakes, but keep the original meaning.\n"
        "- Fix unclear or incorrect phrases naturally without changing intent.\n"
        "- Do not add any extra text, explanations, intro, or outro.\n"
        "- If multiple speakers exist, separate them clearly.\n\n"
        "Output only the cleaned transcript."
    )
    
    if lang_code == "auto":
        prompt = base_prompt + "Transcribe the audio exactly. If multiple languages are detected, capture them all accurately."
    else:
        prompt = base_prompt + f"Transcribe this audio. Ensure the final output text is generated strictly in this language: {lang_code}"

    try:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[prompt, uploaded_file]
            )
        except Exception:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, uploaded_file]
            )
        text_result = response.text
    except Exception as e:
        text_result = f"Error: {str(e)}"
    finally:
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass
            
    return text_result

async def process_audio_gemini(file_path, lang_code, api_key):
    try:
        processed_path = await asyncio.to_thread(convert_to_gemini_audio, file_path)
        if not processed_path:
            return "Error: Audio processing failed."
        
        result = await asyncio.to_thread(_sync_process_audio_gemini, processed_path, lang_code, api_key)
        
        if os.path.exists(processed_path):
            os.remove(processed_path)
            
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def _sync_process_image_gemini(file_path, api_key):
    client = genai.Client(api_key=api_key)
    uploaded_file = client.files.upload(file=file_path)
    
    prompt = (
        "If there is any text visible in this image, extract it exactly as it appears. "
        "If there is no text at all, provide a detailed description of what you see in the image."
    )
    
    try:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, uploaded_file]
            )
        except Exception:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[prompt, uploaded_file]
            )
        text_result = response.text
    except Exception as e:
        text_result = f"Error: {str(e)}"
    finally:
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass
            
    return text_result

async def process_image_gemini(file_path, api_key):
    return await asyncio.to_thread(_sync_process_image_gemini, file_path, api_key)

def ask_gemini_summary(text, instruction, api_key):
    try:
        client = genai.Client(api_key=api_key)
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[f"{instruction}\n\n{text}"]
            )
        except Exception:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[f"{instruction}\n\n{text}"]
            )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

async def ask_gemini_summary_async(text, instruction, api_key):
    return await asyncio.to_thread(ask_gemini_summary, text, instruction, api_key)

def build_action_keyboard(msg_id):
    return InlineKeyboardMarkup([[InlineKeyboardButton(" Summary", callback_data=f"summopt|Summary|{msg_id}")]])

def build_lang_keyboard(origin):
    btns = [[InlineKeyboardButton(lbl, callback_data=f"lang|{code}|{lbl}|{origin}") for lbl, code in LANGS[i:i+3]] for i in range(0, len(LANGS), 3)]
    return InlineKeyboardMarkup(btns)

def bind_handlers(app):
    @app.on_message(filters.command(['start']))
    async def send_welcome(client, message):
        welcome_text = (
            "👋 Salaam!\n"
            "• Send me\n"
            "• voice message\n"
            "• audio file\n"
            "• video\n"
            "• photo\n"
            "• Get Text for free\n\n"
            "Select the language you want the final text to be generated in (optional): Bot Builder @luna3403"
        )
        await message.reply_text(welcome_text, reply_markup=build_lang_keyboard("file"), quote=True)

    @app.on_message(filters.command(['help']))
    async def help_cmd(client, message):
        help_text = (
            "**Here is how to use the bot:**\n\n"
            "1️⃣ **API Key:** Send your Gemini API key first if you haven't already.\n"
            "2️⃣ **Language:** Use /lang to choose the language you want the text generated in.\n"
            "3️⃣ **Output:** Use /mode to choose if you want a split message or a text file.\n"
            "4️⃣ **Process:** Send any voice, audio, video, or photo.\n"
            "5️⃣ **Result:** Wait a moment to get your highly accurate extraction!"
        )
        await message.reply_text(help_text, quote=True)

    @app.on_message(filters.command(['mode']))
    async def choose_mode(client, message):
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("💬 Split", callback_data="mode|Split messages")], [InlineKeyboardButton("📄 File", callback_data="mode|Text File")]])
        await message.reply_text("Choose output mode:", reply_markup=kb, quote=True)

    @app.on_callback_query(filters.regex(r'^mode\|'))
    async def mode_cb(client, call):
        mode_val = call.data.split("|")[1]
        await update_db_data(call.from_user.id, {"mode": mode_val})
        await safe_execute(call.message.delete())
        await call.answer(f"Mode: {mode_val}")

    @app.on_message(filters.command(['lang']))
    async def lang_command(client, message):
        await message.reply_text("Select the language you want the final text to be generated in:", reply_markup=build_lang_keyboard("file"), quote=True)

    @app.on_callback_query(filters.regex(r'^lang\|'))
    async def lang_cb(client, call):
        parts = call.data.split("|")
        await update_db_data(call.message.chat.id, {"lang": parts[1]})
        await safe_execute(call.message.delete())
        await call.answer(f"Language: {parts[2]}")

    @app.on_callback_query(filters.regex(r'^summopt\|'))
    async def summopt_cb(client, call):
        uid = call.from_user.id
        user_data = await get_db_data(uid)
        api_key = user_data.get("api_key")
        
        if not api_key:
            await call.answer("Please send me Gemini API Key first", show_alert=True)
            await call.message.reply_text(API_KEY_REQ_MSG)
            return

        parts = call.data.split("|")
        style, msg_id = parts[1], parts[2]
        await safe_execute(call.message.edit_reply_markup(reply_markup=None))
        
        p = {
            "Summary": "Task: Summarize the text concisely.\nLanguage: Use the same language as the input.\nConstraint: Return ONLY the summary. No conversational filler, no explanations, no 'Here is the summary'.\n\nText to summarize: "
        }
        
        await process_text_action(client, call, msg_id, style, p.get(style), api_key)

    async def process_text_action(client, call, msg_id, style, prompt_instr, api_key):
        chat_id = call.message.chat.id
        data = user_transcriptions.get(chat_id, {}).get(int(msg_id))
        if not data:
            await call.answer("Error: Transcription data not found. Please resend the file.", show_alert=True)
            return
        await call.answer(f"Generating {style}...")
        try:
            res = await ask_gemini_summary_async(data["text"], prompt_instr, api_key)
            await send_long_text(client, chat_id, res, data["origin"], call.from_user.id)
        except Exception:
            await client.send_message(chat_id, "Error Please try again later")

    @app.on_message(filters.text & filters.regex(r'^AIza'))
    async def save_key(client, message):
        await update_db_data(message.from_user.id, {"api_key": message.text.strip()})
        await message.reply_text("Gemini API Key saved successfully! You can send me now audio video or photo 🥰", quote=True)

    @app.on_message(filters.photo)
    async def handle_photo(client, message):
        uid = message.from_user.id
        user_data = await get_db_data(uid)
        api_key = user_data.get("api_key")
        
        if not api_key:
            await message.reply_text(API_KEY_REQ_MSG, quote=True)
            return

        status = await message.reply_text("Processing photo...", quote=True)
        tmp = os.path.join(DOWNLOADS_DIR, f"ocr_{message.from_user.id}.jpg")
        try:
            await client.download_media(message, file_name=tmp)
            txt = await process_image_gemini(tmp, api_key)
            if txt and not txt.startswith("Error:"):
                await status.delete()
                sent = await send_long_text(client, message.chat.id, txt, message.id, uid)
                user_transcriptions.setdefault(message.chat.id, {})[sent.id] = {"text": txt, "origin": message.id}
                if len(txt) > 2000:
                    await sent.edit_reply_markup(reply_markup=build_action_keyboard(sent.id))
            else: 
                await status.edit_text(txt if txt else "Could not process image.")
        finally:
            if os.path.exists(tmp): os.remove(tmp)

    @app.on_message(filters.voice | filters.audio | filters.video | (filters.document & ~filters.photo))
    async def handle_media(client, message):
        uid = message.from_user.id
        user_data = await get_db_data(uid)
        api_key = user_data.get("api_key")
        
        if not api_key:
            await message.reply_text(API_KEY_REQ_MSG, quote=True)
            return
        
        chat_data = await get_db_data(message.chat.id)
        lang = chat_data.get("lang", "auto")
        
        status = await message.reply_text("Transcribing...", quote=True)
        tmp = tempfile.NamedTemporaryFile(delete=False, dir=DOWNLOADS_DIR).name
        try:
            await client.download_media(message, file_name=tmp)
            txt = await process_audio_gemini(tmp, lang, api_key)
            if txt and not txt.startswith("Error:"):
                await status.delete()
                sent = await send_long_text(client, message.chat.id, txt, message.id, uid)
                user_transcriptions.setdefault(message.chat.id, {})[sent.id] = {"text": txt, "origin": message.id}
                if len(txt) > 2000:
                    await sent.edit_reply_markup(reply_markup=build_action_keyboard(sent.id))
            else:
                await status.edit_text(txt if txt else "Could not transcribe")
        finally:
            if os.path.exists(tmp): os.remove(tmp)

    async def send_long_text(client, chat_id, text, reply_id, uid):
        mode = await get_user_mode(uid)
        if len(text) > MAX_MESSAGE_CHUNK:
            if mode == "Split messages":
                s = None
                for i in range(0, len(text), MAX_MESSAGE_CHUNK):
                    s = await client.send_message(chat_id, text[i:i+MAX_MESSAGE_CHUNK], reply_to_message_id=reply_id)
                return s
            else:
                path = os.path.join(DOWNLOADS_DIR, f"result_{reply_id}.txt")
                with open(path, "w", encoding="utf-8") as f: f.write(text)
                s = await client.send_document(chat_id, path, reply_to_message_id=reply_id)
                os.remove(path)
                return s
        return await client.send_message(chat_id, text, reply_to_message_id=reply_id)

apps = []
for index, token_str in enumerate(BOT_TOKENS):
    clean_token = token_str.strip()
    if clean_token:
        app_instance = Client(
            f"whisper_session_{index}",
            api_id=API_ID,
            api_hash=API_HASH,
            bot_token=clean_token
        )
        bind_handlers(app_instance)
        apps.append(app_instance)

if __name__ == "__main__":
    if apps:
        asyncio.run(compose(apps))
