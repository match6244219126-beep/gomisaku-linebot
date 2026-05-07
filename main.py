"""LINE Bot webhook server for garbage sorting (ごみ分別).

Flow:
  User sends photo → LINE Webhook → Claude identifies item
  → Search gomisaku.jp dictionary → Reply with sorting info
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from gomisaku import GomisakuDB

load_dotenv()

LINE_CHANNEL_ACCESS_TOKEN: str = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
LINE_CHANNEL_SECRET: str = os.environ["LINE_CHANNEL_SECRET"]
ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

db = GomisakuDB()
claude = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

CONTACT_INFO = "役場 建設課 環境衛生係\n📞 0167-52-2179"


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading Gomisaku database …")
    await db.load()
    logger.info("Loaded %d items, %d types", len(db.items), len(db.types))
    yield


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# LINE helpers
# ---------------------------------------------------------------------------

def verify_signature(body: bytes, signature: str) -> bool:
    digest = hmac.new(LINE_CHANNEL_SECRET.encode(), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode()
    return hmac.compare_digest(expected, signature)


async def get_line_image(message_id: str) -> tuple[bytes, str]:
    """Download image content from LINE Content API."""
    url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
    ctype = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    return resp.content, ctype


async def reply_message(reply_token: str, text: str) -> None:
    """Send a text reply via LINE Messaging API."""
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": text}],
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()


# ---------------------------------------------------------------------------
# Claude image recognition
# ---------------------------------------------------------------------------

_VALID_MEDIA_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

_IDENTIFY_PROMPT = (
    "この画像に写っているゴミを日本のゴミ分別辞典で検索するための品名・検索ワードを、"
    "カンマ区切りで最大5つ答えてください。\n"
    "一般的な呼び名だけでなく、辞典に載っていそうな具体的な品目名も含めてください。\n"
    "例）カップ麺の画像 → カップ麺容器, インスタントラーメン容器, 発泡スチロール容器, カップ麺, 容器\n"
    "ゴミが特定できない、またはゴミではない場合のみ「不明」と答えてください。"
)


async def identify_item(image_bytes: bytes, content_type: str) -> list[str]:
    """Ask Claude for multiple search-term candidates for the garbage item."""
    if content_type not in _VALID_MEDIA_TYPES:
        content_type = "image/jpeg"

    image_b64 = base64.standard_b64encode(image_bytes).decode()

    response = await claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": content_type,
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": _IDENTIFY_PROMPT},
            ],
        }],
    )
    raw = response.content[0].text.strip()
    if raw == "不明" or not raw:
        return []
    # カンマ・読点・全角カンマで分割
    import re as _re
    candidates = [c.strip() for c in _re.split(r"[,、，]", raw) if c.strip()]
    return candidates[:5]


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------

def format_result(queried_name: str, result: dict) -> str:
    """Format a found result. Shows a suggestion note when the matched name
    differs from what the user (Claude) originally identified."""
    type_info = result.get("type_info", {})
    type_name = type_info.get("name", "不明")
    howto = type_info.get("howto", "").strip()
    comment = result.get("comment", "").strip()
    display_name = result.get("name", queried_name)

    lines = []
    # 候補経由でヒットした場合（例：カップ麺→インスタントラーメン容器）に注記を追加
    if display_name != queried_name:
        lines.append(f"「{queried_name}」→「{display_name}」として分別情報をお伝えします\n")

    lines += [
        f"【{display_name}】",
        f"▶ 分別: {type_name}",
    ]
    if howto:
        lines.append(f"\n📋 出し方:\n{howto}")
    if comment:
        lines.append(f"\n⚠️ 注意: {comment}")

    return "\n".join(lines)


def format_not_found(queried_name: str, suggestions: list[dict]) -> str:
    """Format a not-found response, optionally listing partial matches."""
    if not suggestions:
        return (
            f"「{queried_name}」の分別情報が見つかりませんでした。\n\n"
            f"品名を変えて再度お試しいただくか、\n"
            f"直接お問い合わせください。\n{CONTACT_INFO}"
        )

    lines = [f"「{queried_name}」は辞典に見つかりませんでした。\n近い品目はこちらです：\n"]
    for s in suggestions:
        type_name = s.get("type_info", {}).get("name", "不明")
        lines.append(f"・{s['name']}（{type_name}）")
    lines.append(f"\n詳しくはお問い合わせください。\n{CONTACT_INFO}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Event handling
# ---------------------------------------------------------------------------

async def handle_event(event: dict) -> None:
    if event.get("type") != "message":
        return

    reply_token: str = event.get("replyToken", "")
    message: dict = event.get("message", {})
    msg_type: str = message.get("type", "")

    try:
        if msg_type == "image":
            message_id = message["id"]
            image_bytes, content_type = await get_line_image(message_id)
            candidates = await identify_item(image_bytes, content_type)

            if not candidates:
                reply = "うまく判別できませんでした。\nゴミが写るように別の角度から撮り直してください📷"
            else:
                primary_name = candidates[0]  # Claudeが最初に挙げた名前
                result = None
                matched_candidate = None

                # 候補を順番に検索し、最初にヒットしたものを使う
                for candidate in candidates:
                    result = db.search(candidate)
                    if result:
                        matched_candidate = candidate
                        break

                if result:
                    reply = format_result(primary_name, result)
                else:
                    # どの候補もヒットしなければ近い品目を提案
                    suggestions = db.get_suggestions(primary_name)
                    reply = format_not_found(primary_name, suggestions)

            await reply_message(reply_token, reply)

        else:
            # Text, sticker, or any other message type
            await reply_message(reply_token, "ゴミの写真を送ってください📷")

    except Exception:
        logger.exception("Error handling event: %s", event)
        try:
            await reply_message(
                reply_token,
                "エラーが発生しました。しばらくしてから再試行してください。",
            )
        except Exception:
            logger.exception("Failed to send error reply")


# ---------------------------------------------------------------------------
# Webhook endpoint
# ---------------------------------------------------------------------------

@app.post("/webhook")
async def webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")

    if not verify_signature(body, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    payload = json.loads(body)
    for event in payload.get("events", []):
        # Fire-and-forget so LINE gets its 200 OK within 1 s
        asyncio.create_task(handle_event(event))

    return {"status": "ok"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "items": len(db.items),
        "types": len(db.types),
    }
