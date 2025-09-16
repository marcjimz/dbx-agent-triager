def _chat_content(resp) -> str:
    # Works for both dataclass ChatCompletionResponse and dict-like responses
    try:
        # dataclass style
        return resp.choices[0].message.content
    except Exception:
        # dict style or other shapes
        if isinstance(resp, dict):
            # OpenAI-like: {'choices':[{'message':{'content': '...'}}]}
            ch = resp.get("choices", [{}])
            if ch and "message" in ch[0]:
                return ch[0]["message"].get("content", "")
            # raw 'content'
            return resp.get("content", "")
        return str(resp)