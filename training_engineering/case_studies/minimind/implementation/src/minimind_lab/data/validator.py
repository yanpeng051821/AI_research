


ALLOWED_ROLES = {"system", "user", "assistant", "tool"}

def validate_pretrain_record(record: object)-> list[str]:

    if not isinstance(record, dict):
        return ["record_not_object"]

    if "text" not in record:
        return ["missing_text"]

    text = record["text"]

    if not isinstance(text, str):
        return ["text_not_string"]
    
    if text.strip() == "":
        return ["empty_text"]

    return []


def validate_sft_record(record: object)-> list[str]:
    issues = []
    if not isinstance(record, dict):
        return ["record_not_object"]

    if "conversations" not in record:
        return ["missing_conversations"]
    
    conversation = record["conversations"]

    if not isinstance(conversation, list):
        return ["conversations_not_list"]

    if len(conversation) == 0:
        return ["empty_conversations"]

    has_assistant = False
    for message in conversation:
        if not isinstance(message, dict):
            issues.append("message_not_object")
            continue

        # 检查role
        role = message.get("role")
        if "role" not in message:
            issues.append("missing_role")
        elif not isinstance(role, str):
            issues.append("role_not_string")
        elif role not in ALLOWED_ROLES:
            issues.append("invalid_role")
        elif role == "assistant":
            has_assistant = True

        # 检查content
        content = message.get("content")
        if "content" not in message:
            issues.append("missing_content")
        elif not isinstance(content, str):
            issues.append("content_not_string")

    if not has_assistant:
            issues.append("missing_assistant")

    return list(dict.fromkeys(issues))