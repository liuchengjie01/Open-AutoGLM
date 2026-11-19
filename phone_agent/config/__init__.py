"""Configuration module for Phone Agent."""

from phone_agent.config.apps import APP_PACKAGES
from phone_agent.config.i18n import get_message, get_messages
from phone_agent.config.prompts_en import SYSTEM_PROMPT as SYSTEM_PROMPT_EN
from phone_agent.config.prompts_zh import SYSTEM_PROMPT as SYSTEM_PROMPT_ZH
from phone_agent.config.prompts_en_mai import MAI_MOBILE_SYS_PROMPT as SYSTEM_PROMPT_EN_MAI


def get_system_prompt(lang: str = "cn", model_name: str = "autoglm-phone-9b") -> str:
    """
    Get system prompt by language.

    Args:
        lang: Language code, 'cn' for Chinese, 'en' for English.

    Returns:
        System prompt string.
    """
    if lang == "en" and model_name in ["autoglm-phone-9b", "gpt-6.1-sol", "gpt-6-astra"]:
        return SYSTEM_PROMPT_EN
    if lang == "en" and model_name in ["mai-ui-8b", "gpt-6.1-sol", "gpt-6-astra"]:
        return SYSTEM_PROMPT_EN_MAI
    return SYSTEM_PROMPT_ZH


# Default to Chinese for backward compatibility
SYSTEM_PROMPT = SYSTEM_PROMPT_ZH

__all__ = [
    "APP_PACKAGES",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_ZH",
    "SYSTEM_PROMPT_EN",
    "get_system_prompt",
    "get_messages",
    "get_message",
]
