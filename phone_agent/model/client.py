"""Model client for AI inference using OpenAI-compatible API."""

import ast
import json
from dataclasses import dataclass, field
import re
from typing import Any, Dict

from json_repair import repair_json
from openai import OpenAI

# Constants
SCALE_FACTOR = 999
WIDTH = 1080
HEIGHT = 2400

@dataclass
class ModelConfig:
    """Configuration for the AI model."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model_name: str = "autoglm-phone-9b"
    max_tokens: int = 2048
    temperature: float = 1.0
    do_sample: bool = True
    top_p: float = 0.90
    frequency_penalty: float = 0.2
    logprobs: bool = True
    local: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Response from the AI model."""

    thinking: str
    action: str
    log_probs: list[dict[str, Any]]
    prompt_tokens: int | None
    completion_tokens: int | None
    raw_content: str


class ModelClient:
    """
    Client for interacting with OpenAI-compatible vision-language models.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    def request(self, messages: list[dict[str, Any]]) -> ModelResponse:
        """
        Send a request to the model.

        Args:
            messages: List of message dictionaries in OpenAI format.

        Returns:
            ModelResponse containing thinking and action.

        Raises:
            ValueError: If the response cannot be parsed.
                        logprobs=self.config.logprobs,
        """
        if self.config.local:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                logprobs=self.config.logprobs,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                extra_body=self.config.extra_body,
                stream=False,
            )
        else:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                extra_body=self.config.extra_body,
                stream=False,
            )

        raw_content = response.choices[0].message.content
        log_probs = response.choices[0].logprobs
        prompt_tokens = response.usage.prompt_tokens if hasattr(response.usage, "prompt_tokens") else 0
        completion_tokens = response.usage.completion_tokens if hasattr(response.usage, "completion_tokens") else 0
        extracted_token_probs = []
        if hasattr(log_probs, "content"):
            for item in log_probs.content:
                extracted_token_probs.append({
                    "token": item.token,
                    "log_prob": item.logprob
                })
        else:
            print("Logprobs content not found in the response.")
        # Parse thinking and action from response
        thinking, action = self._parse_response(raw_content)
        # start_idx = 0
        # for idx, token_info in enumerate(extracted_token_probs):
        #     if token_info["token"] in action.strip():
        #         start_idx = idx
        #         break

        return ModelResponse(thinking=thinking, 
                             action=action, 
                             log_probs=extracted_token_probs, 
                             raw_content=raw_content,
                             prompt_tokens=prompt_tokens,
                             completion_tokens=completion_tokens
                             )

    def _parse_response(self, content: str):
        
        if self.config.model_name in ["autoglm-phone-9b", "gpt-6.1-sol", "gpt-6-astra"]:
            return self._parse_autoglm_response(content)
        if self.config.model_name in ["mai-ui-8b", "gpt-6.1-sol", "gpt-6-astra"]:
            return self._parse_mai_ui_response(content)
        
    def _parse_autoglm_response(self, content: str):
        """
        Parse the model response into thinking and action parts.

        Parsing rules:
        1. If content contains 'finish(message=', everything before is thinking,
            everything from 'finish(message=' onwards is action.
        2. If rule 1 doesn't apply but content contains 'do(action=',
            everything before is thinking, everything from 'do(action=' onwards is action.
        3. Fallback: If content contains '<answer>', use legacy parsing with XML tags.
        4. Otherwise, return empty thinking and full content as action.

        Args:
            content: Raw response content.

        Returns:
            Tuple of (thinking, action).
        """
        if "<answer>" in content:
            parts = content.split("<answer>", 1)
            thinking = parts[0].replace("<think>", "").replace("</think>", "").strip()
            action = parts[1].replace("</answer>", "").strip()
            return thinking, action
        # Rule 1: Check for finish(message=
        if "finish(message=" in content:
            parts = content.split("finish(message=", 1)
            thinking = parts[0].strip()
            action = "finish(message=" + parts[1]
            return thinking, action

        # Rule 2: Check for do(action=
        if "do(action=" in content:
            parts = content.split("do(action=", 1)
            thinking = parts[0].strip()
            action = "do(action=" + parts[1]
            return thinking, action
        return "", content


    def parse_tagged_text(self, text: str) -> Dict[str, Any]:
        """
        Parse text containing XML-style tags to extract thinking and tool_call content.

        Args:
            text: Text containing <thinking> and <tool_call> tags.

        Returns:
            Dictionary with keys:
                - "thinking": Content inside <thinking> tags (str or None)
                - "tool_call": Parsed JSON content inside <tool_call> tags (dict or None)

        Raises:
            ValueError: If tool_call content is not valid JSON.
        """
        # Handle thinking model output format (uses </think> instead of </thinking>)
        if "</think>" in text and "</thinking>" not in text:
            text = text.replace("</think>", "</thinking>")
            text = "<thinking>" + text
        if "<answer>" in text and "</answer>" in text:
            text = text.replace("<answer>", "<tool_call>")
            text = text.replace("</answer>", "</tool_call>")
        # Define regex pattern with non-greedy matching
        pattern = r"<thinking>(.*?)</thinking>.*?<tool_call>(.*?)</tool_call>"

        result: Dict[str, Any] = {
            "thinking": None,
            "tool_call": None,
        }

        # Use re.DOTALL to match newlines
        match = re.search(pattern, text, re.DOTALL)
        if match:
            result = {
                "thinking": match.group(1).strip().strip('"'),
                "tool_call": match.group(2).strip().strip('"'),
            }
        else:
            m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
            tool_call_str = m.group(1) if m else None
            if tool_call_str:
                result["tool_call"] = tool_call_str.strip().strip('"')
                result["thinking"] = "No thinking content found."

        # Parse tool_call as JSON
        if result["tool_call"]:
            try:
                result["tool_call"] = repair_json(result["tool_call"], return_objects=False)
                result["tool_call"] = json.loads(result["tool_call"])
            except Exception as e:
                result["tool_call"] = ast.literal_eval(result["tool_call"])
                assert isinstance(result["tool_call"], dict)
                pass

        return result

    
    def _parse_mai_ui_response(self, text: str):
        """
            Parse model output text into structured action format.

            Args:
                text: Raw model output containing thinking and tool_call tags.

            Returns:
                Dictionary with keys:
                    - "thinking": The model's reasoning process
                    - "action_json": Parsed action with normalized coordinates

            Note:
                Coordinates are normalized to [0, 1] range by dividing by SCALE_FACTOR.
        """
        text = text.strip()

        results = self.parse_tagged_text(text)
        thinking = results["thinking"]
        tool_call = results["tool_call"]
        if not tool_call:
            raise ValueError("Invalid tool_call format: missing 'name' or 'arguments'")
        if "arguments" in tool_call:
            action = tool_call["arguments"]
        elif "action" in tool_call:
            action = tool_call

        # Normalize coordinates from SCALE_FACTOR range to [0, 1]
        if "coordinate" in action:
            coordinates = action["coordinate"]
            if len(coordinates) == 2:
                point_x, point_y = coordinates
            elif len(coordinates) == 4:
                x1, y1, x2, y2 = coordinates
                point_x = (x1 + x2) / 2
                point_y = (y1 + y2) / 2
            else:
                raise ValueError(
                    f"Invalid coordinate format: expected 2 or 4 values, got {len(coordinates)}"
                )
            point_x = point_x / SCALE_FACTOR * WIDTH
            point_y = point_y / SCALE_FACTOR * HEIGHT
            action["coordinate"] = [point_x, point_y]
        
        if "start_coordinate" in action:
            coordinates = action["start_coordinate"]
            if len(coordinates) == 2:
                point_x, point_y = coordinates
            elif len(coordinates) == 4:
                x1, y1, x2, y2 = coordinates
                point_x = (x1 + x2) / 2
                point_y = (y1 + y2) / 2
            else:
                raise ValueError(
                    f"Invalid coordinate format: expected 2 or 4 values, got {len(coordinates)}"
                )
            point_x = point_x / SCALE_FACTOR * WIDTH
            point_y = point_y / SCALE_FACTOR * HEIGHT
            action["start_coordinate"] = [point_x, point_y]
        
        if "end_coordinate" in action:
            coordinates = action["end_coordinate"]
            if len(coordinates) == 2:
                point_x, point_y = coordinates
            elif len(coordinates) == 4:
                x1, y1, x2, y2 = coordinates
                point_x = (x1 + x2) / 2
                point_y = (y1 + y2) / 2
            else:
                raise ValueError(
                    f"Invalid coordinate format: expected 2 or 4 values, got {len(coordinates)}"
                )
            point_x = point_x / SCALE_FACTOR * WIDTH
            point_y = point_y / SCALE_FACTOR * HEIGHT
            action["end_coordinate"] = [point_x, point_y]

        return thinking, action
    
class MessageBuilder:
    """Helper class for building conversation messages."""

    @staticmethod
    def create_system_message(content: str) -> dict[str, Any]:
        """Create a system message."""
        return {"role": "system", "content": content}

    @staticmethod
    def create_user_message(
        text: str, image_base64: str | None = None
    ) -> dict[str, Any]:
        """
        Create a user message with optional image.

        Args:
            text: Text content.
            image_base64: Optional base64-encoded image.

        Returns:
            Message dictionary.
        """
        content = []

        if image_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                }
            )

        content.append({"type": "text", "text": text})

        return {"role": "user", "content": content}

    @staticmethod
    def create_assistant_message(content: str) -> dict[str, Any]:
        """Create an assistant message."""
        return {"role": "assistant", "content": content}

    @staticmethod
    def remove_images_from_message(message: dict[str, Any]) -> dict[str, Any]:
        """
        Remove image content from a message to save context space.

        Args:
            message: Message dictionary.

        Returns:
            Message with images removed.
        """
        if isinstance(message.get("content"), list):
            message["content"] = [
                item for item in message["content"] if item.get("type") == "text"
            ]
        return message

    @staticmethod
    def build_screen_info(current_app: str, **extra_info) -> str:
        """
        Build screen info string for the model.

        Args:
            current_app: Current app name.
            **extra_info: Additional info to include.

        Returns:
            JSON string with screen info.
        """
        info = {"current_app": current_app, **extra_info}
        return json.dumps(info, ensure_ascii=False)
