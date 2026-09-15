"""Main PhoneAgent class for orchestrating phone automation."""
from datetime import datetime
import os
import re
import json
import time
import traceback
import subprocess
from dataclasses import dataclass
from typing import Any, Callable

from phone_agent.actions import ActionHandler
from phone_agent.actions.handler import do, finish, parse_action
from phone_agent.adb import get_current_app, get_screenshot
from phone_agent.config import get_messages, get_system_prompt
from phone_agent.model import ModelClient, ModelConfig
from phone_agent.model.client import MessageBuilder


@dataclass
class AgentConfig:
    """Configuration for the PhoneAgent."""

    max_steps: int = 100
    device_id: str | None = None
    lang: str = "cn"
    system_prompt: str | None = None
    verbose: bool = True

    def __post_init__(self):
        if self.system_prompt is None:
            self.system_prompt = get_system_prompt(self.lang)


@dataclass
class StepResult:
    """Result of a single agent step."""

    success: bool
    finished: bool
    action: dict[str, Any] | None
    original_action: str | None = None
    raw_content: str | None = None
    thinking: str | None = None
    message: str | None = None
    log_probs: list[dict[str, Any]] | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    package_name: str | None = None
    activity_name: str | None = None


def get_package_activity(device_id):
    cmd = f"adb -s {device_id} shell dumpsys window | findstr mCurrentFocus"
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=5,
            encoding='utf-8'  # Windows 可能需要 GBK
        )
        
        if result.returncode == 0 and result.stdout.strip():
            output = result.stdout.strip()
            match = re.search(r'(\S+)/(\S+?)(?=\s|}|$)', output)
            if match:
                package = match.group(1)
                activity = match.group(2)
                return package, activity
            
            # 备用匹配：可能格式是 Window{... com.android.chrome/.MainActivity}
            match = re.search(r'Window\{[^}]*\s+([^\s/]+)/([^\s}]+)', output)
            if match:
                return match.group(1), match.group(2)
    except subprocess.TimeoutExpired:
        print("ADB 命令超时")
    except Exception as e:
        print(f"ADB 执行失败: {e}")
    return "", ""

def compute_avg_log_prob(s, logprobs_content, action_str=None) -> float:
    """
    最简单实用的方法：通过字符串查找 + 字符位置映射
    """
    import re
    
    # 1. 找 action
    action_pattern = r'do\(action="[^"]*",\s*element=\[[^\]]*\]\)'
    # 定义所有支持的 action 模式
    if action_str is not None:
        action_text = action_str
        action_start = s.find(action_text)
        if action_start == -1:
            return 0.0
        action_end = action_start + len(action_text)
    else:
        action_patterns = [
            # do(action="Tap", element=[x,y])
            r'do\(action="Tap",\s*element=\[[\d\s,]+\]\)',
            
            # do(action="Type", text="...")
            r'do\(action="Type",\s*text="[^"]*"\)',
            
            # do(action="Swipe", start=[x1,y1], end=[x2,y2])
            r'do\(action="Swipe",\s*start=\[[\d\s,]+\],\s*end=\[[\d\s,]+\]\)',
            
            # do(action="Long Press", element=[x,y])
            r'do\(action="Long Press",\s*element=\[[\d\s,]+\]\)',
            
            # do(action="Launch", app="...")
            r'do\(action="Launch",\s*app="[^"]*"\)',
            
            # do(action="Back")
            r'do\(action="Back"\)',
            
            # finish(message="...")
            r'finish\(message="[^"]*"\)',
            
            # 通用匹配（兜底）：do(...) 或 finish(...)
            r'(?:do|finish)\([^)]*\)',
        ]

        match = None
        for pattern in action_patterns:
            match = re.search(pattern, s)
            if match:
                break
        if not match:
            return 0.0
        
        action_text = match.group(0)
        action_start = match.start()
        action_end = match.end()
    
    # 2. 获取 tokens
    tokens = [item["token"] for item in logprobs_content]
    # assert ''.join(tokens) == s, "Tokens do not match the original string"
    logprobs = [item["log_prob"] for item in logprobs_content]
    
    # 3. 使用字符累积找到 token 范围
    special_tokens = {"<think>", "</think>", "<answer>", "</answer>"}
    current_pos = action_start
    start_idx = None
    end_idx = None
    
    for i, token in enumerate(tokens):
        # 跳过特殊 token（如果有）
        if token in special_tokens:
            continue
        
        # 找到当前 token 在 s 中的位置
        # 从 current_pos 开始找
        pos = s.find(token, current_pos)
        if pos == -1:
            continue
            # 如果找不到（可能是空格被重新组合），尝试去掉空格
            # token_stripped = token.strip()
            # if token_stripped and token_stripped != token:
            #     pos = s.find(token_stripped, current_pos)
            #     if pos != -1:
            #         # 找到的是简化版，需要修正
            #         # 简单处理：使用 current_pos
            #         pos = current_pos
            #     else:
                    # pos = current_pos
        
        token_end = pos + len(token)
        current_pos = token_end
        
        # 检查 token 是否与 action 区域相交
        if pos < action_end and token_end > action_start:
            if start_idx is None:
                start_idx = i
            end_idx = i + 1
    
    if start_idx is None:
        return 0.0
    log_probs_result = logprobs[start_idx:end_idx]
    hit_actions = tokens[start_idx:end_idx]
    print(f"Action tokens: {hit_actions}")
    """Compute the average log probability from a list of token log probabilities."""
    if len(log_probs_result) == 0:
        return 0.0
    total_log_prob = sum(log_probs_result)
    return total_log_prob / len(log_probs_result)

def compute_avg_log_prob_v2(s, logprobs_content, action_str=None) -> float:
    """
    计算 action 部分的平均 log probability
    直接基于 token 序列定位，不依赖 s.find()
    """
    import re
    
    # 1. 获取 tokens
    tokens = [item["token"] for item in logprobs_content]
    logprobs = [item["log_prob"] for item in logprobs_content]
    
    # 2. 找 action 起始 token
    # 方法：找到 <answer> 后面的第一个 'do' 或 'finish'
    start_idx = None
    
    # 先找 <answer> 的位置
    answer_idx = None
    for i, token in enumerate(tokens):
        if token == '<answer>':
            answer_idx = i
            break
    
    if answer_idx is not None:
        # 从 <answer> 后面开始找 'do' 或 'finish'
        for i in range(answer_idx + 1, len(tokens)):
            if tokens[i] in ('do', 'finish'):
                start_idx = i
                break
    
    # 如果没找到，全局找
    if start_idx is None:
        for i, token in enumerate(tokens):
            if token in ('do', 'finish'):
                start_idx = i
                break
    
    if start_idx is None:
        return 0.0
    
    # 3. 找结束位置：到 </answer> 或字符串结束
    end_idx = None
    for i in range(start_idx, len(tokens)):
        if tokens[i] == '</answer>':
            end_idx = i
            break
    
    if end_idx is None:
        end_idx = len(tokens)
    
    # 4. 提取结果
    log_probs_result = logprobs[start_idx:end_idx]
    hit_actions = tokens[start_idx:end_idx]
    print(f"Action tokens: {hit_actions}")
    
    if len(log_probs_result) == 0:
        return 0.0
    total_log_prob = sum(log_probs_result)
    return total_log_prob / len(log_probs_result)


class PhoneAgent:
    """
    AI-powered agent for automating Android phone interactions.

    The agent uses a vision-language model to understand screen content
    and decide on actions to complete user tasks.

    Args:
        model_config: Configuration for the AI model.
        agent_config: Configuration for the agent behavior.
        confirmation_callback: Optional callback for sensitive action confirmation.
        takeover_callback: Optional callback for takeover requests.

    Example:
        >>> from phone_agent import PhoneAgent
        >>> from phone_agent.model import ModelConfig
        >>>
        >>> model_config = ModelConfig(base_url="http://localhost:8000/v1")
        >>> agent = PhoneAgent(model_config)
        >>> agent.run("Open WeChat and send a message to John")
    """

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        agent_config: AgentConfig | None = None,
        confirmation_callback: Callable[[str], bool] | None = None,
        takeover_callback: Callable[[str], None] | None = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.agent_config = agent_config or AgentConfig()

        self.model_client = ModelClient(self.model_config)
        self.action_handler = ActionHandler(
            device_id=self.agent_config.device_id,
            confirmation_callback=confirmation_callback,
            takeover_callback=takeover_callback,
        )

        self._context: list[dict[str, Any]] = []
        self._step_count = 0
        self.save_dir = ""

    def run(self, task: str, idx: int = 0) -> str:
        """
        Run the agent to complete a task.

        Args:
            task: Natural language description of the task.

        Returns:
            Final message from the agent.
        """
        self._context = []
        self._step_count = -1
        log_data = []
        # First step with user prompt
        start_tick = time.time()
        package, activity = get_package_activity(self.agent_config.device_id)
        result = self._execute_step(task, is_first=True, idx=idx)
        step_info = {
                        "step": self._step_count,
                        "time_consumption": time.time() - start_tick,
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "log_prob_avg": compute_avg_log_prob_v2(result.raw_content, result.log_probs, result.original_action.strip()),
                        "package": package,
                        "activity": activity,
                        "action": result.original_action.strip()
                    }
        task_id = self.save_dir.split(os.sep)[-2]  # 获取当前任务的唯一标识符
        if os.path.exists(os.path.join("signal_files", f'{task_id}.txt')):
            os.remove(os.path.join("signal_files", f'{task_id}.txt'))
            step_info['is_attack'] = True
        log_data.append(step_info)
        if result.finished:
            return result.message or "Task completed"

        # Continue until finished or max steps reached
        while self._step_count < self.agent_config.max_steps:
            start_tick = time.time()
            package, activity = get_package_activity(self.agent_config.device_id)
            result = self._execute_step(is_first=False, idx=idx)

            # compute log probabilities for the last step
            step_info = {
                "step": self._step_count,
                "time_consumption": time.time() - start_tick,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "log_prob_avg": compute_avg_log_prob_v2(result.raw_content, result.log_probs, result.original_action.strip()),
                "package": package,
                "activity": activity,
                "action": result.original_action.strip()
            }
            if os.path.exists(os.path.join("signal_files", f'{task_id}.txt')):
                os.remove(os.path.join("signal_files", f'{task_id}.txt'))
                step_info['is_attack'] = True
            log_data.append(step_info)
            
            if result.finished:
                print("Finished")
                # print(log_data)
                with open(os.path.join(self.save_dir, f"{idx}.json"), "w", encoding="utf-8") as f:
                    json.dump(log_data, f, ensure_ascii=False, indent=2)
                return result.message or "Task completed"
        print(log_data)
        with open(os.path.join(self.save_dir, f"{idx}.json"), "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        return "Max steps reached"

    def step(self, task: str | None = None) -> StepResult:
        """
        Execute a single step of the agent.

        Useful for manual control or debugging.

        Args:
            task: Task description (only needed for first step).

        Returns:
            StepResult with step details.
        """
        is_first = len(self._context) == 0

        if is_first and not task:
            raise ValueError("Task is required for the first step")

        return self._execute_step(task, is_first)

    def reset(self) -> None:
        """Reset the agent state for a new task."""
        self._context = []
        self._step_count = -1

    def _execute_step(
        self, user_prompt: str | None = None, is_first: bool = False, idx: int = 0
    ) -> StepResult:
        """Execute a single step of the agent loop."""
        self._step_count += 1

        # Capture current screen state
        target_img = os.path.join(self.save_dir, f'step_{idx}_{self._step_count:02d}.jpg')
        screenshot = get_screenshot(target_img, self.agent_config.device_id)
        current_app = get_current_app(self.agent_config.device_id)

        # Build messages
        if is_first:
            self._context.append(
                MessageBuilder.create_system_message(self.agent_config.system_prompt)
            )

            screen_info = MessageBuilder.build_screen_info(current_app)
            text_content = f"{user_prompt}\n\n{screen_info}"

            self._context.append(
                MessageBuilder.create_user_message(
                    text=text_content, image_base64=screenshot.base64_data
                )
            )
        else:
            screen_info = MessageBuilder.build_screen_info(current_app)
            text_content = f"** Screen Info **\n\n{screen_info}"

            self._context.append(
                MessageBuilder.create_user_message(
                    text=text_content, image_base64=screenshot.base64_data
                )
            )

        # Get model response
        try:
            response = self.model_client.request(self._context)
        except Exception as e:
            if self.agent_config.verbose:
                traceback.print_exc()
            return StepResult(
                success=False,
                finished=True,
                action=None,
                thinking="",
                raw_content="",
                message=f"Model error: {e}",
                log_probs=response.log_probs,
            )

        # Parse action from response
        try:
            action = parse_action(response.action)
        except ValueError:
            if self.agent_config.verbose:
                traceback.print_exc()
            action = finish(message=response.action)

        if self.agent_config.verbose:
            # Print thinking process
            msgs = get_messages(self.agent_config.lang)
            print("\n" + "=" * 24 + f"{idx}-{self._step_count}" + "=" * 24)
            print(f"💭 {msgs['thinking']}:")
            print("-" * 50)
            print(response.thinking)
            print("-" * 50)
            print(f"🎯 {msgs['action']}:")
            print(json.dumps(action, ensure_ascii=False, indent=2))
            print("=" * 50 + "\n")

        # Remove image from context to save space
        self._context[-1] = MessageBuilder.remove_images_from_message(self._context[-1])

        # Execute action
        try:
            result = self.action_handler.execute(
                action, screenshot.width, screenshot.height
            )
            time.sleep(3.0)
        except Exception as e:
            if self.agent_config.verbose:
                traceback.print_exc()
            result = self.action_handler.execute(
                finish(message=str(e)), screenshot.width, screenshot.height
            )

        # Add assistant response to context
        self._context.append(
            MessageBuilder.create_assistant_message(
                f"<think>{response.thinking}</think><answer>{response.action}</answer>"
            )
        )

        # Check if finished
        finished = action.get("_metadata") == "finish" or result.should_finish

        if finished and self.agent_config.verbose:
            msgs = get_messages(self.agent_config.lang)
            print("\n" + "🎉 " + "=" * 48)
            print(
                f"✅ {msgs['task_completed']}: {result.message or action.get('message', msgs['done'])}"
            )
            print("=" * 50 + "\n")

        return StepResult(
            success=result.success,
            finished=finished,
            action=action,
            original_action=response.action,
            thinking=response.thinking,
            raw_content=response.raw_content,
            message=result.message or action.get("message"),
            log_probs=response.log_probs,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens
        )

    @property
    def context(self) -> list[dict[str, Any]]:
        """Get the current conversation context."""
        return self._context.copy()

    @property
    def step_count(self) -> int:
        """Get the current step count."""
        return self._step_count
