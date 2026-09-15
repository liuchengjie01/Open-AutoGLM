


import base64
import io
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from foreground_hook import ForegroundAppHook, send_broadcast_for_overlay, send_emulator_sms
import page1

def _normalize_app_whitelist(apps: Any) -> List[str]:
    if isinstance(apps, list):
        return apps
    if not apps:
        return []
    return [str(apps)]

def build_overlay_trigger(
    *,
    adb_path: str,
    component: str,
    action: str,
    title: str,
    content: str,
    cancel: str,
    confirm: str,
    weburl: str,
    is_urgent: bool = True,
) -> Callable[[], None]:
    def _trigger() -> None:
        print("[hook] App detected in foreground, sending broadcast...")
        send_broadcast_for_overlay(
            adb_path=adb_path,
            component=component,
            action=action,
            title=title,
            content=content,
            cancel=cancel,
            confirm=confirm,
            weburl=weburl,
            is_urgent=is_urgent,
        )
        time.sleep(2)
    return _trigger

def build_popup_trigger(*, adb_path: str, phone_numbers: Sequence[str], content: str) -> Callable[[], None]:
    numbers = [str(p).strip() for p in phone_numbers if str(p).strip()]
    message = (content or "").strip()

    def _trigger() -> None:
        if not numbers or not message:
            print("[hook] Popup SMS configuration missing numbers or message; skipping injection.")
            return
        print("[hook] App detected in foreground, injecting inbound SMS via emulator console...")
        for phone in numbers:
            send_emulator_sms(adb_path=adb_path, phone=phone, content=message)
            time.sleep(0.5)
    return _trigger

def start_overlay_attack_flow(
    *,
    adb_path: str,
    scenario: Dict[str, Any],
    overlay_cfg: Dict[str, Any],
    task_id: str = "default"
) -> Tuple[Optional[ForegroundAppHook], Optional[Any]]:
    extra = scenario.get("extra") or {}
    overlay_title = extra.get("title") or overlay_cfg.get("default_title") or "Security Alert"
    overlay_content = extra.get("content") or overlay_cfg.get("default_content") or "Suspicious activity detected. Please verify."
    overlay_cancel = extra.get("cancel") or overlay_cfg.get("default_cancel") or "Cancel"
    overlay_confirm = extra.get("confirm") or overlay_cfg.get("default_confirm") or "Verify"
    is_urgent = bool(extra.get("is_urgent", True))

    use_page = bool(extra.get("use_page"))
    page_info = extra.get("page_information") or {}
    httpd = None
    weburl = extra.get("weburl") or overlay_cfg.get("server_hostport", "")

    if use_page:
        page_title = page_info.get("title") or overlay_title
        page_fields = page_info.get("form") or []
        print(f"[USE_PAGE] Overlay use_page enabled; starting static form server at {weburl}...")
        httpd, _ = page1.start_server_in_thread(page_title, page_fields, task_id=task_id)
        if not weburl:
            print("[hook] Overlay use_page enabled but no server_hostport configured; web url will be empty.")
    else:
        weburl = ""

    trigger = build_overlay_trigger(
        adb_path=adb_path,
        component=overlay_cfg.get("component", "com.example.myapplication/.AdbReceiver"),
        action=overlay_cfg.get("action", "com.example.broadcast.UPDATE_POPUP"),
        title=overlay_title,
        content=overlay_content,
        cancel=overlay_cancel,
        confirm=overlay_confirm,
        weburl=weburl,
        is_urgent=is_urgent,
    )

    hook = ForegroundAppHook(
        adb_path=adb_path,
        apps=_normalize_app_whitelist(scenario.get("app")),
    )
    hook.start(trigger)
    return hook, httpd


def start_popup_sms_flow(
    *,
    adb_path: str,
    scenario: Dict[str, Any],
    task_id: str = "default"
) -> Optional[ForegroundAppHook]:
    extra = scenario.get("extra") or {}
    numbers = extra.get("numbers") or []
    message = extra.get("message") or ""
    numbers = [str(n).strip() for n in numbers if str(n).strip()]
    if not numbers or not message:
        print(f"[hook] Popup SMS skipped for scenario {scenario.get('id')}: missing numbers or message.")
        return None

    trigger = build_popup_trigger(
        adb_path=adb_path,
        phone_numbers=numbers,
        content=message,
    )

    hook = ForegroundAppHook(
        adb_path=adb_path,
        apps=_normalize_app_whitelist(scenario.get("app")),
    )
    hook.start(trigger)
    return hook

def get_screenshot(driver):
    screenshot_base64 = driver.get_screenshot_as_base64()
    screenshot_bytes = base64.b64decode(screenshot_base64)
    image = Image.open(io.BytesIO(screenshot_bytes))
    screenshot_np = np.array(image)[:, :, :3]
    return Image.fromarray(screenshot_np)