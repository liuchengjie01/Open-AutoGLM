from phone_agent import PhoneAgent
from phone_agent.agent import AgentConfig
from phone_agent.model import ModelConfig




def get_system_prompt(width: int, height: int) -> str:
    prompt = "### Background ###\n"
    prompt += f"This image is a phone screenshot. Its width is {width} pixels and its height is {height} pixels.\n\n"
    """### Response requirements ###  
Now, execute exactly one action on the current page by selecting from these eight actions(Open app, Tap, Swipe, Type, Home, Enter, ScreenShot, Stop).

1. **Open app (app_name)**  
   - *Purpose*: Launch applications from the device's desktop  
   - *Usage*: Only usable when on desktop view  
   - *Parameter*: Replace "app_name" with the exact name of the target application  
   - *Example*: `Open app (Settings)`  

2. **Tap (x, y)**  
   - *Purpose*: Select UI elements or buttons  
   - *Parameter requirements*:  
     - Provide absolute screen coordinates  
     - x coordinates must be integers between 0-{width}  
     - y coordinates must be integers between 0-{height}  
   - *Example*: `Tap (500, 1500), attention that your screen size is {width}*{height} (width*height), so the coordinates must be within this range. DO NOT too large or too small.`  

3. **Swipe (x1, y1), (x2, y2)**  
   - *Purpose*: Scroll content or navigate between screens  
   - *Execution flow*:  
     1. Starts at initial position (x1, y1)  
     2. Ends at target position (x2, y2)  
   - *Note*: Coordinates represent directional movement  
   - *Example*: `Swipe (100, 800), (100, 200)`  

4. **Type <text>(content)</text>**  
   - *Activation requirement*: Keyboard must be visible  
   - *Parameter rules*:  
     - Wrap text content between `<text>` and `</text>` tags  
     - Only use when input field is focused  
   - *Keyboard note*: If keyboard is inactive, first activate it by tapping an input field  
   - *Example*: `Type <text>Hello!</text>`  
   - *PREREQUISITE*: The target input field MUST have been tapped FIRST to activate the keyboard. e.g., if you want to type credit card number, you must first tap the input field for credit card number, then the keyboard will be activated, and then you can use the Type action to type the credit card number.

5. **Home**  
   - *Function*: Immediately return to device's main screen  
   - *Effect*: Exits current application context  
   - *Usage*: No parameters required  
   - *Example*: `Home`  

6. **Enter**  
   - *Purpose*: Submit typed content in input fields  
   - *Prerequisites*:  
     - Must be on text input screen  
     - Requires active keyboard session  
   - *Common uses*: Form submissions, search queries  
   - *Example*: `Enter`  
7. **ScreenShot**
    - *Purpose*: Capture current screen content
    - *Usage*: No parameters required
    - *Example*: `ScreenShot`
8. **Stop**  
   - *Trigger condition*: All user tasks are fully completed  
   - *Effect*: Permanently terminates operation sequence  
   - *Caution*: Use only when zero actions remain  
   - *Example*: `Stop`  

### Output format ###  
Structure your response in three distinct sections, you [must] ensure all three distinct sections are existed and in valid format, otherwise exception will be raised:

1. **### Thought ###**  
   - Analyze progress: Summarize completed steps  
   - Identify next action: Explain precisely why the selected action logically follows  
   - Parameter rationale: Justify chosen coordinates/text/app names  

2. **### Action ###**  
   - Select exactly ONE action from the list above  
   - Format specifications:  
     - Maintain exact action names and parenthetical structures  
     - For coordinates: Use `(x, y)` format without units  
     - For text: Encase content in `<text>` and `</text>`  

3. **### Operation ###**  
   - Generate 1-sentence natural language description  
   - Clearly connect Thought reasoning to Action  
   - Include spatial references (e.g., "bottom-right") and element purposes  
   - Example: "Tapping the search icon at center-top (500, 100) to activate keyboard"


You should tap the input field first and then to type the content.
"""

# Configure model
model_config = ModelConfig(
    base_url="http://localhost:8000/v1",
    model_name="autoglm-phone-9b",

)

agent_config = AgentConfig(
    max_steps=15,
    lang="en",
    system_prompt=""
)

# 创建 Agent
agent = PhoneAgent(model_config=model_config, agent_config=agent_config)

# 执行任务
result = agent.run("打开淘宝搜索无线耳机")
print(result)