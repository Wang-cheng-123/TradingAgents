import questionary
from typing import List, Optional, Tuple, Dict

from cli.models import AnalystType

ANALYST_ORDER = [
    ("市场分析师", AnalystType.MARKET),
    ("社交媒体分析师", AnalystType.SOCIAL),
    ("新闻分析师", AnalystType.NEWS),
    ("基本面分析师", AnalystType.FUNDAMENTALS),
]

ANALYST_DISPLAY_MAP = {
    AnalystType.MARKET.value: "市场分析师",
    AnalystType.SOCIAL.value: "社交媒体分析师",
    AnalystType.NEWS.value: "新闻分析师",
    AnalystType.FUNDAMENTALS.value: "基本面分析师",
}

def get_ticker() -> str:
    """Prompt the user to enter a ticker symbol."""
    ticker = questionary.text(
        "请输入要分析的股票代码：",
        validate=lambda x: len(x.strip()) > 0 or "请输入有效的股票代码。",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not ticker:
        console.print("\n[red]未提供股票代码，程序退出。[/red]")
        exit(1)

    return ticker.strip().upper()


def get_analysis_date() -> str:
    """Prompt the user to enter a date in YYYY-MM-DD format."""
    import re
    from datetime import datetime

    def validate_date(date_str: str) -> bool:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return False
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    date = questionary.text(
        "请输入分析日期 (YYYY-MM-DD)：",
        validate=lambda x: validate_date(x.strip())
        or "请按照 YYYY-MM-DD 格式输入有效日期。",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not date:
        console.print("\n[red]未提供日期，程序退出。[/red]")
        exit(1)

    return date.strip()


def select_analysts() -> List[AnalystType]:
    """Select analysts using an interactive checkbox."""
    choices = questionary.checkbox(
        "请选择【分析师团队】：",
        choices=[
            questionary.Choice(display, value=value) for display, value in ANALYST_ORDER
        ],
        instruction="\n- 按空格键选择或取消分析师\n- 按 a 键全选或取消全选\n- 按 Enter 键确认",
        validate=lambda x: len(x) > 0 or "至少需要选择一名分析师。",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        console.print("\n[red]未选择分析师，程序退出。[/red]")
        exit(1)

    return choices


def select_research_depth() -> int:
    """Select research depth using an interactive selection."""

    # Define research depth options with their corresponding values
    DEPTH_OPTIONS = [
        ("浅层 - 快速研究，少量讨论轮次", 1),
        ("中等 - 平衡速度与深度，适中讨论轮次", 3),
        ("深度 - 全面研究，充分讨论与推演", 5),
    ]

    choice = questionary.select(
        "请选择【研究深度】：",
        choices=[
            questionary.Choice(display, value=value) for display, value in DEPTH_OPTIONS
        ],
        instruction="\n- 使用方向键移动\n- 按 Enter 键确认",
        style=questionary.Style(
            [
                ("selected", "fg:yellow noinherit"),
                ("highlighted", "fg:yellow noinherit"),
                ("pointer", "fg:yellow noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]未选择研究深度，程序退出。[/red]")
        exit(1)

    return choice


def select_shallow_thinking_agent(provider) -> str:
    """Select shallow thinking llm engine using an interactive selection."""

    # Define shallow thinking llm engine options with their corresponding model names
    SHALLOW_AGENT_OPTIONS = {
        "openai": [
            ("GPT-4o-mini - 快速高效，适合即时任务", "gpt-4o-mini"),
            ("GPT-4.1-nano - 超轻量模型，适合基础操作", "gpt-4.1-nano"),
            ("GPT-4.1-mini - 体积小性能佳", "gpt-4.1-mini"),
            ("GPT-4o - 标准型模型，能力均衡", "gpt-4o"),
        ],
        "anthropic": [
            ("Claude Haiku 3.5 - 推理快速，能力稳定", "claude-3-5-haiku-latest"),
            ("Claude Sonnet 3.5 - 高性能标准模型", "claude-3-5-sonnet-latest"),
            ("Claude Sonnet 3.7 - 出色的混合推理与代理能力", "claude-3-7-sonnet-latest"),
            ("Claude Sonnet 4 - 高性能与优秀推理力", "claude-sonnet-4-0"),
        ],
        "google": [
            ("Gemini 2.0 Flash-Lite - 成本友好、延迟低", "gemini-2.0-flash-lite"),
            ("Gemini 2.0 Flash - 新一代特性、速度与思考力", "gemini-2.0-flash"),
            ("Gemini 2.5 Flash - 自适应思考，成本优", "gemini-2.5-flash-preview-05-20"),
        ],
        "openrouter": [
            ("Meta：Llama 4 Scout", "meta-llama/llama-4-scout:free"),
            ("Meta：Llama 3.3 8B Instruct - 轻量且高速的 Llama 3.3 变体", "meta-llama/llama-3.3-8b-instruct:free"),
            ("google/gemini-2.0-flash-exp:free - 更快的首 Token 响应", "google/gemini-2.0-flash-exp:free"),
        ],
        "ollama": [
            ("qwen-30b_q6k_xl 本地", "qwen-30b_q6k_xl"),
            ("lqwen-30b_tink_q6k_xl 本地", "qwen-30b_tink_q6k_xl"),
        ],
        "deepseek": [
            ("DeepSeek Chat - 通用对话与快速分析", "deepseek-chat"),
            ("DeepSeek Reasoner - 精于复杂推理", "deepseek-reasoner"),
        ],
    }

    choice = questionary.select(
        "请选择【快速思考模型】：",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in SHALLOW_AGENT_OPTIONS[provider.lower()]
        ],
        instruction="\n- 使用方向键移动\n- 按 Enter 键确认",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print(
            "\n[red]未选择快速思考模型，程序退出。[/red]"
        )
        exit(1)

    return choice


def select_deep_thinking_agent(provider) -> str:
    """Select deep thinking llm engine using an interactive selection."""

    # Define deep thinking llm engine options with their corresponding model names
    DEEP_AGENT_OPTIONS = {
        "openai": [
            ("GPT-4.1-nano - 超轻量模型，适合基础操作", "gpt-4.1-nano"),
            ("GPT-4.1-mini - 体积小性能佳", "gpt-4.1-mini"),
            ("GPT-4o - 标准型模型，能力均衡", "gpt-4o"),
            ("o4-mini - 精简版加强推理模型", "o4-mini"),
            ("o3-mini - 高级推理模型（轻量）", "o3-mini"),
            ("o3 - 全功能高级推理模型", "o3"),
            ("o1 - 顶级推理与问题求解模型", "o1"),
        ],
        "anthropic": [
            ("Claude Haiku 3.5 - 推理快速，能力稳定", "claude-3-5-haiku-latest"),
            ("Claude Sonnet 3.5 - 高性能标准模型", "claude-3-5-sonnet-latest"),
            ("Claude Sonnet 3.7 - 出色的混合推理与代理能力", "claude-3-7-sonnet-latest"),
            ("Claude Sonnet 4 - 高性能与优秀推理力", "claude-sonnet-4-0"),
            ("Claude Opus 4 - Anthropic 最强模型", "	claude-opus-4-0"),
        ],
        "google": [
            ("Gemini 2.0 Flash-Lite - 成本友好、延迟低", "gemini-2.0-flash-lite"),
            ("Gemini 2.0 Flash - 新一代特性、速度与思考力", "gemini-2.0-flash"),
            ("Gemini 2.5 Flash - 自适应思考，成本优", "gemini-2.5-flash-preview-05-20"),
            ("Gemini 2.5 Pro", "gemini-2.5-pro-preview-06-05"),
        ],
        "openrouter": [
            ("DeepSeek V3 - 685B 参数专家混合模型", "deepseek/deepseek-chat-v3-0324:free"),
            ("DeepSeek - 最新一代旗舰对话模型", "deepseek/deepseek-chat-v3-0324:free"),
        ],
        "ollama": [
            ("qwen-30b_tink_q6k_xl 本地", "qwen-30b_tink_q6k_xl"),
        ],
        "deepseek": [
            ("DeepSeek Chat - 通用对话与快速分析", "deepseek-chat"),
            ("DeepSeek Reasoner - 精于复杂推理", "deepseek-reasoner"),
        ],
    }
    
    choice = questionary.select(
        "请选择【深度思考模型】：",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in DEEP_AGENT_OPTIONS[provider.lower()]
        ],
        instruction="\n- 使用方向键移动\n- 按 Enter 键确认",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]未选择深度思考模型，程序退出。[/red]")
        exit(1)

    return choice

def select_llm_provider() -> tuple[str, str]:
    """Select the OpenAI api url using interactive selection."""
    # Define OpenAI api options with their corresponding endpoints
    BASE_URLS = [
        ("OpenAI", "https://api.openai.com/v1"),
        ("Anthropic", "https://api.anthropic.com/"),
        ("Google", "https://generativelanguage.googleapis.com/v1"),
        ("Openrouter", "https://openrouter.ai/api/v1"),
        ("Ollama", "http://192.168.196.150:11434/v1"),
        ("DeepSeek", "https://api.deepseek.com/v1"),
    ]
    
    choice = questionary.select(
        "请选择模型服务提供商：",
        choices=[
            questionary.Choice(display, value=(display, value))
            for display, value in BASE_URLS
        ],
        instruction="\n- 使用方向键移动\n- 按 Enter 键确认",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()
    
    if choice is None:
        console.print("\n[red]未选择模型服务，程序退出。[/red]")
        exit(1)
    
    display_name, url = choice
    print(f"已选择：{display_name}\t接口地址：{url}")
    
    return display_name, url
