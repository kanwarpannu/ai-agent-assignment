from typing import Any
from openai import OpenAI
import os
from lib.messages import (
    AnyMessage,
    AIMessage,
    BaseMessage,
    UserMessage,
)
from lib.tooling import Tool


class LLM:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        tools: list[Tool] | None = None,
        api_key: str | None = None
    ):
        self.model = model
        self.temperature = temperature

        resolved_api_key = os.getenv("OPENAI_API_KEY")
        resolved_base_url = os.getenv("OPENAI_BASE_URL")

        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url
        )

        self.tools: dict[str, Tool] = {
            tool.name: tool for tool in (tools or [])
        }

    def register_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def _build_payload(self, messages: list[BaseMessage]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [m.dict() for m in messages],
        }

        if self.tools:
            payload["tools"] = [tool.dict() for tool in self.tools.values()]
            payload["tool_choice"] = "auto"

        return payload

    def _convert_input(self, input: Any) -> list[BaseMessage]:
        if isinstance(input, str):
            return [UserMessage(content=input)]
        elif isinstance(input, BaseMessage):
            return [input]
        elif isinstance(input, list) and all(isinstance(m, BaseMessage) for m in input):
            return input
        else:
            raise ValueError(f"Invalid input type {type(input)}.")

    def invoke(self, input: str | BaseMessage | list[BaseMessage]) -> AIMessage:
        messages = self._convert_input(input)
        payload = self._build_payload(messages)
        response = self.client.chat.completions.create(**payload)
        choice = response.choices[0]
        message = choice.message

        return AIMessage(
            content=message.content,
            tool_calls=message.tool_calls
        )