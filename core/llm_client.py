"""
Simplified LLM Client using LiteLLM.

Provides:
- Direct API configuration (url, key, model)
- Automatic retries with exponential backoff
- Token usage tracking
- JSON output parsing
"""
import os
import json
import re
import logging
from typing import Optional, Any, TypeVar, Type, List, Dict, Union
from dataclasses import dataclass
from contextlib import contextmanager
from collections import defaultdict
import threading

import litellm
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

# Suppress LiteLLM verbose output
os.environ['LITELLM_LOG'] = 'WARNING'
litellm.set_verbose = False

logger = logging.getLogger(__name__)

# Type variable for Pydantic models
T = TypeVar("T", bound=BaseModel)


class TokenTracker:
    """
    Token usage tracker with thread-safe step-based statistics.
    """

    def __init__(self):
        self._local = threading.local()
        self._lock = threading.Lock()
        self._stats: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"input": 0, "output": 0})
        )
        self._call_count: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    @property
    def current_step(self) -> str:
        """Get current step name"""
        return getattr(self._local, "current_step", "default")

    @current_step.setter
    def current_step(self, value: str):
        self._local.current_step = value

    @contextmanager
    def step(self, name: str):
        """Switch to specified step, restore on exit"""
        old = self.current_step
        self.current_step = name
        try:
            yield
        finally:
            self.current_step = old

    def record(self, model: str, usage: Dict[str, int]):
        """Record token usage for a call"""
        step = self.current_step
        with self._lock:
            self._stats[step][model]["input"] += usage.get("prompt_tokens", 0)
            self._stats[step][model]["output"] += usage.get("completion_tokens", 0)
            self._call_count[step][model] += 1

    def summary(self) -> dict:
        """Get statistics summary"""
        with self._lock:
            result = {}
            for step, models in self._stats.items():
                result[step] = {
                    model: {
                        "input_tokens": data["input"],
                        "output_tokens": data["output"],
                        "total_tokens": data["input"] + data["output"],
                        "call_count": self._call_count[step][model],
                    }
                    for model, data in models.items()
                }
            return result

    def reset(self):
        """Clear all statistics"""
        with self._lock:
            self._stats.clear()
            self._call_count.clear()

    def print_summary(self):
        """Print formatted statistics"""
        print("\n" + "=" * 70)
        print("Token Usage Summary")
        print("=" * 70)
        for step, models in self.summary().items():
            print(f"\n[{step}]")
            for model, stats in models.items():
                print(f"  {model}:")
                print(f"    Input:  {stats['input_tokens']:,}")
                print(f"    Output: {stats['output_tokens']:,}")
                print(f"    Total:  {stats['total_tokens']:,}")
                print(f"    Calls:  {stats['call_count']}")
        print("=" * 70 + "\n")


@dataclass
class LLMResponse:
    """Structured LLM response with metadata."""

    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    api_base: Optional[str] = None
    raw_response: Optional[Any] = None

    def parse_json(self) -> Any:
        """Extract and parse JSON from response."""
        return extract_json(self.content)

    def parse_as(self, model_class: Type[T]) -> T:
        """Parse response as a Pydantic model."""
        data = self.parse_json()
        return model_class.model_validate(data)


def extract_json(text: str) -> Any:
    """
    Extract and parse JSON from text that may contain markdown code blocks.
    """
    # Try to find JSON in code blocks first
    json_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```"
    match = re.search(json_pattern, text)

    if match:
        json_str = match.group(1).strip()
    else:
        # Try to find JSON array or object directly
        array_match = re.search(r'(\[[\s\S]*\])', text)
        object_match = re.search(r'(\{[\s\S]*\})', text)

        if array_match and object_match:
            if array_match.start() < object_match.start():
                json_str = array_match.group(1)
            else:
                json_str = object_match.group(1)
        elif array_match:
            json_str = array_match.group(1)
        elif object_match:
            json_str = object_match.group(1)
        else:
            json_str = text.strip()

    # Clean up common LLM output issues
    json_str = _clean_json_string(json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}, trying aggressive cleanup")
        json_str = _aggressive_json_cleanup(json_str)
        return json.loads(json_str)


def _clean_json_string(json_str: str) -> str:
    """Clean up common JSON formatting issues."""
    json_str = re.sub(r',\s*]', ']', json_str)
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = json_str.lstrip('\ufeff\u200b')
    return json_str


def _aggressive_json_cleanup(json_str: str) -> str:
    """More aggressive JSON cleanup for problematic output."""
    json_str = re.sub(r'//[^\n]*\n', '\n', json_str)
    json_str = re.sub(r'/\*[\s\S]*?\*/', '', json_str)
    json_str = re.sub(r'(?<=[{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r' "\1":', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = ''.join(c for c in json_str if c.isprintable() or c in '\n\r\t')
    return json_str


@dataclass
class LLMConfig:
    """LLM configuration."""
    api_base: Union[str, List[str]]
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 40960
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0


class URLDispatcher:
    """Thread-safe round-robin dispatcher for multiple API base URLs."""

    def __init__(self, api_bases: Union[str, List[str]]):
        if isinstance(api_bases, str):
            api_bases = [api_bases]
        self.api_bases = [url.strip() for url in api_bases if url and url.strip()]
        if not self.api_bases:
            raise ValueError("At least one api_base/api_url is required")
        self._counter = 0
        self._lock = threading.Lock()

    def get_next(self) -> str:
        with self._lock:
            api_base = self.api_bases[self._counter % len(self.api_bases)]
            self._counter += 1
            return api_base

    def __len__(self) -> int:
        return len(self.api_bases)


class LLMClient:
    """
    Simplified LLM client with direct configuration.

    Usage:
        client = LLMClient(
            api_base="http://localhost:3000/v1",
            api_key="sk-xxx",
            model="deepseek-v3.2"
        )
        response = client.chat("Hello!")
        print(response.content)
    """

    def __init__(
        self,
        api_base: Union[str, List[str], None] = None,
        api_key: str = "",
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        api_urls: Union[str, List[str], None] = None,
    ):
        if api_urls is not None:
            if api_base is not None:
                raise ValueError("Use either api_base or api_urls, not both")
            api_base = api_urls
        if api_base is None:
            raise ValueError("api_base or api_urls is required")

        self.config = LLMConfig(
            api_base=api_base,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self.dispatcher = URLDispatcher(api_base)
        self.token_tracker = TokenTracker()
        self._stop_events: Dict[int, threading.Event] = {}

    def _check_stop(self, thread_id: int) -> None:
        """Check if current operation should be stopped."""
        event = self._stop_events.get(thread_id)
        if event and event.is_set():
            raise KeyboardInterrupt("Operation stopped by user")

    @contextmanager
    def stoppable(self):
        """Context manager for stoppable operations."""
        thread_id = threading.get_ident()
        self._stop_events[thread_id] = threading.Event()
        try:
            yield self._stop_events[thread_id]
        finally:
            self._stop_events.pop(thread_id, None)

    def stop(self, thread_id: Optional[int] = None):
        """Signal to stop operations."""
        if thread_id:
            event = self._stop_events.get(thread_id)
            if event:
                event.set()
        else:
            for event in self._stop_events.values():
                event.set()

    def _create_retry_decorator(self):
        """Create retry decorator with current config."""
        return retry(
            retry=retry_if_exception_type((Exception,)),
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(
                multiplier=self.config.retry_delay,
                min=self.config.retry_delay,
                max=self.config.retry_delay * 8,
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Internal LLM call using LiteLLM."""
        thread_id = threading.get_ident()
        self._check_stop(thread_id)

        # Build model name with provider prefix for LiteLLM
        model = self.config.model
        if not model.startswith("openai/"):
            # Auto-add openai/ prefix for OpenAI-compatible APIs
            model = f"openai/{model}"

        api_base = self.dispatcher.get_next()
        call_kwargs = {
            "model": model,
            "messages": messages,
            "api_base": api_base,
            "api_key": self.config.api_key,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
        }

        # Override with any kwargs
        call_kwargs.update(kwargs)

        logger.debug(f"LiteLLM call: model={model}, api_base={call_kwargs.get('api_base')}")

        response = litellm.completion(**call_kwargs)

        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason

        # Convert usage to dict
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            self.token_tracker.record(self.config.model, usage)

        return LLMResponse(
            content=content,
            model=self.config.model,
            usage=usage,
            finish_reason=finish_reason,
            api_base=call_kwargs.get("api_base"),
            raw_response=response,
        )

    def chat(
        self,
        query: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send a chat request to the LLM.

        Args:
            query: Simple user query (alternative to messages)
            messages: Full conversation history
            temperature: Override temperature
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            LLMResponse with content and metadata
        """
        # Build messages
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if query:
                messages.append({"role": "user", "content": query})

        # Override temperature if provided
        if temperature is not None:
            kwargs["temperature"] = temperature

        # Create retry-wrapped call
        retry_decorator = self._create_retry_decorator()
        retrying_call = retry_decorator(self._call_llm)

        return retrying_call(messages=messages, **kwargs)

    def generate_structured(
        self,
        query: str,
        output_model: Type[T],
        max_retries: int = 3,
        **kwargs
    ) -> T:
        """
        Generate structured output validated against a Pydantic model.
        """
        schema = output_model.model_json_schema()
        enhanced_query = f"""{query}

Please respond with valid JSON matching this schema:
```json
{json.dumps(schema, indent=2)}
```
Output only the JSON, no explanations."""

        for attempt in range(max_retries):
            try:
                response = self.chat(query=enhanced_query, **kwargs)
                return response.parse_as(output_model)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Validation failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise
                enhanced_query = f"""{enhanced_query}

Previous attempt failed with error: {str(e)}
Please fix the JSON format and try again."""

        raise RuntimeError("Failed to generate valid structured output")
