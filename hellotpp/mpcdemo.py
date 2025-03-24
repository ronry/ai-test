import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from typing import Any, Tuple

import requests
from vipserver.vip_client import get_one_validate_host
from whale import TextGeneration, ChatFunctionCall

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

response = requests.get('http://jmenv.tbsite.net:8080/env')
if response.status_code != 200:
    raise Exception(
        "can not get unit info, curl http://jmenv.tbsite.net:8080/env failed")
unit = response.text.strip()


def get_tpp_sse_endpoint(app_id: int):
    global unit
    if unit == "daily":
        return f"https://tppwork.taobao.com/mcppre/mcp/{app_id}/sse"
    elif unit == "pre":
        host = get_one_validate_host("pre.mcp.tpp.vipserver")
        return f"http://{host.ip}:{host.port}/mcp/{app_id}/sse"
    else:
        host = get_one_validate_host("mcp.tpp.vipserver")
        return f"http://{host.ip}:{host.port}/mcp/{app_id}/sse"


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, endpoint: str) -> None:
        self.name: str = name
        self.endpoint: str = endpoint
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        try:
            sse_transport = await self.exit_stack.enter_async_context(
                sse_client(self.endpoint)
            )
            read, write = sse_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append(
                        Tool(tool.name, tool.description, tool.inputSchema))

        return tools

    async def execute_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any],
            retries: int = 2,
            delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)

                return result

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(
                    f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
            self, name: str, description: str, input_schema: dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> dict:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema
        }


class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, model: str, api_key: str) -> None:
        self.api_key: str = api_key
        self.model: str = model
        if unit == "daily" or unit == "pre":
            TextGeneration.set_api_key(api_key,
                                       base_url="https://pre-whale-wave.alibaba-inc.com")
        else:
            TextGeneration.set_api_key(api_key)


    def get_response(self, messages: list[dict],
                     functions: list) -> Tuple[ChatFunctionCall, str]:
        """Get a response from the LLM.

        Args:
            messages: A list of message dictionaries.

        Returns:
            The LLM's response as a string.

        Raises:
            httpx.RequestError: If the request to the LLM fails.
        """
        config = {
            "max_length": 200
        }
        kwargs = {
            'model': self.model,
            'top_p': 0.1,
            'temperature': 1.0,
            'messages': messages,
            'functions': functions,
            'timeout': 120,
            'generate_config': config
        }

        try:
            data = TextGeneration.chat(**kwargs)
            message = data.choices[0].message
            if message.function_call is not None:
                return message.function_call, message.content
            else:
                return None, message.content
        except Exception as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)

            return None, (
                f"I encountered an error: {error_message}. "
                "Please try again or rephrase your request."
            )


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))

        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def process_llm_response(self, function_call: ChatFunctionCall,
                                   content: str) -> str:
        """Process the LLM response and execute tools if needed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            The result of tool execution or the original response.
        """
        if function_call is not None:
            logging.info(f"Executing tool: {function_call.name}")
            logging.info(f"With arguments: {function_call.arguments}")

            for server in self.servers:
                tools = await server.list_tools()
                if any(tool.name == function_call.name for tool in tools):
                    try:
                        result: CallToolResult = await server.execute_tool(
                            function_call.name,
                            json.loads(function_call.arguments)
                        )
                        return result.content[0].text
                    except Exception as e:
                        error_msg = f"Error executing tool: {str(e)}"
                        logging.error(error_msg)
                        return error_msg

            return f"No server found with tool: {function_call.name}"
        return content


    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

            all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)

            tools_list = [tool.format_for_llm() for tool in all_tools]

            system_message = (
                "You are a helpful assistant."
            )

            messages = [{"role": "system", "content": system_message}]

            while True:
                try:
                    user_input = input("You: ").strip().lower()
                    if user_input in ["quit", "exit"]:
                        logging.info("\nExiting...")
                        break

                    messages.append({"role": "user", "content": user_input})

                    function_call, content = self.llm_client.get_response(
                        messages,
                        tools_list)
                    logging.info("\nAssistant: %s", content)

                    result = await self.process_llm_response(function_call,
                                                             content)

                    if result != content:
                        messages.append(
                            {"role": "assistant",
                             "function_call": function_call.to_dict()})
                        messages.append({"role": "function", "content": result})

                        _, final_response = self.llm_client.get_response(
                            messages,
                            tools_list)
                        logging.info("\nFinal response: %s", final_response)
                        messages.append(
                            {"role": "assistant", "content": final_response}
                        )
                    else:
                        messages.append(
                            {"role": "assistant", "content": content})

                except KeyboardInterrupt:
                    logging.info("\nExiting...")
                    break

        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    server_endpoints = {
        "47840": "https://tppwork.taobao.com/mcppre/mcp/47840/sse"
    }
    servers = [
        Server(name, endpoint)
        for name, endpoint in server_endpoints.items()
    ]
    api_key = os.getenv("WHALE_API_KEY")
    model = "Qwen-72B-Chat-Latest"
    llm_client = LLMClient(model, api_key)
    chat_session = ChatSession(servers, llm_client)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())
