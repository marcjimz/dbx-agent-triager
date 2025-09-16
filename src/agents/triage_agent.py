import json
from typing import Annotated, Any, Generator, Optional, Sequence, TypedDict, Union, List
from uuid import uuid4

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
    DatabricksFunctionClient,
    set_uc_function_client,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

# Ensure MLflow autolog is on for end-to-end traces
mlflow.langchain.autolog()

PROMPT_INSTRUCTION = "classification_instruction"
PROMPT_CATALOG = "marcin_demo"
PROMPT_SCHEMA = "demo_schema_v2"
PROMPT_VERSION = 3

# Default LLM endpoint (can be overridden)
DEFAULT_LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"

#####################
# Define agent state
#####################
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]


# --------------------------------------------------------------------------------------
# Core helpers to create the LangGraph tool-calling agent
# --------------------------------------------------------------------------------------
def _create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[Sequence[BaseTool], ToolNode],
    system_prompt: Optional[str] = None,
):
    """Create a tool-calling agent using LangGraph."""
    model = model.bind_tools(tools)

    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there are tool_calls on the last message, keep looping
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        else:
            return "end"

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    
    model_runnable = preprocessor | model

    def call_model(
        state: AgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        }
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()


# --------------------------------------------------------------------------------------
# ResponsesAgent wrapper for MLflow compatibility
# --------------------------------------------------------------------------------------
class LangGraphResponsesAgent(ResponsesAgent):
    """ResponsesAgent wrapper for LangGraph agents."""
    
    def __init__(self, agent):
        self.agent = agent
    
    def _ensure_responses_request(self, input_data) -> ResponsesAgentRequest:
        """Convert various input formats to ResponsesAgentRequest."""
        if isinstance(input_data, ResponsesAgentRequest):
            return input_data
        
        # Handle simple message format
        if isinstance(input_data, dict):
            # Check if it's already in ResponsesAgentRequest format
            if "input" in input_data:
                return ResponsesAgentRequest(**input_data)
            
            # Handle simple {"messages": [...]} format
            if "messages" in input_data:
                messages = input_data["messages"]
                formatted_input = []
                for msg in messages:
                    if isinstance(msg, dict):
                        # Ensure the message has the right structure
                        formatted_input.append({
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", "")
                        })
                    else:
                        formatted_input.append(msg)
                
                return ResponsesAgentRequest(
                    input=formatted_input,
                    custom_inputs=input_data.get("custom_inputs")
                )
            
            # Handle single message
            if "role" in input_data and "content" in input_data:
                return ResponsesAgentRequest(
                    input=[input_data]
                )
        
        # If we can't convert, raise an error
        raise ValueError(
            f"Invalid input format. Expected ResponsesAgentRequest or dict with 'input' or 'messages' key. Got: {type(input_data)}"
        )

    def _responses_to_cc(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert from a Responses API output item to ChatCompletion messages."""
        msg_type = message.get("type")
        if msg_type == "function_call":
            return [
                {
                    "role": "assistant",
                    "content": "tool call",
                    "tool_calls": [
                        {
                            "id": message["call_id"],
                            "type": "function",
                            "function": {
                                "arguments": message["arguments"],
                                "name": message["name"],
                            },
                        }
                    ],
                }
            ]
        elif msg_type == "message" and isinstance(message.get("content"), list):
            return [
                {"role": message["role"], "content": content["text"]}
                for content in message["content"]
            ]
        elif msg_type == "reasoning":
            return [{"role": "assistant", "content": json.dumps(message["summary"])}]
        elif msg_type == "function_call_output":
            return [
                {
                    "role": "tool",
                    "content": message["output"],
                    "tool_call_id": message["call_id"],
                }
            ]
        # Handle simple message format
        elif msg_type == "message":
            return [{"role": message.get("role", "user"), "content": message.get("content", "")}]
        
        compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
        filtered = {k: v for k, v in message.items() if k in compatible_keys}
        return [filtered] if filtered else []

    def _prep_msgs_for_cc_llm(self, responses_input) -> list[dict[str, Any]]:
        """Convert from Responses input items to ChatCompletion dictionaries."""
        cc_msgs = []
        for msg in responses_input:
            cc_msgs.extend(self._responses_to_cc(msg.model_dump()))
        return cc_msgs

    def _langchain_to_responses(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """Convert from LangChain messages to Responses output item dictionaries."""
        output_items = []
        for message in messages:
            message_dict = message.model_dump() if hasattr(message, 'model_dump') else message.dict()
            msg_type = message_dict.get("type", "")
            
            if msg_type == "ai":
                if tool_calls := message_dict.get("tool_calls"):
                    for tool_call in tool_calls:
                        output_items.append(
                            self.create_function_call_item(
                                id=message_dict.get("id") or str(uuid4()),
                                call_id=tool_call.get("id", str(uuid4())),
                                name=tool_call.get("name", ""),
                                arguments=json.dumps(tool_call.get("args", {})),
                            )
                        )
                else:
                    output_items.append(
                        self.create_text_output_item(
                            text=message_dict.get("content", ""),
                            id=message_dict.get("id") or str(uuid4()),
                        )
                    )
            elif msg_type == "tool":
                output_items.append(
                    self.create_function_call_output_item(
                        call_id=message_dict.get("tool_call_id", ""),
                        output=message_dict.get("content", ""),
                    )
                )
            elif msg_type == "user":
                output_items.append(message_dict)
                
        return output_items

    def predict(self, request: Union[ResponsesAgentRequest, dict]) -> ResponsesAgentResponse:
        """Process a non-streaming request."""
        # Convert input to proper format if needed
        request = self._ensure_responses_request(request)
        
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(
            output=outputs, 
            custom_outputs=request.custom_inputs
        )

    def predict_stream(
        self,
        request: Union[ResponsesAgentRequest, dict],
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Process a streaming request."""
        # Convert input to proper format if needed
        request = self._ensure_responses_request(request)
        
        # Convert input messages to chat completion format
        cc_msgs = self._prep_msgs_for_cc_llm(request.input)

        # Stream through the agent
        for event in self.agent.stream(
            {"messages": cc_msgs}, 
            stream_mode=["updates", "messages"]
        ):
            if event[0] == "updates":
                for node_data in event[1].values():
                    messages = node_data.get("messages", [])
                    for item in self._langchain_to_responses(messages):
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done", 
                            item=item
                        )
            # Filter the streamed messages to just the generated text messages
            elif event[0] == "messages":
                try:
                    chunk = event[1][0]
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(
                                delta=chunk.content, 
                                item_id=chunk.id
                            ),
                        )
                except Exception as e:
                    print(f"Error processing message stream: {e}")


# --------------------------------------------------------------------------------------
# Public factory for ServiceNow Assignment Agent
# --------------------------------------------------------------------------------------
class ServiceNowAssignmentAgent:
    """
    Factory for building a UC-tool-calling LangGraph agent and returning 
    an MLflow ResponsesAgent.
    """

    DEFAULT_UC_FUNCTIONS = ["prod_silver.dts_ops.get_incident_info"]

    @staticmethod
    def create(
        *,
        system_prompt: str,
        endpoint_name: str = DEFAULT_LLM_ENDPOINT_NAME,
        uc_function_names: Optional[Sequence[str]] = None,
        databricks_function_client: Optional[DatabricksFunctionClient] = None,
    ) -> ResponsesAgent:
        """
        Build and return a ready-to-use ResponsesAgent that can be passed 
        to MLflow or called directly.
        
        Args:
            endpoint_name: Name of the Databricks model serving endpoint
            system_prompt: System prompt for the agent
            uc_function_names: List of Unity Catalog function names to use as tools
            databricks_function_client: Optional Databricks function client
            
        Returns:
            ResponsesAgent: Configured agent ready for use
        """
        # Wire UC function client once
        client = databricks_function_client or DatabricksFunctionClient()
        set_uc_function_client(client)

        # Create LLM
        llm = ChatDatabricks(endpoint=endpoint_name)

        # Get tools from Unity Catalog
        tool_names = list(uc_function_names or ServiceNowAssignmentAgent.DEFAULT_UC_FUNCTIONS)
        uc_toolkit = UCFunctionToolkit(function_names=tool_names)
        tools = uc_toolkit.tools  # type: ignore

        # Compile the graph
        compiled = _create_tool_calling_agent(llm, tools, system_prompt)
        
        # Return wrapped in ResponsesAgent
        return LangGraphResponsesAgent(compiled)

    @staticmethod
    def from_defaults() -> ResponsesAgent:
        """
        Convenience constructor with default endpoint and prompt.
        Edit the defaults at the top of the file as desired.
        
        Returns:
            ResponsesAgent: Agent with default configuration
        """
        # prompt = manager.load_prompt(PROMPT_INSTRUCTION) #version=2)
        return ServiceNowAssignmentAgent.create(
            endpoint_name=DEFAULT_LLM_ENDPOINT_NAME,
            system_prompt=prompt,
            uc_function_names=ServiceNowAssignmentAgent.DEFAULT_UC_FUNCTIONS,
        )


# --------------------------------------------------------------------------------------
# Initialize the agent for MLflow model registration
# --------------------------------------------------------------------------------------
from src.prompts.prompt_manager import PromptManager
prompt_manager = PromptManager(catalog=PROMPT_CATALOG, schema=PROMPT_SCHEMA)
prompt = prompt_manager.load_prompt(PROMPT_INSTRUCTION, version=PROMPT_VERSION)
AGENT = ServiceNowAssignmentAgent.create(
    endpoint_name="databricks-meta-llama-3-3-70b-instruct",
    system_prompt=prompt.template,
    uc_function_names=["prod_silver.dts_ops.get_incident_info"],
)

# Set the model for MLflow
mlflow.models.set_model(AGENT)