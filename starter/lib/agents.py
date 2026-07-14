from typing import TypedDict, List, Optional, Union, Callable, Dict
import json

from pydantic import BaseModel

from lib.state_machine import StateMachine, Step, EntryPoint, Termination, Run
from lib.llm import LLM
from lib.messages import AIMessage, UserMessage, SystemMessage, ToolMessage
from lib.tooling import Tool, ToolCall
from lib.memory import ShortTermMemory, LongTermMemory, MemoryFragment


class MemoryEvaluation(BaseModel):
    useful: bool
    description: str


class AgentState(TypedDict):
    user_query: str
    instructions: str
    messages: List[dict]
    current_tool_calls: Optional[List[ToolCall]]
    total_tokens: int
    session_id: str
    long_term_context: Optional[str]
    next_tool: Optional[str]


class Agent:
    def __init__(self,
                 model_name: str,
                 instructions: str,
                 tools: List[Tool] = None,
                 temperature: float = 0.7,
                 long_term_memory: Optional[LongTermMemory] = None,
                 memory_owner: str = "udaplay_user"):
        self.instructions = instructions
        self.tools = tools if tools else []
        self.model_name = model_name
        self.temperature = temperature
        self.long_term_memory = long_term_memory
        self.memory_owner = memory_owner

        self.memory = ShortTermMemory()
        self.workflow = self._create_state_machine()

    def _memory_recall_step(self, state: AgentState) -> dict:
        """Step logic: Query long-term memory and inject relevant context if useful"""
        try:
            result = self.long_term_memory.search(
                query_text=state["user_query"],
                owner=self.memory_owner,
                limit=3,
                namespace="udaplay"
            )
        except Exception:
            return {"long_term_context": None}

        if not result.fragments:
            return {"long_term_context": None}

        recalled_content = "\n\n".join([f.content for f in result.fragments])

        # Evaluate relevance before injecting — only use if useful
        llm = LLM(model=self.model_name, temperature=0.0)
        eval_messages = [
            SystemMessage(content=(
                "Evaluate whether the recalled memories are relevant and useful "
                "for answering the question."
            )),
            UserMessage(content=f"Question: {state['user_query']}\n\nRecalled memories:\n{recalled_content}")
        ]
        try:
            response = llm.invoke(eval_messages, response_format=MemoryEvaluation)
            evaluation = MemoryEvaluation.model_validate_json(response.content)
            if evaluation.useful:
                return {"long_term_context": recalled_content}
        except Exception:
            pass

        return {"long_term_context": None}

    def _memory_store_step(self, state: AgentState) -> dict:
        """Step logic: Persist Q&A pair and retrieved game facts to long-term memory"""
        messages = state.get("messages", [])

        final_answer = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_answer = msg.content
                break

        if not final_answer:
            return {}

        # Store Q&A pair
        self.long_term_memory.register(MemoryFragment(
            content=f"Q: {state['user_query']}\nA: {final_answer}",
            owner=self.memory_owner,
            namespace="udaplay"
        ))

        # Store retrieved game facts so future sessions can skip retrieval
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.name == "retrieve_game":
                try:
                    game_data = json.loads(msg.content)
                    if game_data and game_data != "No games found matching the query.":
                        self.long_term_memory.register(MemoryFragment(
                            content=f"Retrieved game facts: {game_data}",
                            owner=self.memory_owner,
                            namespace="udaplay"
                        ))
                except (json.JSONDecodeError, ValueError):
                    pass

        return {}

    def _prepare_messages_step(self, state: AgentState) -> dict:
        """Step logic: Prepare messages for LLM consumption"""
        messages = state.get("messages", [])

        if not messages:
            instructions = state["instructions"]
            ltm_context = state.get("long_term_context")
            if ltm_context:
                instructions = (
                    f"{instructions}\n\n---\n"
                    f"Relevant knowledge from past sessions:\n{ltm_context}"
                )
            messages = [SystemMessage(content=instructions)]

        messages.append(UserMessage(content=state["user_query"]))

        return {"messages": messages}

    def _llm_step(self, state: AgentState) -> dict:
        """Step logic: Process the current state through the LLM"""
        llm = LLM(
            model=self.model_name,
            temperature=self.temperature,
            tools=self.tools
        )

        response = llm.invoke(state["messages"])
        tool_calls = response.tool_calls if response.tool_calls else None

        current_total = state.get("total_tokens", 0)
        if response.token_usage:
            current_total += response.token_usage.total_tokens

        ai_message = AIMessage(
            content=response.content,
            tool_calls=tool_calls,
        )

        return {
            "messages": state["messages"] + [ai_message],
            "current_tool_calls": tool_calls,
            "total_tokens": current_total,
        }

    def _tool_router_step(self, state: AgentState) -> dict:
        """Step logic: Identify which tool to execute next"""
        tool_calls = state.get("current_tool_calls") or []
        next_tool = tool_calls[0].function.name if tool_calls else None
        return {"next_tool": next_tool}

    def _make_tool_step(self, tool: Tool) -> Callable:
        """Factory: returns a step logic function bound to a specific tool"""
        def step_logic(state: AgentState) -> dict:
            tool_calls = state.get("current_tool_calls") or []
            tool_messages = []
            remaining_calls = []

            for call in tool_calls:
                if call.function.name == tool.name:
                    function_args = json.loads(call.function.arguments)
                    result = str(tool(**function_args))
                    tool_message = ToolMessage(
                        content=json.dumps(result),
                        tool_call_id=call.id,
                        name=tool.name,
                    )
                    tool_messages.append(tool_message)
                else:
                    remaining_calls.append(call)

            return {
                "messages": state["messages"] + tool_messages,
                "current_tool_calls": remaining_calls or None,
                "next_tool": None,
            }
        return step_logic

    def _create_state_machine(self) -> StateMachine[AgentState]:
        """Build the state machine graph with per-tool nodes and optional memory steps"""
        machine = StateMachine[AgentState](AgentState)

        # Core steps
        entry = EntryPoint[AgentState]()
        message_prep = Step[AgentState]("message_prep", self._prepare_messages_step)
        llm_processor = Step[AgentState]("llm_processor", self._llm_step)
        termination = Termination[AgentState]()
        machine.add_steps([entry, message_prep, llm_processor, termination])

        # Optional long-term memory steps
        memory_recall = None
        memory_store = None
        if self.long_term_memory:
            memory_recall = Step[AgentState]("memory_recall", self._memory_recall_step)
            memory_store = Step[AgentState]("memory_store", self._memory_store_step)
            machine.add_steps([memory_recall, memory_store])

        # Per-tool steps + router (replaces the single tool_executor)
        tool_steps: Dict[str, Step[AgentState]] = {}
        tool_router = None
        if self.tools:
            tool_router = Step[AgentState]("tool_router", self._tool_router_step)
            machine.add_steps([tool_router])
            for t in self.tools:
                ts = Step[AgentState](t.name, self._make_tool_step(t))
                tool_steps[t.name] = ts
            machine.add_steps(list(tool_steps.values()))

        # --- Wire transitions ---

        # Entry → first step (memory recall if LTM enabled, else message prep)
        first_step = memory_recall if self.long_term_memory else message_prep
        machine.connect(entry, first_step)

        if self.long_term_memory:
            machine.connect(memory_recall, message_prep)

        machine.connect(message_prep, llm_processor)

        # The final step before termination (memory store if LTM enabled)
        end_step = memory_store if self.long_term_memory else termination

        if self.tools:
            tool_step_list = list(tool_steps.values())

            # LLM → tool_router (pending calls) or end
            machine.connect(
                llm_processor,
                [tool_router, end_step],
                lambda state: tool_router if state.get("current_tool_calls") else end_step
            )

            # Router → specific named tool step
            machine.connect(
                tool_router,
                tool_step_list,
                lambda state: tool_steps.get(state.get("next_tool"), tool_step_list[0])
            )

            # Each tool step → router (more calls pending) or back to LLM
            for ts in tool_step_list:
                machine.connect(
                    ts,
                    [tool_router, llm_processor],
                    lambda state: tool_router if state.get("current_tool_calls") else llm_processor
                )
        else:
            machine.connect(llm_processor, end_step)

        if self.long_term_memory:
            machine.connect(memory_store, termination)

        return machine

    def invoke(self, query: str, session_id: Optional[str] = None) -> Run:
        """
        Run the agent on a query.

        Args:
            query: The user's query to process
            session_id: Optional session identifier (uses "default" if None)

        Returns:
            The Run object for this invocation
        """
        session_id = session_id or "default"
        self.memory.create_session(session_id)

        previous_messages = []
        last_run: Run = self.memory.get_last_object(session_id)
        if last_run:
            last_state = last_run.get_final_state()
            if last_state:
                previous_messages = last_state["messages"]

        initial_state: AgentState = {
            "user_query": query,
            "instructions": self.instructions,
            "messages": previous_messages,
            "current_tool_calls": None,
            "total_tokens": 0,
            "session_id": session_id,
            "long_term_context": None,
            "next_tool": None,
        }

        run_object = self.workflow.run(initial_state)
        self.memory.add(run_object, session_id)

        return run_object

    def get_session_runs(self, session_id: Optional[str] = None) -> List[Run]:
        """Get all Run objects for a session"""
        return self.memory.get_all_objects(session_id)

    def reset_session(self, session_id: Optional[str] = None):
        """Reset short-term memory for a specific session"""
        self.memory.reset(session_id)
