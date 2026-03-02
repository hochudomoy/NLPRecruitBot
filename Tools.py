from langchain.tools import tool
from langchain_core.messages import ToolMessage
def build_tools(context):

    @tool
    def change_difficulty(level:str)-> str:
        """
        Изменяет уровень сложности интервью.
        """
        context["difficulty"] = level
        print(f"[TOOL CALLED] difficult {level}:")
        return f"Difficulty changed to {level}"

    @tool
    def mark_hallucination(reason:str)-> str:
        """
        Регистрирует «галлюцинацию» с указанной причиной.
        """
        context['hallucinations'] +=1
        print(f"[TOOL CALLED] mark_hallucination")
        return f"Hallucination recorded: {reason}"

    @tool
    def end_interview(reason:str)-> str:
        """
        Завершает интервью с указанной причиной.
        """
        context['finished'] = True
        print(f"[TOOL CALLED] ending")
        return f"Interview finished: {reason}"

    @tool
    def send_signal_to_interviewer(message:str)-> str:
        """
        Отправляет сообщение интервьюеру.
        """
        context['interviewer_signal'] = message
        print(f"[TOOL CALLED] send signal {message}")
        return f"Signal sent to interviewer: {message}"

    return [
        change_difficulty,
        mark_hallucination,
        end_interview,
        send_signal_to_interviewer
    ]
def invoke_with_tools(llm, messages,tools_dict):
    response = llm.invoke(messages)
    loop_count = 0
    while response.tool_calls and loop_count<2:
        loop_count += 1
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            tool = tools_dict[tool_name]
            result = tool.invoke(tool_args)

            messages.append(response)
            messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                )
            )

        response = llm.invoke(messages)

    return response