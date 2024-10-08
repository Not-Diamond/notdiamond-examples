import logging
import os
import sys
from typing import List, Any, Tuple

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from notdiamond import NotDiamond, LLMConfig
from notdiamond.settings import NOTDIAMOND_API_URL
from notdiamond.toolkit.langchain import NotDiamondRoutedRunnable
import openai

import streamlit as st
from streamlit import session_state as state

st.set_page_config(
    page_title="Not Diamond | Routing Quickstart",
    page_icon="notdiamond_examples/streamlit/nd_logo_icon.png",
    layout="wide",
    menu_items={
        "Get Help": "https://notdiamond.readme.io/docs/what-is-not-diamond",
        "Report a Bug": "mailto:support@notdiamond.ai",
        "About": """
        Not Diamond is a locally deployable AI model router which automatically determines the best LLM to respond to each query.

        You can [learn more](https://notdiamond.ai) or [chat with Not Diamond](https://chat.notdiamond.ai).
        """,
    },
)

load_dotenv()

st.markdown("""
<style>
    * {
       overflow-anchor: none !important;
       }
</style>""", unsafe_allow_html=True)

st.logo("notdiamond_examples/streamlit/icon.png")

PROVIDER_TO_COST = {
    "gpt-4o-2024-05-13": {
        "input": 5,
        "output": 15,
        "max_tokens": 128_000
    },
    "gpt-4o-mini-2024-07-18": {
        "input": 0.15,
        "output": 0.6,
        "max_tokens": 128_000
    },
    "claude-3-5-sonnet-20240620": {
        "input": 3,
        "output": 15,
        "max_tokens": 200_000
    },
    "o1-preview-2024-09-12": {
        "input": 15,
        "output": 60,
        "max_tokens": 128_000
    },
    "o1-mini-2024-09-12": {
        "input": 3,
        "output": 12,
        "max_tokens": 128_000
    },
    "Meta-Llama-3.1-405B-Instruct-Turbo": {
        "input": 5,
        "output": 5,
        "max_tokens": 128_000
    },
    "mistral-large-2407": {
        "input": 2,
        "output": 6,
        "max_tokens": 128_000
    }
}

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)
_LOGGER.addHandler(logging.StreamHandler(sys.stdout))

DEFAULT_LLM_CONFIGS = [
    LLMConfig.from_string("openai/gpt-4o-2024-05-13"),
    LLMConfig.from_string("openai/gpt-4o-mini-2024-07-18"),
    LLMConfig.from_string("anthropic/claude-3-5-sonnet-20240620"),
    LLMConfig.from_string("openai/o1-preview-2024-09-12"),
    LLMConfig.from_string("openai/o1-mini-2024-09-12"),
    LLMConfig.from_string("mistral/mistral-large-2407"),
    # "google/gemini-1.5-pro-latest",
]

def get_cost(messages: List[str], output: str, model: str) -> int:
    """
    params:
      messages: List[str] - messages submitted to the model
      output: str - output from the model
      model: str - model ID
    returns:
      int - cost in 1/100 cents
    """
    input_length = sum(len(msg) for msg in messages)
    output_length = len(output)

    provider_costs = PROVIDER_TO_COST.get(model, PROVIDER_TO_COST.get(model.split("/")[-1]))
    if not provider_costs:
        _LOGGER.error(f"Provider not found for model: {model}")
        return 0

    input_tokens = input_length * 1.33
    output_tokens = output_length * 1.33

    input_cost = (input_tokens / 1_000_000) * provider_costs["input"]
    output_cost = (output_tokens / 1_000_000) * provider_costs["output"]

    total_cost = input_cost + output_cost
    cost_in_hundredth_cents = round(total_cost * 1000000)
    return cost_in_hundredth_cents

def _get_nd_user_agent(client: NotDiamond) -> None:
    user_agent_elems = client.user_agent.split('/')
    user_agent_str = "/".join(user_agent_elems[:-1] + ['streamlit'] + user_agent_elems[-1:])
    client.user_agent = user_agent_str

def stream_search(question: str, client: NotDiamond, llm_configs: List[str] = None) -> Tuple[str, Any]:
    _LOGGER.info(f"Question: {question}")

    if not llm_configs:
        llm_configs = DEFAULT_LLM_CONFIGS
    prompt_template = PromptTemplate.from_template("{question}")

    client.llm_configs = llm_configs
    nd_routed_runnable = NotDiamondRoutedRunnable(nd_client=client, temperature=1.0, nd_kwargs={'tradeoff': ND_TRADEOFF})
    # For streaming we call model_select and stream separately
    model = nd_routed_runnable._ndrunnable._model_select(question)
    chain = prompt_template | nd_routed_runnable
    result = chain.stream({"question": question})

    return model, result

def search(question: str, client: NotDiamond, llm_configs: List[str] = None) -> Tuple[str, Any]:
    _LOGGER.info(f"Question: {question}")

    if not llm_configs:
        llm_configs = DEFAULT_LLM_CONFIGS
    prompt_template = PromptTemplate.from_template("{question}")

    client.llm_configs = llm_configs
    nd_routed_runnable = NotDiamondRoutedRunnable(nd_client=client, temperature=1.0, nd_kwargs={'tradeoff': ND_TRADEOFF})
    chain = prompt_template | nd_routed_runnable
    result = chain.invoke({"question": question})

    try:
        model = result.response_metadata['model']
    except KeyError:
        model = result.response_metadata['model_name']

    return model, result.content

col1, col2, col3 = st.columns([0.5, 0.25, 0.25])
with col1:
    st.header("¬◇ | Not Diamond Routing Quickstart")
    caption = f"Use this application to explore how Not Diamond routes your prompts. Not Diamond will route to the best model for each query using the tradeoffs you specify."
    st.caption(caption)

ND_TRADEOFF = "quality"
with col2:
    ND_TRADEOFF = st.selectbox(f"Choose a routing tradeoff [default: {ND_TRADEOFF}]", ["quality", "cost", "latency"], placeholder="Choose a tradeoff")
    if ND_TRADEOFF == "quality":
        ND_TRADEOFF = None

providers_to_use = {}
with col3:
    with st.expander("Choose which models to route between, or leave blank to route to all of them:"):
        for provider in DEFAULT_LLM_CONFIGS:
            provider_str = str(provider)
            providers_to_use[provider_str] = st.checkbox(provider_str)

if 'nd_api_key' not in state:
    state.nd_api_key = os.getenv("NOTDIAMOND_API_KEY")

def _write_stream(answer):
    response_contents = []
    for chunk in answer:
        content = chunk.content
        response_contents.append(content)
        yield content
    state.response_content = ''.join(response_contents)

with st.container():
    if not state.nd_api_key or state.nd_api_key == "" or state.nd_api_key is None:

        def _set_api_key():
            os.environ["NOTDIAMOND_API_KEY"] = state.nd_api_key_input
            state.nd_api_key = state.nd_api_key_input

        api_key = st.text_input(
            "We did not detect `NOTDIAMOND_API_KEY` in your environment. Enter your API key below or create one [here](https://app.notdiamond.ai/keys).",
            type="password",
            on_change=_set_api_key,
            key="nd_api_key_input",
        )

    elif state.nd_api_key:
        nd_client = NotDiamond(
            api_key=state.nd_api_key,
            tradeoff=ND_TRADEOFF,
            nd_api_url=NOTDIAMOND_API_URL,
        )
        _get_nd_user_agent(nd_client)

        if question := st.chat_input(placeholder="Try routing with Not Diamond..."):
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant", avatar="notdiamond_examples/streamlit/nd_logo_icon.png"):
                with st.spinner("Querying..."):
                    llm_configs = [
                        LLMConfig.from_string(provider)
                        for provider in providers_to_use
                        if providers_to_use[provider]
                    ]
                    routed_model, answer = stream_search(question, nd_client, llm_configs)

                    with st.container():
                        routed_to = "Routing target: "
                        if routed_model == "gpt-4o-mini-2024-07-18":
                            routed_to += f" (simple query detected)"
                        routed_to += f" _{routed_model}_"
                        st.success(routed_to, icon="💠")

                        try:
                            st.write_stream(_write_stream(answer))
                        except openai.BadRequestError as e:
                            print("Could not call OpenAI with streamed response - probably for an o1 model. Falling back to non-streaming response.")
                            routed_model, answer = search(question, nd_client, llm_configs)
                            state.response_content = answer
                            st.markdown(answer)

                        cost_str = f"Response cost: \${get_cost(question, state.response_content, routed_model) / 100000:.4f}"
                        if 'o1' in routed_model:
                            cost_str += f"\n\n(o1 models do not support streaming)"
                        st.info(cost_str, icon='ℹ️')
