import logging
import os
import sys
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from notdiamond import NotDiamond, LLMConfig
from notdiamond.settings import NOTDIAMOND_API_URL
from notdiamond.toolkit.langchain import NotDiamondRoutedRunnable

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

ndLLMProviders = [
    {
        "ndProvider": "openai",
        "cloudProvider": "openai",
        "ndModelId": "gpt-4o",
        "providerModelId": "gpt-4o",
        "label": "GPT-4o",
        "inputCost": 5,
        "outputCost": 15,
        "maxTokens": 128_000
    },
    {
        "ndProvider": "openai",
        "cloudProvider": "openai",
        "ndModelId": "gpt-4o-mini",
        "providerModelId": "gpt-4o-mini",
        "label": "GPT-4o Mini",
        "inputCost": 0.15,
        "outputCost": 0.6,
        "maxTokens": 128_000
    },
    {
        "ndProvider": "anthropic",
        "cloudProvider": "anthropic",
        "ndModelId": "claude-3-5-sonnet-20240620",
        "providerModelId": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "label": "Claude 3.5 Sonnet",
        "inputCost": 3,
        "outputCost": 15
    },
]

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)
_LOGGER.addHandler(logging.StreamHandler(sys.stdout))

model_id_to_label = {
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gpt-4o-2024-05-13": "gpt-4o",
    "claude-3-5-sonnet-20240620": "anthropic.claude-3-5-sonnet-20240620-v1:0"
}

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

    # Map the model ID to the corresponding label
    model_label = model_id_to_label.get(model)
    if not model_label:
        _LOGGER.error(f"Model not found: {model}")
        return 0

    provider = next((item for item in ndLLMProviders if item["providerModelId"] == model_label), None)
    if not provider:
        _LOGGER.error(f"Provider not found for model: {model_label}")
        return 0

    input_tokens = input_length * 1.33
    output_tokens = output_length * 1.33

    input_cost = (input_tokens / 1_000_000) * provider["inputCost"]
    output_cost = (output_tokens / 1_000_000) * provider["outputCost"]

    total_cost = input_cost + output_cost
    cost_in_hundredth_cents = round(total_cost * 1000000)
    return cost_in_hundredth_cents

def _get_nd_user_agent(client: NotDiamond) -> None:
    user_agent_elems = client.user_agent.split('/')
    user_agent_str = "/".join(user_agent_elems[:-1] + ['streamlit'] + user_agent_elems[-1:])
    client.user_agent = user_agent_str


def search(question: str, client: NotDiamond, llm_configs: List[str] = None) -> Tuple[str, str]:
    _LOGGER.info(f"Question: {question}")

    if not llm_configs:
        llm_configs = [
            LLMConfig.from_string("openai/gpt-4o"),
            LLMConfig.from_string("openai/gpt-4o-mini"),
            LLMConfig.from_string("anthropic/claude-3-5-sonnet-20240620"),
            # "google/gemini-1.5-pro-latest",
        ]

    prompt_template = PromptTemplate.from_template("{question}")

    client.llm_configs = llm_configs
    nd_routed_runnable = NotDiamondRoutedRunnable(nd_client=client)
    chain = prompt_template | nd_routed_runnable
    result = chain.invoke({"question": question})
    print(result.content)

    # Calculate costs
    messages = [question]
    output = result.content
    try:
        routed_model = result.response_metadata.get('model')
        if not routed_model:
            routed_model = result.response_metadata.get('model_name')
    except KeyError as kerr:
        print(result.response_metadata)
        raise kerr
    response_cost = get_cost(messages, output, routed_model)
    gpt4o_cost = get_cost(messages, output, "gpt-4o-2024-05-13")  # Use the mapped model ID
    savings = gpt4o_cost - response_cost

    return routed_model, result.content, response_cost, savings

col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.header("¬◇ | Not Diamond Routing Quickstart")
    caption = f"Use this application to explore how Not Diamond routes your prompts. Not Diamond will route to the best model for each query, while using cheaper models for simple questions."
    st.caption(caption)

providers_to_use = {}
with col2:
    with st.expander("Choose your models below:"):
        for provider in ndLLMProviders:
            provider_str = f"{provider['ndProvider']}/{provider['ndModelId']}"
            providers_to_use[provider_str] = st.checkbox(provider_str)

if 'nd_api_key' not in state:
    state.nd_api_key = os.getenv("NOTDIAMOND_API_KEY")

with st.container():
    if not state.nd_api_key:

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
            tradeoff="cost",
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
                    model, answer, response_cost, savings = search(question, nd_client, llm_configs)

                response_cost_dollars = response_cost / 100000
                savings_dollars = savings / 100000
                total_savings_1000_queries = savings_dollars * 100000

                with st.container():
                    col1, col2 = st.columns([0.5, 0.5])
                    with col1:
                        if model == "gpt-4o-mini-2024-07-18":
                            routed_to = f"""
                            Routing target (simple query detected):

                            _{model}_
                            """
                        else:
                            routed_to = f"Routing target: _{model}_"
                        st.success(routed_to, icon="💠")
                    with col2:
                        if model == "gpt-4o-mini-2024-07-18":
                            cost_str = f"""
                            Cost of response: \${response_cost_dollars}

                            Savings compared to gpt-4o over 100K queries: ${total_savings_1000_queries:,.2f}
                            """
                        else:
                            cost_str = f"""
                            Cost of response: \${response_cost_dollars}
                            """
                        st.info(cost_str, icon="💰")

                st.markdown(answer)
