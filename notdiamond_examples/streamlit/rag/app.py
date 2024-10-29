import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import uuid
from urllib.parse import urlparse

import streamlit as st
from streamlit_javascript import st_javascript

from notdiamond_examples.streamlit.rag.scraping import scrape_website

st.set_page_config(
    page_title="Not Diamond | RAG Documentation Finder",
    page_icon="notdiamond_examples/streamlit/icon.png",
    layout="wide",
    menu_items={
        "Get Help": "https://docs.notdiamond.ai/what-is-not-diamond",
        "Report a Bug": "mailto:support@notdiamond.ai",
        "About": """
        Not Diamond is a locally deployable AI model router which automatically determines the best LLM to respond to each query.

        You can [learn more](https://notdiamond.ai) or [chat with Not Diamond](https://chat.notdiamond.ai).
        """,
    },
)
from dotenv import load_dotenv
from haystack import component, Pipeline, Document
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack_integrations.components.generators.anthropic import AnthropicGenerator
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.preprocessors import (
    DocumentCleaner,
    DocumentSplitter,
)
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
# from haystack.core.pipeline import Pipeline
from haystack.dataclasses import GeneratedAnswer
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

from notdiamond import NotDiamond
from notdiamond.settings import NOTDIAMOND_API_KEY, NOTDIAMOND_API_URL

CONTEXT_SEPARATOR = "|||||;;;;;|||||"

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

_APP_URL = st_javascript("await fetch('').then(r => window.parent.location.href)")

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)
_LOGGER.addHandler(logging.StreamHandler(sys.stdout))

model_id_to_label = {
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gpt-4o-2024-05-13": "gpt-4o",
    "claude-3-5-sonnet-20240620": "anthropic.claude-3-5-sonnet-20240620-v1:0"
}

def get_cost(messages: List[str], output: str, model: str) -> int:
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
    cost_in_hundredth_cents = round(total_cost * 1000000)  # Convert to 1/100 cents
    return cost_in_hundredth_cents

def calculate_savings(response_cost: int, gpt4o_cost: int) -> int:
    return gpt4o_cost - response_cost

def _get_nd_client(llm_configs: List[Dict[str, Any]]):
    client = NotDiamond(
        api_key=NOTDIAMOND_API_KEY,
        nd_api_url=NOTDIAMOND_API_URL,
        llm_configs=llm_configs,
        tradeoff="cost",
    )
    user_agent_elems = client.user_agent.split('/')
    user_agent_str = "/".join(user_agent_elems[:-1] + ['rag-demo'] + user_agent_elems[-1:])
    client.user_agent = user_agent_str
    return client

@component
class NotDiamondRouter:
    def __init__(self, client: NotDiamond):
        self.nd_client = client
        assert (
            self.nd_client.llm_configs
        ), f"Expected non-zero configs but found none for {self.nd_client}"
        component.set_output_types(
            self,
            **{llmc.model: str for llmc in self.nd_client.llm_configs},
        )

    @component.output_types(routes=Dict[Any, Any])
    def run(self, prompt: str):
        try:
            system_prompt, user_prompt = prompt.split(CONTEXT_SEPARATOR)
        except ValueError as verr:
            if len(prompt.split(CONTEXT_SEPARATOR)) == 1:
                raise ValueError(
                    "Could not split prompt into system and user prompts. Is the separator correct?"
                )
            raise verr
        session_id, best_llm = self.nd_client.model_select(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tradeoff="cost",
        )
        _LOGGER.debug(
            f"Session ID: {session_id} LLM: {best_llm} Tradeoff: {self.nd_client.tradeoff} Prompt: {user_prompt}"
        )
        return {best_llm.model: prompt}

# Load the environment variables, we're going to need it for OpenAI
load_dotenv()

# This is the list of documentation that we're going to fetch
DOCUMENTATIONS = []

DOCS_PATH = Path(__file__).parent / "downloaded_docs"


def fetch(documentations: List[Tuple[str, str, Any]]):
    documents = []
    # Create the docs path if it doesn't exist
    DOCS_PATH.mkdir(parents=True, exist_ok=True)

    for name, url, pattern in documentations:
        if name == url:
            # Handle scraped website data
            for data in pattern:  # pattern contains the scraped data list
                doc = Document(
                    content=data["Content"],
                    meta={
                        "url_source": data["URL"],
                        "repo_name": name,
                    }
                )
                documents.append(doc)
            st.write(f"Fetched {len(pattern)} pages from {url}")
        else:
            # Handle GitHub repositories
            repo = DOCS_PATH / name
            # Attempt cloning only if it doesn't exist
            if not repo.exists():
                subprocess.run(["git", "clone", "--depth", "1", url, str(repo)], check=True)
            res = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                check=True,
                capture_output=True,
                encoding="utf-8",
                cwd=repo,
            )
            branch = res.stdout.strip()
            results = list(repo.glob(pattern))
            _LOGGER.info(f"{name} - {url} - {pattern} - {len(results)}")
            for p in results:
                with open(p, 'r', encoding='utf-8') as file:
                    content = file.read()
                doc = Document(
                    content=content,
                    meta={
                        "url_source": f"{url}/tree/{branch}/{p.relative_to(repo)}",
                        "suffix": p.suffix,
                        "repo_name": name,
                    }
                )
                documents.append(doc)
            st.write(f"Fetched {len(results)} files from {url.replace('https://github.com/', '')}")

    return documents


@st.cache_data
def add_extra_website(extra_website: str, follow_links: bool, exclude_header_footer: bool, max_depth: int=1):
    filter = urlparse(extra_website).netloc or '.'.join(extra_website.split(".")[-2:])
    print(f"Scraping website {extra_website} with filter {filter}")

    scraped_data = scrape_website(
        extra_website,
        filter,
        follow_links,
        exclude_header_footer,
        max_depth=max_depth,
    )

    return extra_website, scraped_data


def document_store(index: str = "documentation"):
    if "doc_store" not in st.session_state:
        st.session_state.doc_store = InMemoryDocumentStore(index=index)
    return st.session_state.doc_store


def index_files(documents):
    # Create components
    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(split_by='word', split_length=400, split_overlap=50)
    document_writer = DocumentWriter(
        document_store=document_store(), policy=DuplicatePolicy.OVERWRITE
    )

    # Build the pipeline
    indexing_pipeline = Pipeline()
    # Add components
    indexing_pipeline.add_component("cleaner", document_cleaner)
    indexing_pipeline.add_component("splitter", document_splitter)
    indexing_pipeline.add_component("writer", document_writer)
    # Connect components
    indexing_pipeline.connect("cleaner", "splitter")
    indexing_pipeline.connect("splitter", "writer")

    st.write(f"Indexing {len(documents)} documents")

    # Run the pipeline, passing 'documents' as input to 'cleaner'
    indexing_pipeline.run({"cleaner": {"documents": documents}})


def search(question: str, extra_repo: str = None, extra_website: str = None) -> GeneratedAnswer:
    doc_store = document_store()
    filters = None
    if extra_repo:
        filters = {"field": "meta.repo_name", "value": extra_repo.split("/")[-1], "operator": "=="}
    elif extra_website:
        filters = {"field": "meta.repo_name", "value": extra_website, "operator": "=="}
    else:
        # default to Not Diamond if no repo or website is provided
        filters = {"field": "meta.repo_name", "value": "https://docs.notdiamond.ai", "operator": "=="}
    _LOGGER.info(f"Question: {question} filters: {filters}")

    template = """
        Using the information contained in the context, give a comprehensive answer to the question.
        If the answer cannot be deduced from the context, explain that you're unable to deduce the
        answer from your internal data sources, and then try to answer the question as best you can.
        If it's a particular complex or specific question about a niche domain, make clear that your
        answer may be incorrect, and encourage the user to try rephrasing their question if the response
        doesn't meet their expectations.
        Context: {{ documents|map(attribute='content')|join(';')|replace('\n', ' ') }} \n
        |||||;;;;;|||||\n
        Question: {{ query }}
        Answer:
        """
    prompt_builder = PromptBuilder(template)
    gpt4o_answer_builder = AnswerBuilder()
    gpt4o_mini_answer_builder = AnswerBuilder()
    # google_answer_builder = AnswerBuilder()
    anthropic_answer_builder = AnswerBuilder()

    llm_configs = [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "anthropic/claude-3-5-sonnet-20240620",
        # "google/gemini-1.5-pro-latest",
    ]
    client = _get_nd_client(llm_configs)
    nd_router = NotDiamondRouter(client)

    # google_generator = GoogleAIGeminiGenerator(model="gemini-1.5-pro")
    anthropic_generator = AnthropicGenerator(model="claude-3-5-sonnet-20240620")
    gpt4o_generator = OpenAIGenerator(model="gpt-4o")
    gpt4o_mini_generator = OpenAIGenerator(model="gpt-4o-mini")

    # Define the retriever component with filters
    retriever = InMemoryBM25Retriever(document_store=doc_store)

    query_pipeline = Pipeline()
    query_pipeline.add_component("docs_retriever", retriever)
    query_pipeline.add_component("prompt_builder", prompt_builder)
    query_pipeline.add_component("not_diamond_router", nd_router)
    # query_pipeline.add_component("google_llm", google_generator)
    query_pipeline.add_component("anthropic_llm", anthropic_generator)
    query_pipeline.add_component("gpt4o_llm", gpt4o_generator)
    query_pipeline.add_component("gpt4o_mini_llm", gpt4o_mini_generator)
    # query_pipeline.add_component("google_answer_builder", google_answer_builder)
    query_pipeline.add_component("anthropic_answer_builder", anthropic_answer_builder)
    query_pipeline.add_component("gpt4o_answer_builder", gpt4o_answer_builder)
    query_pipeline.add_component("gpt4o_mini_answer_builder", gpt4o_mini_answer_builder)

    query_pipeline.connect("docs_retriever.documents", "prompt_builder.documents")
    # query_pipeline.connect("docs_retriever.documents", "google_answer_builder.documents")
    query_pipeline.connect(
        "docs_retriever.documents", "anthropic_answer_builder.documents"
    )
    query_pipeline.connect("docs_retriever.documents", "gpt4o_answer_builder.documents")
    query_pipeline.connect(
        "docs_retriever.documents", "gpt4o_mini_answer_builder.documents"
    )

    query_pipeline.connect("prompt_builder.prompt", "not_diamond_router.prompt")
    # query_pipeline.connect(
    #     "not_diamond_router.gemini-1.5-pro-latest", "google_llm"
    # )
    query_pipeline.connect(
        "not_diamond_router.claude-3-5-sonnet-20240620", "anthropic_llm"
    )
    query_pipeline.connect("not_diamond_router.gpt-4o", "gpt4o_llm")
    query_pipeline.connect("not_diamond_router.gpt-4o-mini", "gpt4o_mini_llm")

    # query_pipeline.connect("google_llm.replies", "google_answer_builder.replies")
    query_pipeline.connect("anthropic_llm.replies", "anthropic_answer_builder.replies")
    query_pipeline.connect("gpt4o_llm.replies", "gpt4o_answer_builder.replies")
    query_pipeline.connect(
        "gpt4o_mini_llm.replies", "gpt4o_mini_answer_builder.replies"
    )

    res = query_pipeline.run({"query": question, "filters": filters})
    _LOGGER.debug("Query pipeline result:", res)

    builder_key = [k for k in res.keys() if "answer_builder" in k][0]
    llm_key = [k for k in res.keys() if "llm" in k][0]

    # Calculate costs
    messages = [question]
    output = res[builder_key]["answers"][0].data
    response_cost = get_cost(messages, output, res[llm_key]["meta"][0]["model"])
    gpt4o_cost = get_cost(messages, output, "gpt-4o-2024-05-13")  # Use the mapped model ID
    savings = calculate_savings(response_cost, gpt4o_cost)

    return res[llm_key]["meta"][0]["model"], res[builder_key]["answers"][0], response_cost, savings


def get_extras():
    extra_repo = st.query_params.get("repo", None)
    extra_website = st.query_params.get("website", None)
    follow_links = st.query_params.get("follow_links", "false").lower() == "true"
    exclude_header_footer = st.query_params.get("exclude_header_footer", "true").lower() == "true"
    message = st.query_params.get("message", None)

    extra_dir = st.query_params.get("repo_dir", None)
    extra_ext = st.query_params.get("repo_ext", None)

    if extra_repo and not extra_dir:
        raise ValueError("repo_dir is required when repo is provided")
    if extra_repo and not extra_ext:
        _LOGGER.warning("repo_ext not provided - defaulting to .rst")
        extra_ext = "rst"

    # Compute a unique key based on parameters
    params_key = f"{extra_repo}_{extra_dir}_{extra_ext}_{extra_website}_{follow_links}_{exclude_header_footer}_{message}"

    return extra_repo, extra_dir, extra_ext, extra_website, follow_links, exclude_header_footer, message, params_key

# Get the current parameters and compute params_key
extra_repo, extra_dir, extra_ext, extra_website, follow_links, exclude_header_footer, message, params_key = get_extras()

# Check if the parameters have changed
if 'params_key' in st.session_state:
    if st.session_state.params_key != params_key:
        # Parameters have changed
        # Clear the document store
        if "doc_store" in st.session_state:
            del st.session_state.doc_store
            _LOGGER.info("Document store has been reset.")
        # Update the params_key in session_state
        st.session_state.params_key = params_key
        # Reset any document loaded flags
        for key in list(st.session_state.keys()):
            if key.startswith('documents_'):
                del st.session_state[key]
else:
    # First time setting params_key
    st.session_state.params_key = params_key

if message and "pre_filled_message" not in st.session_state:
    st.session_state.pre_filled_message = message

if extra_repo:
    DOCUMENTATIONS = [
        (
            extra_repo.split("/")[-1],
            f"https://github.com/{extra_repo}",
            f"./{extra_dir}/**/*.{extra_ext}",
        )
    ]
elif extra_website:
    extra_website, scraped_data = add_extra_website(extra_website, follow_links, exclude_header_footer)
    DOCUMENTATIONS = [(extra_website, extra_website, scraped_data)]
else:
    _, scraped_nd = add_extra_website("https://docs.notdiamond.ai", True, True, max_depth=1)
    DOCUMENTATIONS = [
        (
            "https://docs.notdiamond.ai",
            "https://docs.notdiamond.ai",
            scraped_nd
        )
    ]

col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.header("¬◇ | Not Diamond RAG Documentation Finder")

    caption = f"Use this [RAG application](https://aws.amazon.com/what-is/retrieval-augmented-generation/) to answer questions about {DOCUMENTATIONS[0][0]}. Not Diamond will route to the best model for each query, using cheaper models for simple questions"
    if len(DOCUMENTATIONS) > 2:
        caption += f", {', '.join([d[0] for d in DOCUMENTATIONS][1:-1])}"
    if len(DOCUMENTATIONS) > 1:
        caption += f" and {DOCUMENTATIONS[-1][0]}"
    caption += "."
    st.caption(caption)

with col2:
    st.text(" ")
    # Determine document key
    if extra_website:
        doc_key = f"documents_{extra_website}_{follow_links}_{exclude_header_footer}"
    elif extra_repo:
        doc_key = f"documents_{extra_repo}_{extra_dir}_{extra_ext}"
    else:
        doc_key = f"documents_https://docs.notdiamond.ai_True_True"

    # Check if the document key has changed
    if 'doc_key' in st.session_state:
        if st.session_state['doc_key'] != doc_key:
            # Document key has changed, clear document store by deleting the instance
            if 'doc_store' in st.session_state:
                del st.session_state.doc_store
                _LOGGER.info("Document store has been reset.")
            # Update the doc_key in session_state
            st.session_state['doc_key'] = doc_key
            # Remove any document loaded flags
            for key in list(st.session_state.keys()):
                if key.startswith('documents_') and key != 'doc_key':
                    del st.session_state[key]
    else:
        # First time setting doc_key
        st.session_state['doc_key'] = doc_key

    # Proceed to load and index documents if not already loaded
    if doc_key not in st.session_state:
        with st.status(
            "Downloading documentation files...",
            expanded=st.session_state.get("expanded", True),
        ) as status:
            if extra_website:
                _LOGGER.info(f"Fetching {[doc.get('URL') for d in DOCUMENTATIONS for doc in d[-1]]}")
            documents = fetch(DOCUMENTATIONS)
            status.update(label="Indexing documentation...")
            index_files(documents)
            status.update(
                label="Download and indexing complete!", state="complete", expanded=False
            )
            st.session_state["expanded"] = False
        st.session_state[doc_key] = True  # Mark documents as loaded

with st.container():
    if question := st.chat_input(placeholder=message if message else "Search documentation with Not Diamond..."):
        with st.chat_message("user"):
            st.markdown(question)

        properties = {
            "rag_query": question,
            "$process_person_profile": False,
        }
        # Use extras from st.session_state (already unpacked earlier in the script)
        if extra_repo or extra_website:
            properties["params"] = {
                "rag_repo": extra_repo,
                "rag_dir": extra_dir,
                "rag_ext": extra_ext,
                "website": extra_website,
                "message": message,
            }

        with st.chat_message("assistant", avatar="notdiamond_examples/streamlit/nd_logo_icon.png"):
            with st.spinner("Querying..."):
                model, answer, response_cost, savings = search(question, extra_repo=extra_repo, extra_website=extra_website)

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
            st.markdown(answer.data)
            with st.expander("Answer sources"):
                for document in answer.documents:
                    with st.container():
                        col1, col2 = st.columns([0.8, 0.2])
                        with col1:
                            url_source = document.meta.get("url_source", "")
                            st.write(url_source)
                        with col2:
                            st.markdown(f"_Score_: {document.score:.2f}")
                    st.markdown(f"```\n{document.content}\n```")
                    st.divider()
