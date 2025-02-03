import requests

def check_entitlement(soeid):
    """Call entitlement API to check user access."""
    url = "https://example.com/entitlement_api"  # Replace with actual API
    payload = {"soeid": soeid}
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        data = response.json()
        if data.get("entitled") and data.get("cagids"):
            return True, data["cagids"]
    return False, []


    def check_case_authorization(case_id, authorized_case_ids):
    """Validate if user has access to the specified case ID."""
    if case_id in authorized_case_ids:
        return True
    return False

    from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS database
vectorstore = FAISS.load_local("faiss_db", embedding_model)


def keyword_search(query, documents):
    """Simple keyword search in the stored documents."""
    matching_docs = [doc.page_content for doc in documents if query.lower() in doc.page_content.lower()]
    return matching_docs[:5]  # Return top 5 matches


    def hybrid_search(query):
    """Perform both vector and keyword search."""
    
    # Step 1: Retrieve using vector search
    retriever = vectorstore.as_retriever()
    vector_results = retriever.get_relevant_documents(query)

    # Step 2: Retrieve using keyword search
    keyword_results = keyword_search(query, vector_results)
    
    # Merge results (ensuring uniqueness)
    combined_results = list({doc.page_content for doc in vector_results + keyword_results})
    
    return combined_results if combined_results else ["No relevant data found."]

    from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# Initialize LLM (Replace with Mistral if needed)
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Define tools (API check, case validation, retrieval)
tools = [
    Tool(
        name="Check Entitlement",
        func=lambda soeid: check_entitlement(soeid),
        description="Checks user entitlement via API. Returns (True/False, cagid_list)."
    ),
    Tool(
        name="Check Case Authorization",
        func=lambda case_id, case_list: check_case_authorization(case_id, case_list),
        description="Verifies if user has access to a given case ID."
    ),
    Tool(
        name="Hybrid Search",
        func=hybrid_search,
        description="Performs both keyword and vector search in the database."
    )
]

# Initialize agent with tools
memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", memory=memory, verbose=True)


def process_user_query(soeid, query, case_id=None):
    """Agent workflow for entitlement, case validation, and hybrid retrieval."""
    
    # Step 1: Check user entitlement
    entitled, cagids = check_entitlement(soeid)
    if not entitled:
        return "Access Denied: No entitlements found."

    # Step 2: Check case authorization if a case ID is mentioned
    authorized_cases = ["case123", "case456"]  # Example case list (should be fetched dynamically)
    
    if case_id and not check_case_authorization(case_id, authorized_cases):
        return "Access Denied: Unauthorized case ID."

    # Step 3: Perform hybrid search
    retrieved_docs = hybrid_search(query)
    return retrieved_docs

# Example Query
response = process_user_query(soeid="user123", query="Explain AI in banking", case_id="case123")
print(response)

