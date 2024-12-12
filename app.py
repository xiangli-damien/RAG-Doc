import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.output_parsers import RegexParser
import base64
import glob
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

# Custom theme and styling
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session states
if 'docembeddings' not in st.session_state:
    st.session_state.docembeddings = None
if 'index_built' not in st.session_state:
    st.session_state.index_built = False
if 'uploaded' not in st.session_state:
    st.session_state.uploaded = False
if 'texts' not in st.session_state:
    st.session_state.texts = []
if 'embeddings_2d' not in st.session_state:
    st.session_state.embeddings_2d = []
if 'feedback' not in st.session_state:
    st.session_state.feedback = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'model_initialized' not in st.session_state:
    st.session_state.model_initialized = False

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ----------------------------
# Check for API key
# ----------------------------
if not OPENAI_API_KEY:
    st.error("OpenAI API Key not found. Please add it in the .env file as OPENAI_API_KEY=your_key_here.")
    st.stop()

# ----------------------------
# Prompt Template Setup
# ----------------------------
prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

This should be in the following format:

Question: [question here]
Helpful Answer: [answer here]
Score: [score between 0 and 100]

Begin!

Context:
---------
{context}
---------
Question: {question}
Helpful Answer:"""

output_parser = RegexParser(
    regex=r"(.*?)\nScore: (.*)",
    output_keys=["answer", "score"],
)

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
    output_parser=output_parser
)


# ----------------------------
# Helper Functions
# ----------------------------

def clear_all():
    """Clear all session states and docs folder to restart the entire process."""
    # Clear docs folder
    files = glob.glob('docs/*')
    for f in files:
        os.remove(f)
    # Clear FAISS index if exists
    if os.path.exists("llm_faiss_index"):
        shutil.rmtree("llm_faiss_index")
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def load_documents():
    """
    Enhanced document loader supporting multiple file formats.
    Supports: PDF, TXT, DOCX, HTML, MD
    """
    docs = []
    from langchain.docstore.document import Document
    from langchain_community.document_loaders import (
        PyPDFLoader,
        UnstructuredWordDocumentLoader,
        UnstructuredHTMLLoader,
        UnstructuredMarkdownLoader,
        TextLoader
    )
    
    # Show progress bar
    with st.spinner("Loading documents..."):
        progress_bar = st.progress(0)
        files = os.listdir('docs')
        total_files = len(files)
        
        for idx, file in enumerate(files):
            filepath = os.path.join('docs', file)
            try:
                # Select appropriate loader based on file extension
                if file.lower().endswith('.pdf'):
                    loader = PyPDFLoader(filepath)
                elif file.lower().endswith('.docx'):
                    loader = UnstructuredWordDocumentLoader(filepath)
                elif file.lower().endswith('.html'):
                    loader = UnstructuredHTMLLoader(filepath)
                elif file.lower().endswith('.md'):
                    loader = UnstructuredMarkdownLoader(filepath)
                else:  # Default to text loader for .txt and other files
                    loader = TextLoader(filepath)
                
                # Load documents
                loaded_docs = loader.load()
                
                # Add metadata
                for doc in loaded_docs:
                    doc.metadata.update({
                        'source_file': file,
                        'file_type': file.split('.')[-1].lower(),
                        'file_size': os.path.getsize(filepath),
                        'created_time': os.path.getctime(filepath)
                    })
                docs.extend(loaded_docs)
                
                # Update progress
                progress = (idx + 1) / total_files
                progress_bar.progress(progress)
                
            except Exception as e:
                st.error(f"Error loading {file}: {str(e)}")
                continue
    
    return docs

def select_embeddings(embedding_type):
    """
    Return the embedding object based on user selection.
    Here we provide two examples: OpenAI and HuggingFace.
    You can add more if desired.
    """
    if embedding_type == "OpenAI Embeddings":
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    elif embedding_type == "HuggingFace Embeddings":
        # Example using a HuggingFace model. Ensure 'sentence-transformers' is installed.
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) # fallback
    return embeddings

def prepare_vectorstore(embedding_type, chunk_size, chunk_overlap):
    """
    Prepare the FAISS vector store from documents in 'docs' directory using the selected embedding and chunk sizes.
    1. Loads all documents
    2. Splits them into text chunks
    3. Creates embeddings and stores them in a FAISS index
    """
    try:
        documents = load_documents()
        if len(documents) == 0:
            st.warning("No documents found. Please upload files first.")
            return None
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        
        # Get embeddings
        embeddings = select_embeddings(embedding_type)
        
        # Create vector store
        docembeddings = FAISS.from_documents(texts, embeddings)
        
        # Store in session state
        st.session_state.texts = texts
        # Get all vectors for visualization
        all_embeddings = []
        for i in range(len(texts)):
            vector = docembeddings.index.reconstruct(i)
            all_embeddings.append(vector)
        st.session_state.embeddings_2d = all_embeddings
        
        # Save index
        docembeddings.save_local("llm_faiss_index")
        return docembeddings
        
    except Exception as e:
        st.error(f"Error preparing vector store: {str(e)}")
        return None

def load_vectorstore():
    """
    Load existing FAISS vector store from disk if exists.
    """
    # Use the embedding from the session if needed
    # or default to OpenAIEmbeddings if not defined
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.load_local("llm_faiss_index", embeddings)

def build_qa_chain(chain_type_selected):
    """
    Build the QA chain based on the selected chain type.
    Available chain types might be: stuff, map_reduce, refine, map_rerank.
    """
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    
    # 为不同的链类型创建不同的提示模板
    if chain_type_selected == "map_reduce":
        # 使用LangChain默认的提示模板
        chain = load_qa_chain(
            llm=llm,
            chain_type=chain_type_selected,
        )
    elif chain_type_selected == "refine":
        refine_template = """The original question is: {question}
        We have provided an existing answer: {existing_answer}
        We have the opportunity to refine the existing answer with some more context: {context_str}
        Given the new context, refine the original answer. If the context isn't useful, return the original answer.
        
        Refined Answer:"""
        
        refine_prompt = PromptTemplate(
            template=refine_template,
            input_variables=["question", "existing_answer", "context_str"]
        )
        
        initial_template = """Context information is below.
        ---------------------
        {context_str}
        ---------------------
        Given the context information and not prior knowledge, answer the following question: {question}
        
        Answer:"""
        
        initial_prompt = PromptTemplate(
            template=initial_template,
            input_variables=["context_str", "question"]
        )
        
        chain = load_qa_chain(
            llm=llm,
            chain_type=chain_type_selected,
            question_prompt=initial_prompt,
            refine_prompt=refine_prompt
        )
    else:  # stuff 和 map_rerank 使用默认提示模板
        chain = load_qa_chain(
            llm=llm,
            chain_type=chain_type_selected,
            prompt=PROMPT
        )
    
    return chain

def get_answer(query, k, threshold, chain_type_selected):
    """
    Given a query, perform similarity search on the vector store and use the LLM to generate an answer.
    User can specify top_k and a threshold for similarity search.
    """
    try:
        # 执行相似度搜索
        docs_with_score = st.session_state.docembeddings.similarity_search_with_score(query, k=k)
        
        # 根据阈值筛选文档
        filtered_docs = []
        for doc, score in docs_with_score:
            # 注意：score越小表示相似度越高
            if score <= threshold:
                filtered_docs.append(doc)
        
        # 如果没有文档通过阈值，使用得分最高的文档
        if not filtered_docs and docs_with_score:
            filtered_docs = [docs_with_score[0][0]]
        
        # 如果仍然没有相关文档，返回提示信息
        if not filtered_docs:
            return {
                "Answer": "I couldn't find any relevant information in the documents to answer your question.",
                "Reference": "No relevant documents found."
            }
        
        # 构建问答链并生成答案
        chain = build_qa_chain(chain_type_selected)
        results = chain({"input_documents": filtered_docs, "question": query})
        
        # 准备参考文本
        text_reference = "\n\n".join([
            f"From {doc.metadata.get('source_file', 'unknown')}:\n{doc.page_content}"
            for doc in filtered_docs
        ])
        
        return {
            "Answer": results["output_text"],
            "Reference": text_reference
        }
        
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return {
            "Answer": "An error occurred while processing your question.",
            "Reference": f"Error: {str(e)}"
        }

def get_document_names():
    """Get a list of uploaded document names."""
    return os.listdir('docs')

def preview_document(doc_name, page_size=1000):
    """
    Enhanced document preview with pagination and highlighting.
    
    Args:
        doc_name: Name of the document to preview
        page_size: Number of characters per page
    """
    try:
        file_path = os.path.join('docs', doc_name)
        if doc_name.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            text = "\n\n".join([page.page_content for page in pages])
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        # Calculate total pages
        total_pages = (len(text) + page_size - 1) // page_size
        
        # Add page selector
        current_page = st.selectbox(
            "Select page",
            range(1, total_pages + 1),
            key=f"page_selector_{doc_name}"
        )
        
        # Calculate start and end indices for current page
        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, len(text))
        
        # Display document metadata
        st.markdown(f"""
        <div class="metadata">
            📄 File: {doc_name}<br>
            📏 Total length: {len(text)} characters<br>
            📑 Pages: {current_page}/{total_pages}
        </div>
        """, unsafe_allow_html=True)
        
        # Create a text area for the current page
        page_text = text[start_idx:end_idx]
        
        # Add search functionality
        search_term = st.text_input(
            "Search in document",
            key=f"search_{doc_name}",
            placeholder="Enter text to highlight..."
        )
        
        if search_term:
            # Highlight search terms
            highlighted_text = page_text.replace(
                search_term,
                f'<span class="highlight">{search_term}</span>'
            )
        else:
            highlighted_text = page_text
        
        # Display the text in a container with custom styling
        st.markdown(f"""
        <div class="document-preview">
            {highlighted_text}
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error previewing document: {str(e)}")
        return None

def visualize_embeddings_3d(embeddings, texts):
    """
    Reduce embeddings to 3D using PCA and visualize using Plotly.
    Color points by the document source_file.
    Adjust marker size and opacity to enhance clarity.
    """
    if embeddings is None or len(embeddings) == 0:
        return None
        
    try:
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Apply PCA
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(embeddings_array)
        
        # Create DataFrame for plotting
        df = pd.DataFrame(reduced, columns=["x","y","z"])
        
        # Add text snippets and source file info
        df['text'] = [t.page_content[:50].replace('\n', ' ') for t in texts]
        df['source_file'] = [t.metadata.get('source_file', 'unknown') for t in texts]
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            df,
            x='x', y='y', z='z',
            hover_data=['text', 'source_file'],
            color='source_file',
            title="3D Visualization of Document Embeddings",
            opacity=0.7
        )
        
        # Customize the appearance
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(
            scene=dict(
                xaxis_title="First Principal Component",
                yaxis_title="Second Principal Component",
                zaxis_title="Third Principal Component"
            ),
            width=800,
            height=600
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error in visualization: {str(e)}")
        return None

# ----------------------------
# Streamlit UI Setup
# ----------------------------

# Main Title
st.markdown('<div class="header">RAG-based Document QA</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload, Analyze, and Chat with Your Documents</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Menu")
menu = st.sidebar.radio("Select a module", ["Import Data", "Documents", "Chat", "Restart"])

# Restart section
if menu == "Restart":
    if st.sidebar.button("Confirm Restart"):
        clear_all()
        st.experimental_rerun()

# ----------------------------
# Import Data Page
# ----------------------------
if menu == "Import Data":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Upload Your Documents")
    st.write("You can upload PDF, TXT, DOCX, HTML, or MD files. After uploading, set the embedding parameters and build the index.")
    
    uploaded_files = st.file_uploader("Upload files here:", type=['pdf','txt','docx','html','md'], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(os.path.join('docs', uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.session_state.uploaded = True
        st.success("Files uploaded successfully!")
    
    st.markdown("---")
    st.markdown("### Embedding Settings")
    
    # Embedding model selection
    embedding_type = st.selectbox("Select Embedding Model:", ["OpenAI Embeddings", "HuggingFace Embeddings"])
    chunk_size = st.number_input("Chunk Size (characters):", min_value=100, max_value=5000, value=1000)
    chunk_overlap = st.number_input("Chunk Overlap (characters):", min_value=0, max_value=500, value=100)
    
    build_index_button = st.button("Build Index")
    if build_index_button:
        if st.session_state.uploaded:
            with st.spinner("Building FAISS index..."):
                docembeddings = prepare_vectorstore(embedding_type, chunk_size, chunk_overlap)
                if docembeddings:
                    st.session_state.docembeddings = docembeddings
                    st.session_state.index_built = True
                    st.success("Index built successfully! You can now explore documents and chat.")
                else:
                    st.warning("No documents found. Please upload at least one file before building the index.")
        else:
            st.warning("Please upload files before building the index.")
    st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------
# Documents Page
# ----------------------------
if menu == "Documents":
    if not st.session_state.index_built:
        st.info("Please upload and build the index in 'Import Data' first.")
    else:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### Document List")
        doc_names = get_document_names()
        if len(doc_names) == 0:
            st.warning("No documents found. Please upload in 'Import Data' page.")
        else:
            selected_doc = st.selectbox("Select a document to preview and visualize:", doc_names)
            if selected_doc:
                # Preview Document
                st.markdown("**Document Preview (First 500 chars):**")
                content_preview = preview_document(selected_doc)
                
                st.markdown("---")
                # Visualization
                st.markdown("### Document Embeddings Visualization (3D)")
                fig = visualize_embeddings_3d(st.session_state.embeddings_2d, st.session_state.texts)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No embeddings found. Please build the index first.")
        st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------
# Chat Page
# ----------------------------
if menu == "Chat":
    if not st.session_state.index_built:
        st.info("Please upload and build the index in 'Import Data' first.")
    else:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### Ask Questions about Your Documents")
        
        query_input = st.text_input("Type your question here:")
        k_value = st.number_input("Number of documents to retrieve (k):", min_value=1, max_value=20, value=2)
        threshold = st.number_input("Similarity threshold (lower is stricter):", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
        
        # Chain type selection
        chain_type_selected = st.selectbox("Select QA Chain Type:",
                                           ["stuff", "map_reduce", "refine", "map_rerank"])
        
        if st.button("Get Answer"):
            if query_input.strip():
                with st.spinner("Generating answer..."):
                    result = get_answer(query_input.strip(), k_value, threshold, chain_type_selected)
                    answer = result
                    # Display the answer and references
                    st.markdown('<div class="answer-section">', unsafe_allow_html=True)
                    st.markdown("#### Answer")
                    st.write(answer["Answer"])
                    st.markdown("#### Reference")
                    st.write(answer["Reference"])
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter a question before clicking 'Get Answer'.")
        st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------
# Footer
# ----------------------------
st.markdown('<div class="footer">RAG-based Document QA - Extended Version with multiple embeddings, better visualization, and additional RAG parameters.</div>', unsafe_allow_html=True)