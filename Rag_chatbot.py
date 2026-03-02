# import (python built-ins)
import os
import re
import tempfile
import time
import gc
import shutil
from datetime import datetime
from typing import List, Optional
import streamlit as st
from dotenv import load_dotenv

# imports langchain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# New imports for Excel processing
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader

# Import for filtering complex metadata
from langchain_community.vectorstores.utils import filter_complex_metadata


# ====================== HELPER FUNCTIONS ======================

def force_delete_vectorstore(path="chroma_index", max_attempts=5):
    """
    Force delete vectorstore using multiple strategies
    """
    if not os.path.exists(path):
        return True
    
    # Strategy 1: Try normal delete with retries
    for attempt in range(max_attempts):
        try:
            shutil.rmtree(path)
            return True
        except (PermissionError, OSError) as e:
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                gc.collect()
            else:
                # Strategy 2: Try rename (move) instead of delete
                try:
                    timestamp = int(time.time())
                    new_name = f"{path}_to_delete_{timestamp}"
                    os.rename(path, new_name)
                    
                    # Schedule deletion on next restart
                    with open(".cleanup_on_restart", "a") as f:
                        f.write(f"{new_name}\n")
                    
                    return True
                except:
                    # Strategy 3: Last resort - create empty folder and continue
                    try:
                        # Just create a new empty folder
                        shutil.rmtree(path, ignore_errors=True)
                        os.makedirs(path, exist_ok=True)
                        return True
                    except:
                        return False
    return False

def cleanup_on_startup():
    """Clean up any pending folders from previous runs"""
    if os.path.exists(".cleanup_on_restart"):
        with open(".cleanup_on_restart", "r") as f:
            folders = f.readlines()
        
        for folder in folders:
            folder = folder.strip()
            if os.path.exists(folder):
                try:
                    shutil.rmtree(folder, ignore_errors=True)
                except:
                    pass
        
        try:
            os.remove(".cleanup_on_restart")
        except:
            pass

def extract_date_from_text(text: str) -> Optional[datetime]:
    """
    Extract date from transaction text with more formats.
    Looks for date patterns commonly found in bank statements.
    """
    # Common date patterns in bank statements
    date_patterns = [
        r'(\d{1,2}/\d{1,2}/\d{4})',  # MM/DD/YYYY
        r'(\d{4}-\d{1,2}-\d{1,2})',  # YYYY-MM-DD
        r'(\d{1,2}-\d{1,2}-\d{4})',  # MM-DD-YYYY
        r'(\d{2}/\d{2}/\d{2})',       # MM/DD/YY
        r'(\d{4}-\d{2}-\d{2})',       # YYYY-MM-DD (alternative)
        r'(\d{1,2}\.\d{1,2}\.\d{4})', # DD.MM.YYYY (European)
    ]
    
    # Look for dates at the beginning of lines (common in statements)
    lines = text.split('\n')
    for line in lines[:5]:  # Check first few lines
        for pattern in date_patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    date_str = match.group(1)
                    # Try different formats
                    formats = ['%m/%d/%Y', '%Y-%m-%d', '%m-%d-%Y', '%m/%d/%y', '%d.%m.%Y']
                    for fmt in formats:
                        try:
                            return datetime.strptime(date_str, fmt)
                        except ValueError:
                            continue
                except:
                    continue
    return None

def is_date_query(query: str) -> bool:
    """Check if the query is asking about last/recent/first transactions"""
    date_keywords = ['last', 'latest', 'recent', 'final', 'newest', 'previous', 'earliest', 'first', 'oldest']
    return any(word in query.lower() for word in date_keywords)

def is_document_type_query(query: str) -> bool:
    """Check if the query is asking about document type/identification"""
    doc_keywords = [
        'what type', 'what kind', 'document type', 'type of document', 
        'what is this', 'describe this document', 'tell me about this document',
        'what document', 'what pdf', 'document information', 'document details',
        'what file', 'document name', 'what excel', 'what spreadsheet'
    ]
    return any(keyword in query.lower() for keyword in doc_keywords)

def is_invoice_total_query(query: str) -> bool:
    """Check if query is asking for invoice total"""
    total_keywords = [
        'total amount', 'total', 'grand total', 'invoice total', 
        'how much', 'what is the total', 'sum', 'amount due',
        'total cost', 'total price', 'balance', 'amount payable',
        'subtotal', 'grand total', 'total due'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in total_keywords)

def analyze_document_type(text: str, filename: str) -> dict:
    """
    Analyze document content to determine its type.
    Returns a dictionary with document type information.
    """
    text_lower = text.lower()
    
    # Define patterns for different document types
    doc_patterns = {
        'Bank Statement': {
            'keywords': ['bank', 'account', 'statement', 'transaction', 'withdrawal', 'deposit', 'balance', 'debit', 'credit'],
            'patterns': [r'\$\s*\d+', r'rs\.?\s*\d+', r'pk\d{1,2}', r'account\s*#?\s*\d+']
        },
        'Research Paper': {
            'keywords': ['abstract', 'introduction', 'methodology', 'results', 'conclusion', 'references', 'figure', 'table', 'equation'],
            'patterns': [r'doi:\s*10\.', r'et\sal\.', r'\[\d+\]', r'\(\d{4}\)']
        },
        'Legal Document': {
            'keywords': ['agreement', 'contract', 'clause', 'party', 'hereby', 'wherein', 'pursuant', 'liability', 'indemnify'],
            'patterns': [r'section\s+\d+', r'article\s+\d+', r'\([a-z]\)', r'\([ivx]+\)']
        },
        'Invoice': {
            'keywords': ['invoice', 'bill', 'payment due', 'total', 'subtotal', 'tax', 'customer', 'vendor', 'p.o.'],
            'patterns': [r'invoice\s*#?\s*\d+', r'due\s*date', r'amount\s*due']
        },
        'Resume/CV': {
            'keywords': ['experience', 'education', 'skills', 'employment', 'references', 'objective', 'summary'],
            'patterns': [r'\d+\s*years?', r'b\.?s\.?c\.?', r'm\.?b\.?a\.?', r'ph\.?d\.?']
        },
        'Technical Manual': {
            'keywords': ['instruction', 'manual', 'guide', 'step', 'warning', 'caution', 'specification', 'installation'],
            'patterns': [r'step\s+\d+', r'figure\s+\d+', r'table\s+\d+']
        },
        'Financial Report': {
            'keywords': ['revenue', 'profit', 'loss', 'balance sheet', 'income statement', 'cash flow', 'quarter', 'fiscal'],
            'patterns': [r'\$\s*[\d,]+\.?\d*', r'q[1-4]', r'fy\d{2}']
        },
        'Excel Spreadsheet': {
            'keywords': ['sheet', 'workbook', 'cell', 'column', 'row', 'table', 'data', 'spreadsheet', 'excel'],
            'patterns': [r'[A-Z]+\d+', r'sheet\d+', r'table', r'pivot']
        }
    }
    
    # Score each document type
    scores = {}
    for doc_type, patterns in doc_patterns.items():
        score = 0
        
        # Check keywords
        for keyword in patterns['keywords']:
            if keyword in text_lower:
                score += 2
        
        # Check regex patterns
        for pattern in patterns['patterns']:
            if re.search(pattern, text_lower):
                score += 3
        
        # Check filename
        if doc_type.lower().replace(' ', '') in filename.lower().replace(' ', ''):
            score += 5
            
        scores[doc_type] = score
    
    # Get best match
    best_match = max(scores, key=scores.get)
    confidence = scores[best_match] / 10  # Normalize to 0-1
    
    # Get document summary
    first_page = text[:500] + "..." if len(text) > 500 else text
    
    return {
        'type': best_match if scores[best_match] > 5 else 'Unknown',
        'confidence': confidence,
        'all_scores': scores,
        'filename': filename,
        'preview': first_page,
        'suggested_queries': [
            f"What are the key points in this {best_match.lower()}?",
            f"Summarize this {best_match.lower()}",
            f"What is the main purpose of this document?"
        ]
    }

def get_all_documents_with_dates(vectorstore) -> List[tuple]:
    """
    Retrieve all documents from vectorstore and extract their dates.
    Returns list of (date, document) tuples sorted by date.
    """
    try:
        # Get all documents from vectorstore
        all_docs_data = vectorstore.get()
        
        docs_with_dates = []
        
        if not all_docs_data or not all_docs_data.get('documents'):
            st.warning("No documents found in vectorstore")
            return []
        
        st.info(f"Processing {len(all_docs_data['documents'])} documents for dates...")
        
        for i, content in enumerate(all_docs_data['documents']):
            # First check if date is in metadata
            date = None
            metadata = {}
            
            if all_docs_data.get('metadatas') and i < len(all_docs_data['metadatas']):
                metadata = all_docs_data['metadatas'][i]
                
                # Try to get date from metadata (stored as string)
                if metadata.get('transaction_date'):
                    try:
                        date = datetime.strptime(metadata['transaction_date'], '%Y-%m-%d')
                    except:
                        pass
            
            # If not in metadata, try to extract from content
            if not date:
                date = extract_date_from_text(content)
            
            if date:
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                docs_with_dates.append((date, doc))
        
        # Sort by date (descending = newest first)
        docs_with_dates.sort(key=lambda x: x[0], reverse=True)
        
        st.info(f"Found {len(docs_with_dates)} documents with valid dates")
        return docs_with_dates
    
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return []

def format_transactions_for_context(transactions: List[tuple], max_transactions: int = 10, max_chars: int = 3000) -> str:
    """
    Format transaction documents into a readable context string.
    Limits size to avoid token limits.
    """
    formatted = []
    total_chars = 0
    
    for i, (date, doc) in enumerate(transactions[:max_transactions]):
        # Get page number from metadata if available
        page_num = doc.metadata.get('page_num', '?')
        source = doc.metadata.get('source_file', 'Unknown')
        
        # Truncate content if needed
        content = doc.page_content
        if len(content) > 200:  # Limit each transaction to ~200 chars
            content = content[:200] + "..."
        
        entry = f"[{date.strftime('%m/%d/%Y')}] (Page {page_num}): {content}"
        
        # Check if adding this would exceed limit
        if total_chars + len(entry) > max_chars:
            break
            
        formatted.append(entry)
        total_chars += len(entry)
    
    return "\n\n".join(formatted)

def _join_docs(docs, max_chars=4000):  # Reduced from 7000
    """Helper function for joining documents with size limit"""
    chunks, total = [], 0
    for d in docs:
        piece = d.page_content
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "\n\n---\n\n".join(chunks)

def process_excel_file(file_path: str, filename: str) -> List[Document]:
    """
    Process Excel file and convert to Documents with enhanced invoice total extraction
    """
    documents = []
    
    try:
        # Read Excel file
        excel_file = pd.ExcelFile(file_path)
        
        # Process each sheet
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # ===== ENHANCEMENT 1: Create a summary document with key invoice fields =====
            summary_content = f"=== INVOICE SUMMARY for {filename} ===\n"
            summary_content += f"Sheet: {sheet_name}\n"
            summary_content += f"Total Rows: {len(df)}\n"
            summary_content += f"Total Columns: {len(df.columns)}\n"
            summary_content += f"Columns: {', '.join(str(col) for col in df.columns)}\n\n"
            
            # Look for total amount in the dataframe
            total_amount = None
            total_row_index = None
            total_column_name = None
            
            # Common column names that might contain the total
            total_columns = ['total', 'Total', 'TOTAL', 'amount', 'Amount', 'AMOUNT', 
                           'grand total', 'Grand Total', 'GRAND TOTAL', 'invoice total',
                           'subtotal', 'Subtotal', 'SUBTOTAL', 'balance', 'Balance', 'BALANCE']
            
            # Common row indicators for totals
            total_keywords = ['total', 'Total', 'TOTAL', 'grand', 'Grand', 'GRAND', 
                            'subtotal', 'Subtotal', 'SUBTOTAL', 'sum', 'Sum', 'SUM',
                            'final', 'Final', 'FINAL', 'balance', 'Balance', 'BALANCE']
            
            # First, try to find a row that explicitly says "TOTAL"
            for idx, row in df.iterrows():
                row_text = ' '.join(str(val) for val in row.values if pd.notna(val))
                if any(keyword in row_text for keyword in total_keywords):
                    # This might be a total row
                    total_row_index = idx
                    summary_content += f"🔍 Found potential total row at index {idx}: {row_text}\n"
                    
                    # Look for numeric values in this row
                    for col in df.columns:
                        val = row[col]
                        if pd.notna(val) and isinstance(val, (int, float)):
                            total_amount = val
                            total_column_name = col
                            summary_content += f"💰 Found amount {total_amount} in column '{col}'\n"
                            break
                    
                    if total_amount:
                        break
            
            # If no total row found, check the last row (often contains totals)
            if total_amount is None and len(df) > 0:
                last_row = df.iloc[-1]
                for col in df.columns:
                    val = last_row[col]
                    if pd.notna(val) and isinstance(val, (int, float)):
                        # Check if this column name suggests it's a total
                        if any(total_term in str(col).lower() for total_term in ['total', 'amount', 'sum', 'balance']):
                            total_amount = val
                            total_column_name = col
                            summary_content += f"💰 Found amount in last row, column '{col}': {total_amount}\n"
                            break
            
            # Also look for any column that might contain the total
            if total_amount is None:
                for col in df.columns:
                    if any(total_term in str(col).lower() for total_term in ['total', 'amount', 'sum', 'balance']):
                        # Check the last few rows of this column
                        for idx in range(max(0, len(df)-5), len(df)):
                            val = df.iloc[idx][col]
                            if pd.notna(val) and isinstance(val, (int, float)):
                                total_amount = val
                                total_column_name = col
                                total_row_index = idx
                                summary_content += f"💰 Found amount in column '{col}' at row {idx}: {total_amount}\n"
                                break
                    if total_amount:
                        break
            
            # Add the total to summary if found
            if total_amount:
                summary_content += f"\n✅ **INVOICE TOTAL: {total_amount}**\n"
                if total_column_name:
                    summary_content += f"   (Found in column: '{total_column_name}'"
                    if total_row_index is not None:
                        summary_content += f", row: {total_row_index + 1})"
                    summary_content += "\n"
            
            # Create summary document
            summary_doc = Document(
                page_content=summary_content,
                metadata={
                    "source_file": filename,
                    "sheet_name": sheet_name,
                    "file_type": "excel",
                    "document_type": "invoice_summary",
                    "has_total": total_amount is not None,
                    "total_amount": float(total_amount) if total_amount else None
                }
            )
            documents.append(summary_doc)
            
            # ===== ENHANCEMENT 2: Create a totals-only document =====
            if total_amount is not None:
                totals_content = f"=== INVOICE TOTAL ===\n"
                totals_content += f"File: {filename}\n"
                totals_content += f"Sheet: {sheet_name}\n"
                totals_content += f"TOTAL AMOUNT: {total_amount}\n"
                
                if total_column_name:
                    totals_content += f"Found in column: {total_column_name}\n"
                if total_row_index is not None:
                    totals_content += f"Found at row: {total_row_index + 1}\n"
                
                totals_doc = Document(
                    page_content=totals_content,
                    metadata={
                        "source_file": filename,
                        "sheet_name": sheet_name,
                        "file_type": "excel",
                        "document_type": "invoice_total_only",
                        "total_amount": float(total_amount)
                    }
                )
                documents.append(totals_doc)
            
            # ===== ENHANCEMENT 3: Create chunked data documents (for reference) =====
            # Group rows into chunks of 50
            chunk_size = 50
            for chunk_start in range(0, len(df), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(df))
                chunk_df = df.iloc[chunk_start:chunk_end]
                
                chunk_content = f"=== INVOICE DATA - {filename} ===\n"
                chunk_content += f"Sheet: {sheet_name}\n"
                chunk_content += f"Rows {chunk_start + 1} to {chunk_end}\n"
                chunk_content += f"Columns: {', '.join(str(col) for col in df.columns)}\n\n"
                
                # Format as readable text
                for idx, row in chunk_df.iterrows():
                    chunk_content += f"Row {idx + 1}:\n"
                    for col in df.columns:
                        value = row[col]
                        if pd.notna(value):
                            chunk_content += f"  {col}: {value}\n"
                    chunk_content += "\n"
                
                # If this chunk contains the total row, highlight it
                if total_row_index is not None and chunk_start <= total_row_index < chunk_end:
                    chunk_content += f"\n⚠️ **THIS CHUNK CONTAINS THE TOTAL ROW (Row {total_row_index + 1})**\n"
                
                chunk_doc = Document(
                    page_content=chunk_content,
                    metadata={
                        "source_file": filename,
                        "sheet_name": sheet_name,
                        "file_type": "excel",
                        "chunk_start": chunk_start + 1,
                        "chunk_end": chunk_end,
                        "document_type": "invoice_data",
                        "contains_total_row": total_row_index is not None and chunk_start <= total_row_index < chunk_end
                    }
                )
                documents.append(chunk_doc)
            
            # Also create separate documents for each row if there aren't too many rows
            if len(df) <= 100:  # Only for smaller sheets
                for idx, row in df.iterrows():
                    row_content = f"Sheet: {sheet_name}, Row {idx + 1}\n"
                    for col in df.columns:
                        value = row[col]
                        if pd.notna(value):
                            row_content += f"{col}: {value}\n"
                    
                    # Mark if this is the total row
                    is_total_row = (total_row_index is not None and idx == total_row_index)
                    if is_total_row:
                        row_content += f"\n*** THIS IS THE TOTAL ROW - AMOUNT: {total_amount} ***\n"
                    
                    row_doc = Document(
                        page_content=row_content,
                        metadata={
                            "source_file": filename,
                            "sheet_name": sheet_name,
                            "row_num": idx + 1,
                            "file_type": "excel",
                            "is_row": True,
                            "is_total_row": is_total_row,
                            "total_amount": float(total_amount) if is_total_row and total_amount else None
                        }
                    )
                    documents.append(row_doc)
    
    except Exception as e:
        st.error(f"Error processing Excel file {filename}: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
    
    return documents

# ====================== MAIN APP SETUP ======================

# Run cleanup on startup
cleanup_on_startup()

# setup : env + streamlit page
load_dotenv() 
st.set_page_config(page_title="📝 RAG Q&A", layout="wide")
st.title("📝 RAG Q&A with Multiple PDFs + Excel Files + Chat History")

# Sidbar config: Groq API Key input
with st.sidebar:
    st.header("⚙️ Config")
    api_key_input = st.text_input("Groq API Key", type="password")
    st.caption("Upload PDFs/Excel -> Ask questions -> Get Answers")
    
    st.header("📄 Page Options")
    # Option to process all pages or selected range
    process_all_pages = st.checkbox("Process ALL pages", value=True, 
                                   help="If unchecked, you can select a page range (for PDFs only)")
    
    if not process_all_pages:
        col1, col2 = st.columns(2)
        with col1:
            start_page = st.number_input("Start Page", min_value=1, value=1)
        with col2:
            end_page = st.number_input("End Page", min_value=1, value=100)
    
    # Excel specific options
    st.header("📊 Excel Options")
    process_excel_rows = st.checkbox("Process individual rows", value=False,
                                    help="If checked, each row becomes a separate document chunk")
    max_excel_rows = st.slider("Max rows per sheet to process", min_value=10, max_value=1000, value=100,
                              help="Limit the number of rows processed from each sheet")
    
    # Add token limit options
    st.header("⚡ Token Management")
    max_transactions = st.slider("Max transactions to show", min_value=5, max_value=20, value=10,
                                help="Reduce this if you hit token limits")
    
    # Add debug mode toggle
    debug_mode = st.checkbox("🔧 Debug Mode", value=True)
    
    # Add option to clear vectorstore
    if st.button("🗑️ Clear Vector Store"):
        with st.spinner("Clearing vector store..."):
            # Delete the vectorstore object if it exists
            if 'vectorstore' in st.session_state:
                del st.session_state['vectorstore']
            if 'vectorstore' in globals():
                globals().pop('vectorstore', None)
            
            # Force garbage collection
            gc.collect()
            time.sleep(1)
            
            # Force delete with multiple strategies
            if force_delete_vectorstore("chroma_index"):
                st.success("Vector store cleared!")
                st.rerun()
            else:
                st.warning("Vector store will be cleared on next restart. Please restart the app.")
                # Create marker for next restart
                with open(".cleanup_on_restart", "a") as f:
                    f.write("chroma_index\n")

# Accept key from input or .env
api_key = api_key_input or os.getenv("GROQ_API_KEY")

if not api_key:
    st.warning("Please enter your Groq API Key (or set GROQ_API_KEY in .env)")
    st.stop()

# embeddings and llm initialization
@st.cache_resource
def init_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

embeddings = init_embeddings()

# Use a smaller model to save tokens
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.1
)

# upload files (multiple - PDF and Excel)
uploaded_files = st.file_uploader(
    "📚 Upload PDF or Excel files",
    type=["pdf", "xlsx", "xls"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload one or more PDFs or Excel files to begin")
    st.stop()

# ===== USE UNIQUE INDEX FOR EACH SESSION =====
# Instead of trying to delete, use a unique folder per upload session
import uuid
session_index = f"chroma_index_{uuid.uuid4().hex[:8]}"
INDEX_DIR = session_index

# Process all files (PDF and Excel)
all_docs = []
tmp_paths = []
total_pages = 0
pdf_files = []
excel_files = []

# Separate files by type
for file in uploaded_files:
    if file.name.endswith('.pdf'):
        pdf_files.append(file)
    elif file.name.endswith(('.xlsx', '.xls')):
        excel_files.append(file)

# Process PDF files
for pdf in pdf_files:
    with st.spinner(f"Processing PDF: {pdf.name}..."):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(pdf.getvalue())
        tmp.close()
        tmp_paths.append(tmp.name)    

        loader = PyPDFLoader(tmp.name)
        docs = loader.load()
        total_pages += len(docs)
        
        # Filter by page range if needed
        for i, d in enumerate(docs):
            page_num = i + 1  # Pages are 1-indexed for user
            
            # Check if this page should be included
            include_page = process_all_pages
            if not process_all_pages:
                include_page = (start_page <= page_num <= end_page)
            
            if include_page:
                d.metadata["source_file"] = pdf.name
                d.metadata["page_num"] = page_num
                d.metadata["total_pages"] = len(docs)
                d.metadata["file_type"] = "pdf"
                
                # Try to extract and store date in metadata (as string only!)
                date = extract_date_from_text(d.page_content)
                if date:
                    # Store as string ONLY - no datetime objects!
                    d.metadata["transaction_date"] = date.strftime('%Y-%m-%d')

                all_docs.append(d)

# Process Excel files
for excel in excel_files:
    with st.spinner(f"Processing Excel: {excel.name}..."):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(excel.name)[1])
        tmp.write(excel.getvalue())
        tmp.close()
        tmp_paths.append(tmp.name)
        
        # Process Excel file with enhanced function
        excel_docs = process_excel_file(tmp.name, excel.name)
        
        # Limit rows if needed (but keep total documents always)
        if not process_excel_rows:
            # Keep sheet-level documents and total documents, remove individual row docs
            excel_docs = [doc for doc in excel_docs if not doc.metadata.get('is_row', False) or doc.metadata.get('is_total_row', False)]
        
        all_docs.extend(excel_docs)
        st.info(f"Added {len(excel_docs)} documents from {excel.name}")

# Show processing stats
if len(pdf_files) > 0:
    if process_all_pages:
        st.success(f"✅ Loaded {len([d for d in all_docs if d.metadata.get('file_type') == 'pdf'])} pages from {len(pdf_files)} PDFs (all pages)")
    else:
        st.success(f"✅ Loaded {len([d for d in all_docs if d.metadata.get('file_type') == 'pdf'])} pages from pages {start_page}-{end_page} (out of {total_pages} total PDF pages)")

if len(excel_files) > 0:
    excel_docs_count = len([d for d in all_docs if d.metadata.get('file_type') == 'excel'])
    st.success(f"✅ Loaded {excel_docs_count} documents from {len(excel_files)} Excel file(s)")
    
    # Show if totals were found
    total_docs = [d for d in all_docs if d.metadata.get('document_type') == 'invoice_total_only']
    if total_docs:
        st.success(f"💰 Found {len(total_docs)} invoice total(s) in Excel files")

# Clean up temp files
for p in tmp_paths:
    try:
        os.unlink(p)
    except Exception:
        pass

if len(all_docs) == 0:
    st.error("No documents were selected for processing. Please check your files.")
    st.stop()

# chunking (split text) - only for non-excel documents? Actually we can chunk everything
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80
)

# Split all documents
splits = text_splitter.split_documents(all_docs)

# IMPORTANT: Filter out any complex metadata
st.info("Filtering complex metadata...")
splits = filter_complex_metadata(splits)

# Create new vectorstore with unique index name
try:
    vectorstore = Chroma.from_documents(
        splits,
        embeddings,
        persist_directory=INDEX_DIR
    )
    st.sidebar.info(f"✨ Created new index with {len(splits)} chunks from {len(uploaded_files)} file(s)")
except Exception as e:
    st.error(f"Error creating vectorstore: {str(e)}")
    st.stop()

# Store vectorstore in session state
st.session_state.vectorstore = vectorstore

# Create standard retriever
standard_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 20}
)

st.sidebar.write(f"🔍 Total indexed: {len(splits)} chunks")
st.sidebar.write(f"📁 Index folder: {INDEX_DIR}")

# Display file type summary
pdf_count = len([d for d in splits if d.metadata.get('file_type') == 'pdf'])
excel_count = len([d for d in splits if d.metadata.get('file_type') == 'excel'])
st.sidebar.write(f"📄 PDF chunks: {pdf_count}")
st.sidebar.write(f"📊 Excel chunks: {excel_count}")

# Get list of current files for document type queries
current_files = [file.name for file in uploaded_files]

# prompts
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's latest question into a standalone search query using the chat history for the context. Return only the rewritten query, no extra text."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a STRICT RAG assistant. You must answer using ONLY the provided context.\n"
     "If the context does NOT contain the answer, reply exactly:\n"
     "'Out of scope - not found in provided documents.'\n"
     "Do NOT use outside knowledge. \n\n"
     "Context:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# session state for chat history
if "chathistory" not in st.session_state:
    st.session_state.chathistory = {}

def get_history(session_id: str):
    if session_id not in st.session_state.chathistory:
        st.session_state.chathistory[session_id] = ChatMessageHistory()
    return st.session_state.chathistory[session_id]

# chat ui
session_id = st.text_input("🆔 Session ID", value="default_session")
user_q = st.chat_input("💬 Ask a question...")

# ====================== DISPLAY EXISTING CHAT HISTORY ======================

# Display all previous messages first
if session_id in st.session_state.chathistory:
    for message in st.session_state.chathistory[session_id].messages:
        with st.chat_message("user" if message.type == "human" else "assistant"):
            st.markdown(message.content)

# ====================== MAIN CHAT LOGIC ======================

if user_q:
    history = get_history(session_id)
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_q)
    
    with st.spinner("Processing your question..."):
        
        # For LLM context, use limited history to save tokens
        recent_history = history.messages[-4:] if len(history.messages) >= 4 else history.messages
        
        # Check if this is an invoice total query (check first!)
        if is_invoice_total_query(user_q):
            if debug_mode:
                st.info("💰 Detected invoice total query - searching for total amount")
            
            # Try to get totals-only documents first
            try:
                # Get all documents and filter for total documents
                all_docs_data = vectorstore.get()
                total_docs = []
                
                if all_docs_data and all_docs_data.get('documents'):
                    for i, content in enumerate(all_docs_data['documents']):
                        metadata = all_docs_data['metadatas'][i] if all_docs_data.get('metadatas') else {}
                        
                        # Look for invoice_total_only documents (highest priority)
                        if metadata.get('document_type') == 'invoice_total_only':
                            doc = Document(page_content=content, metadata=metadata)
                            total_docs.append(doc)
                        
                        # Also look for invoice_summary with totals
                        elif metadata.get('document_type') == 'invoice_summary' and metadata.get('has_total'):
                            doc = Document(page_content=content, metadata=metadata)
                            total_docs.append(doc)
                        
                        # Look for rows that are total rows
                        elif metadata.get('is_total_row'):
                            doc = Document(page_content=content, metadata=metadata)
                            total_docs.append(doc)
                    
                    if total_docs:
                        # Use the total documents as context
                        context_str = "\n\n".join([doc.page_content for doc in total_docs[:3]])
                        docs = total_docs
                        
                        # Create a prompt specifically for total extraction
                        total_prompt = ChatPromptTemplate.from_messages([
                            ("system", """You are analyzing an invoice. Your task is to extract and report the TOTAL AMOUNT.

From the provided context, find the invoice total. It might be labeled as:
- Total
- Grand Total
- Invoice Total
- Amount Due
- Total Amount
- Subtotal (if that's the final amount)

If you find the total, respond with:
"The total amount for this invoice is [amount]."

If you find multiple amounts, identify which one is the final total.
If you cannot find the total, say "I couldn't find the total amount in this invoice."""),
                            ("human", f"Context:\n{context_str}\n\nQuestion: {user_q}")
                        ])
                        
                        qa_msgs = total_prompt.format_messages()
                        answer = llm.invoke(qa_msgs).content
                        standalone_q = f"INVOICE TOTAL QUERY: {user_q}"
                        
                        if debug_mode:
                            st.success(f"✅ Found {len(total_docs)} total-specific documents")
                    else:
                        # Fall back to regular retrieval but with higher k
                        if debug_mode:
                            st.info("No dedicated total documents found, searching through invoice data...")
                        
                        # Create a specialized retriever for invoice queries
                        invoice_retriever = vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 15}
                        )
                        docs = invoice_retriever.invoke(user_q + " total amount sum invoice")
                        docs = docs[:8]
                        
                        if docs:
                            context_str = _join_docs(docs, max_chars=3000)
                            
                            # Use a focused prompt for total extraction
                            extraction_prompt = ChatPromptTemplate.from_messages([
                                ("system", """You are analyzing invoice data. Find the TOTAL AMOUNT from the context.

Look for:
- A row labeled "Total", "Grand Total", or similar
- The last row of numerical data (often contains totals)
- Any amount that appears to be the sum

If you find the total, respond with:
"The total amount for this invoice is [amount]."

If you find multiple amounts, identify the final total.
Only respond with the total if you are confident."""),
                                ("human", f"Context:\n{context_str}\n\nQuestion: {user_q}")
                            ])
                            
                            qa_msgs = extraction_prompt.format_messages()
                            answer = llm.invoke(qa_msgs).content
                            standalone_q = f"INVOICE SEARCH: {user_q}"
                        else:
                            answer = "No invoice information found in the documents."
                            docs = []
                            standalone_q = user_q
                            
            except Exception as e:
                st.error(f"Error processing invoice query: {str(e)}")
                answer = "Error processing your query. Please try again."
                docs = []
                standalone_q = user_q
        
        # Check if this is a document type query
        elif is_document_type_query(user_q):
            if debug_mode:
                st.info("📄 Detected document type query - analyzing current document content")
            
            # Get all documents to analyze
            all_docs_data = vectorstore.get()
            
            if not all_docs_data or not all_docs_data.get('documents'):
                answer = "No documents found in the vectorstore."
                docs = []
                standalone_q = user_q
            else:
                # Combine all document content for analysis (only current files)
                full_text = " ".join(all_docs_data['documents'][:20])  # Use more chunks for better analysis
                
                # Get the current filename(s)
                if len(current_files) == 1:
                    filename = current_files[0]
                    file_info = f"**Filename:** {filename}"
                else:
                    filename = "multiple files"
                    file_info = "**Files:**\n" + "\n".join([f"• {f}" for f in current_files])
                
                # Analyze document type
                doc_analysis = analyze_document_type(full_text, filename)
                
                # Format response
                if doc_analysis['type'] != 'Unknown':
                    answer = f"""📋 **Document Analysis**

**Document Type:** {doc_analysis['type']} (Confidence: {doc_analysis['confidence']:.1%})

{file_info}

**Document Preview:**
> {doc_analysis['preview']}

**Suggested Questions:**
• {doc_analysis['suggested_queries'][0]}
• {doc_analysis['suggested_queries'][1]}
• {doc_analysis['suggested_queries'][2]}

**Key Indicators Found:**
"""
                    # Add top indicators
                    top_types = sorted(doc_analysis['all_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
                    for doc_type, score in top_types:
                        if score > 0:
                            answer += f"• {doc_type}: {score} indicators\n"
                else:
                    answer = f"""📄 **Document Information**

{file_info}

I couldn't definitively determine the document type, but here's a preview:
> {doc_analysis['preview']}

Try asking more specific questions about the content!"""
                
                docs = []
                standalone_q = f"DOCUMENT TYPE QUERY: {user_q}"
        
        # Check if this is a date-based query
        elif is_date_query(user_q):
            if debug_mode:
                st.info("🔍 Detected date-based query - analyzing ALL documents for chronological order")
            
            # Get all documents with their dates
            docs_with_dates = get_all_documents_with_dates(vectorstore)
            
            if not docs_with_dates:
                answer = "No transactions with dates found in the documents. Please check if your PDF contains date information."
                docs = []
                standalone_q = user_q
            else:
                # Determine what the user is asking
                query_lower = user_q.lower()
                
                # Get the actual first and last for verification
                newest_date, newest_doc = docs_with_dates[0]  # First in list (newest)
                oldest_date, oldest_doc = docs_with_dates[-1]  # Last in list (oldest)
                
                if debug_mode:
                    st.info(f"Date range found: {oldest_date.strftime('%m/%d/%Y')} to {newest_date.strftime('%m/%d/%Y')}")
                
                # Create a SUMMARIZED context with limited transactions
                if 'first' in query_lower or 'oldest' in query_lower:
                    # Show oldest transactions
                    context_transactions = docs_with_dates[-5:]  # Last 5 (oldest)
                    context_desc = "OLDEST transactions first"
                elif 'last' in query_lower or 'latest' in query_lower or 'recent' in query_lower:
                    # Show newest transactions
                    context_transactions = docs_with_dates[:5]  # First 5 (newest)
                    context_desc = "NEWEST transactions first"
                else:
                    # Show both ends
                    context_transactions = docs_with_dates[:3] + docs_with_dates[-3:]  # 3 newest + 3 oldest
                    context_desc = "NEWEST and OLDEST transactions"
                
                # Format with strict size limits
                context_lines = [f"TRANSACTIONS ({context_desc}):"]
                for i, (date, doc) in enumerate(context_transactions, 1):
                    # Extract just the essential info
                    content_preview = doc.page_content[:100].replace('\n', ' ')  # First 100 chars only
                    context_lines.append(f"{i}. {date.strftime('%m/%d/%Y')}: {content_preview}...")
                
                context_str = "\n".join(context_lines)
                
                if debug_mode:
                    st.info(f"Context size: ~{len(context_str)} chars")
                
                # Create a specialized prompt for date queries
                date_prompt = ChatPromptTemplate.from_messages([
                    ("system", f"""You are analyzing bank transactions. Here are key transactions from the document.

Date range in document: {oldest_date.strftime('%m/%d/%Y')} (oldest) to {newest_date.strftime('%m/%d/%Y')} (newest)

IMPORTANT: 
- The OLDEST transaction is from {oldest_date.strftime('%m/%d/%Y')}
- The NEWEST transaction is from {newest_date.strftime('%m/%d/%Y')}

{context_str}

Answer the user's question about transactions. Be specific about dates and amounts."""),
                    ("human", "{input}")
                ])
                
                qa_msgs = date_prompt.format_messages(input=user_q)
                answer = llm.invoke(qa_msgs).content
                
                # For debug panel
                docs = [doc for _, doc in docs_with_dates[:5]]
                standalone_q = f"DATE QUERY (limited to {len(context_transactions)} transactions): {user_q}"
        
        else:
            # Normal RAG flow for non-date queries
            rewrite_msgs = contextualize_q_prompt.format_messages(
                chat_history=recent_history,
                input=user_q
            )
            standalone_q = llm.invoke(rewrite_msgs).content.strip()

            docs = standard_retriever.invoke(standalone_q)
            docs = docs[:3]

            if not docs:
                answer = "Out of scope — not found in provided documents."
            else:
                context_str = _join_docs(docs, max_chars=2000)
                qa_msgs = qa_prompt.format_messages(
                    chat_history=recent_history,
                    input=user_q,
                    context=context_str
                )
                answer = llm.invoke(qa_msgs).content
    
    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Add to history
    history.add_user_message(user_q)
    history.add_ai_message(answer)

    # Debug panels (only if debug mode is on)
    if debug_mode:
        with st.expander("🧪 Debug: Query Info & Retrieval"):
            st.write("**Original question:**")
            st.code(user_q, language="text")
            st.write("**Query type:**")
            query_type = "Invoice Total Query" if is_invoice_total_query(user_q) else ("Document type query" if is_document_type_query(user_q) else ("Date-based query" if is_date_query(user_q) else "Semantic search"))
            st.code(query_type, language="text")
            st.write("**Processed query:**")
            st.code(standalone_q or "(empty)", language="text")
            st.write(f"**Retrieved {len(docs) if docs else 0} chunk(s) for display.**")
            st.write("**Token usage:** Limited to stay under Groq's 12k TPM limit")
            st.write("**Current files:**")
            for f in current_files:
                st.write(f"• {f}")

        with st.expander("📑 Sample Retrieved Chunks"):
            if docs:
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get('source_file', 'Unknown')
                    file_type = doc.metadata.get('file_type', 'unknown').upper()
                    
                    if file_type == 'PDF':
                        page = doc.metadata.get('page_num', doc.metadata.get('page', '?'))
                        page_info = f"(Page {page})"
                    elif file_type == 'EXCEL':
                        sheet = doc.metadata.get('sheet_name', 'Unknown')
                        row = doc.metadata.get('row_num', '')
                        doc_type = doc.metadata.get('document_type', '')
                        
                        if doc_type == 'invoice_total_only':
                            page_info = f"💰 TOTAL DOCUMENT"
                        elif doc.metadata.get('is_total_row'):
                            page_info = f"💰 TOTAL ROW (Sheet: {sheet}, Row {row})"
                        else:
                            row_info = f", Row {row}" if row else ""
                            page_info = f"(Sheet: {sheet}{row_info})"
                    else:
                        page_info = ""
                    
                    # Show total amount in metadata if available
                    total_amount = doc.metadata.get('total_amount')
                    total_str = f" - Amount: {total_amount}" if total_amount else ""
                    
                    # Try to extract and show date if available
                    date = extract_date_from_text(doc.page_content)
                    date_str = f" [{date.strftime('%m/%d/%Y')}]" if date else ""
                    
                    st.markdown(f"**{i}. {source} {page_info}{date_str}{total_str}**")
                    st.write(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
                    st.divider()
            else:
                st.write("No chunks retrieved.")