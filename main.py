from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub

#import os

#os.environ['HUGGINGFACEHUB_API_TOKEN']="YOUR_HUGGINGFACE_API_TOKEN"

PDF_BANK_1 ="PDF_LOCATION.pdf"
PDF_BANK_2 = "PDF_LOCATION.pdf"

#MODEL_NAME = "hkunlp/instructor-large" # Longer runtime same result as instructor-base
MODEL_NAME = "hkunlp/instructor-base"
#MODEL_NAME = "hkunlp/instructor-xl" - Gives more accurate response but model is too big for memory storage and PC crashes (OUT OF MEMORY)

'''
Returns all data in the pdf file.
'''
def getPdfText(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        # Replace bullet points with newlines for better accuracy
        text += page.extract_text().replace("•", "\n")

    return text

'''
 Uses the raw text from the pdfs and split it into text chunks.
 Returns the text chunks created.
'''
def getTextChunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1024, # 1000
        chunk_overlap = 200, # 200
        length_function = len)
    chunks = text_splitter.split_text(text)

    return chunks

'''
 Using the text_chunks created from getTextChunks() to embed the text_chunks and create a vectorestore from it.
 Returns the vectorestore created.
'''
def getVectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name = MODEL_NAME)
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)

    return vectorstore

'''
 Using the vectorestore created, to return a conversation chain from the model.
 Returns the Conversation Chain created.
'''
def getConversationChain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512}) 
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory)

    return conversation_chain

'''
 Using the conversation_chain returns and creates response to the user.
 Returns the response back to the user.
'''
def handleUserInput(user_question, conversation_chain, bank_name):
    response = conversation_chain({'question': user_question})
    chat_history = response['chat_history']

    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            print(f"User: {message.content}")
        else:
            print(f"Bot: \n{bank_name}:\n {message.content}")


'''
 All methods that will run to return a response to the user
 - raw_text: The full text from the pdfs
 - text_chunks: Uses the raw_text and return chunks of that data (1024 per chunk)
 - vectorstore: Turns the text_chunks into embeddings and saves it into a vectorstore using HuggingFaceInstrctEmbeddings()
 - conversation_chain: Used with the vectorestore to look up relevant documents from the retriever, and passes those documents and the question to a question-answering chain to return a response.
'''
def query_bank(bank_name, question):
    raw_text = getPdfText(PDF_BANK_1 if bank_name == "Bank 1" else PDF_BANK_2)
    text_chunks = getTextChunks(raw_text)
    vectorstore = getVectorstore(text_chunks)
    conversation_chain = getConversationChain(vectorstore)

    handleUserInput(question, conversation_chain, bank_name)


def main():
    load_dotenv()
    print("How can I help you compare these documents?")
    query_both = input("Would you like to query both documents? ").strip().lower()

    if query_both in ["y", "yes"]:
        # Query both Banks
        question = input("What would you like to query? ").strip()
        
        query_bank("Bank 1", question)
        query_bank("Bank 2", question)

    else:
        query_bank_name = input("Which bank would you like to query? ").strip().title()
        bank_name = query_bank_name
        if query_bank_name == "Bank 1":
            # Query Bank 1
            question = input("Type your question here... ").strip()
            query_bank(bank_name, question)

        elif query_bank_name == "Bank 2":
            # Query Bank 2
            question = input("Type your question here... ").strip()
            query_bank(bank_name, question)

        else:
            print("You did not select a bank to query")

if __name__ == "__main__":
    main()