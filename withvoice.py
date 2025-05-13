from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline
import assemblyai as aai
import torch
import speech_recognition as sr

# Setup embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
aai.settings.api_key = "9264da470e454f138f482fa2fc92baef"

# AssemblyAI transcription setup
transcriber = aai.Transcriber()

# Transcribe your audio file
transcript = transcriber.transcribe("sam.wav")

if not transcript.text.strip():
    print("Sorry, I couldn't extract any text from the audio. Please check the file.")
    exit()

text = transcript.text
print("🎧 Transcription loaded successfully! I’m ready to answer your questions.\n")

# Split text for semantic search
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = splitter.split_text(text)

# Build vectorstore
vectorstore = FAISS.from_texts(texts, embedding=embeddings)

# Load QA model
device = 0 if torch.cuda.is_available() else -1
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=device)

# Define chatbot response function
def ask_question(question):
    try:
        docs = vectorstore.similarity_search(question, k=1)
        context = docs[0].page_content
        result = qa_pipeline(question=question, context=context)
        return result["answer"]
    except Exception as e:
        return f"⚠️ Oops! I couldn’t process that. Error: {str(e)}"

# Setup speech recognizer
recognizer = sr.Recognizer()
print("💬 ChatBot: Hi! Ask me anything about the conversation from the audio. Say 'exit' to leave.\n")
print("🎙️ Listening...")

with sr.Microphone() as source:
    recognizer.adjust_for_ambient_noise(source)  # reduce noise
    while True:
        try:
            print("\n🔴 Speak now...")
            audio = recognizer.listen(source)

            user_input = recognizer.recognize_google(audio).lower()
            print(f"📝 You said: {user_input}")

            if "exit" in user_input:
                print("👋 Exit command detected. Goodbye!")
                break

            response = ask_question(user_input)
            print(f"💬 ChatBot: {response}\n")

        except sr.UnknownValueError:
            print("😕 Didn't catch that. Please try again.")
        except sr.RequestError as e:
            print(f"⚠️ Could not request results; {e}")
        except KeyboardInterrupt:
            print("🛑 Stopped by user.")
            break
