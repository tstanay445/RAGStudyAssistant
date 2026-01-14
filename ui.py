import gradio as gr
import requests
import os
import shutil
import tempfile
import uuid
SESSION_ID = str(uuid.uuid4())
API_BASE = "http://localhost:8000"
ASK_URL = f"{API_BASE}/ask"
INGEST_URL = f"{API_BASE}/ingest"

# --------- Chat handler ----------
def chat_fn(message, history):
    r = requests.post(ASK_URL, json={
        "question": message,
        "session_id": SESSION_ID
    })
    answer = r.json()["answer"]

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer}
    ]
    return history, ""


# --------- File upload -> ingest ----------
def upload_and_ingest(files):
    if not files:
        return "No files selected."

    multipart = []
    for f in files:
        multipart.append(("files", open(f.name, "rb")))

    r = requests.post(INGEST_URL, files=multipart, timeout=600)
    return "Ingestion complete."



# --------- UI ----------
with gr.Blocks(css=".chat-container {height: 80vh;}") as demo:
    gr.Markdown("# 📚 RAG Study Assistant")

    state = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(elem_classes="chat-container")
            msg = gr.Textbox(placeholder="Ask a question…", show_label=False)
            send = gr.Button("Send")

        with gr.Column(scale=1):
            gr.Markdown("### Upload documents")
            file_box = gr.File(file_types=[".txt"], file_count="multiple")
            ingest_btn = gr.Button("Ingest")
            ingest_status = gr.Textbox(label="Status")

    send.click(chat_fn, inputs=[msg, state], outputs=[chatbot, msg])
    msg.submit(chat_fn, inputs=[msg, state], outputs=[chatbot, msg])
    ingest_btn.click(upload_and_ingest, inputs=[file_box], outputs=[ingest_status])

demo.launch()

