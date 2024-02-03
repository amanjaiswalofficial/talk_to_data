# talk_to_data
Custom repository containing code for using LLama model, in order to chat with your own set of documents (PDFs or Markdown files)

#### Getting started
1. Download LLama model from [here](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q2_K.bin)
2. Save model in `models/llama-2-7b-chat.ggmlv3.q2_K.bin`
3. Setup virtual environment (Python 3.11)
4. Activate virtual environment
5. Run `pip install -r requirements.txt`
6. Copy required markdown/PDF files in `data/`
7. Update value for mode at `create_db: line 8`, `pdf` for PDF files and `md` for Markdown.
8. Run `python create_db.py`
9. Run `python main.py`
10. Interact with the model trained on your data. (Ctrl + C to exit) 