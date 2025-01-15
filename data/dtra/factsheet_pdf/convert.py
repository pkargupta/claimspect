import os
import pdftotext

def pdf2str(pdf_file_path: str) -> str:
    """
    Convert a PDF file to a string.
    """
    # Load your PDF
    with open(pdf_file_path, "rb") as f:
        pdf = pdftotext.PDF(f)

    # Read all the text into one string
    result = " ".join(pdf)
    return result.replace("\n", " ").replace("\t", " ").replace("  ", " ")

def get_pdf_path_list(
    target_dir = "data/dtra/factsheet_pdf"
    ) -> list:
    """
    Get a list of PDF file paths.
    """
    pdf_path_list = []
    pdf_dir = target_dir
    for pdf_file in os.listdir(pdf_dir):
        if not pdf_file.endswith(".pdf"):
            continue
        pdf_path = os.path.join(pdf_dir, pdf_file)
        pdf_path_list.append(pdf_path)
    return pdf_path_list

def get_text_path(pdf_path: str) -> str:
    """
    Get a text file path from a PDF file path.
    """
    text_path = pdf_path.replace("pdf", "txt")
    return text_path
    
def save_text(text_path: str, text: str):
    """
    Save a string to a text file.
    """
    with open(text_path, "w") as f:
        f.write(text)

if __name__ == "__main__":
    
    pdf_path_list = get_pdf_path_list()
    
    text_list = []
    for pdf_path in pdf_path_list:
        text = pdf2str(pdf_path)
        text_list.append(text)
        
    for pdf_path, text in zip(pdf_path_list, text_list):
        text_path = get_text_path(pdf_path)
        save_text(text_path, text)