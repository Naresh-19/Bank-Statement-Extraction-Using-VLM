# table_cropper.py
import os
from pathlib import Path
from transformers import TableTransformerForObjectDetection, DetrImageProcessor
from PIL import Image
import torch
import fitz  # PyMuPDF
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*meta parameter.*")

def crop_with_padding(image, box, padding=10):
    x_min, y_min, x_max, y_max = map(int, box)
    width, height = image.size

    x_min_p = max(x_min - padding, 0)
    y_min_p = max(y_min - padding, 0)
    x_max_p = min(x_max + padding, width)
    y_max_p = min(y_max + padding, height)

    return image.crop((x_min_p, y_min_p, x_max_p, y_max_p))

def crop_tables_from_pdf(pdf_path, output_folder=None, model_name="apkonsta/table-transformer-detection-ifrs",
                         confidence_threshold=0.5, padding=10):
    pdf_name = Path(pdf_path).stem
    output_dir = Path(output_folder) if output_folder else Path(f"table/{pdf_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_processor = DetrImageProcessor.from_pretrained(model_name)
    model = TableTransformerForObjectDetection.from_pretrained(model_name)

    pdf_doc = fitz.open(pdf_path)
    cropped_image_paths = []

    for page_number in range(len(pdf_doc)):
        page = pdf_doc.load_page(page_number)
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        encoding = image_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encoding)

        target_sizes = torch.tensor([img.size[::-1]])
        results = image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=confidence_threshold
        )[0]

        for i, (box, score) in enumerate(zip(results["boxes"], results["scores"])):
            cropped_img = crop_with_padding(img, box.tolist(), padding=padding)
            save_path = output_dir / f"page{page_number+1}_table{i+1}.png"
            cropped_img.save(save_path)
            cropped_image_paths.append(str(save_path))

    pdf_doc.close()
    return cropped_image_paths
