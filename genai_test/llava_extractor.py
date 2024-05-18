from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
import settings
import json
import pandas as pd


def process_image(image_name: str) -> Image:
    """
    Process image to be used as input for LLaVA model.
    """
    image = Image.open(image_name)
    image = image.resize((224, 224))
    image = image.convert("RGB")
    image.save(f"data/processed/{image_name}")
    return image


def initialise_model(model_name: str):
    """
    Initialise LLaVA model and processor.
    """
    processor = AutoProcessor.from_pretrained(settings.model_name)
    model = LlavaForConditionalGeneration.from_pretrained(settings.model_name)
    return processor, model


def get_prompt_template(prompt_id: str, graph: Image) -> str:
    """
    Get prompt template for LLaVA model.
    """
    static_context = """
        You are an expert at interpretting graphs. Your task is to analyse the provided image of a graph and return the raw data points.
        Follow the below instructions exactly:
        
        1. Inspect the graph carefully. 
        <graph>
        {graph}
        </graph> """

    static_format = """

        Output the data points in correct syntax JSON using the following schema: 
        {
            "x": [x1, x2, x3, ...],
            "y": [y1, y2, y3, ...],
            "x_units": "<units of x axis. If no units are clearly shown on the x-axis, this field should be "NULL">",
            "y_units": "<units of y axis. If no units are clearly shown on the y-axis, this field should be "NULL">",
        }

        Output:

    """
    prompt_library = {
        "1": f"""
            {static_context}

            2. Identify the type of graph, establishing the location of x-axis and y-axis.

            3. Identify scale of the axes and the units of measurement.

            4. Identify the data points on the graph. If the data points are not clearly visible, provide an estimate. If there are too many data points, provide a well-distributed sample of a maximum of 100 data points.
            
            {static_format}
            """,
        "2": f"""
            {static_context}

            2. Identify the data points on the graph. If the data points are not clearly visible, provide an estimate. If there are too many data points, provide a well-distributed sample of a maximum of 100 data points.
            
            {static_format}
            """,
    }
    return prompt_library[prompt_id]


def call_model(processor, model, prompt: str, image: Image):
    """
    Call LLaVA model.
    """

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generate_ids = model.generate(**inputs, max_new_tokens=15)
    outputs = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return outputs


def output_parser(str_input: str, keys_to_extract: list[str]) -> dict:
    """Parses JSON output from the LLM"""

    def _parse_keys_from_str(content: str, key: str) -> str:
        """Parses the string field from json with best effort using expected keys"""
        value = (
            content.split(f'"{key}"')[1]
            .split('", "')[0]
            .split('","')[0]
            .strip(":")
            .strip()
            .strip("}")
            .strip('"')
        )
        return value.encode().decode("unicode-escape")

    try:
        return json.loads(str_input)
    except Exception:
        try:
            return_dict = {}
            for key in keys_to_extract:
                return_dict[key] = _parse_keys_from_str(str_input, key)
            return return_dict
        except Exception:
            return_dict = {}
            for key in keys_to_extract:
                return_dict[key] = "N.A."
            return return_dict


def main(image_name: str, prompt_id: str):
    """
    Main function
    """
    image = process_image(image_name)
    processor, model = initialise_model(settings.model_name)
    prompt = get_prompt_template(prompt_id, image)
    outputs = call_model(processor, model, prompt, image)
    output_dict = output_parser(outputs, ["x", "y", "x_units", "y_units"])
    pd.DataFrame(output_dict).to_csv(f"data/output/{image_name.split('.')[0]}.csv")
    return outputs


if __name__ == "__main__":
    image_name = settings.image_name
    prompt_id = settings.prompt_id
    main(image_name, prompt_id)
