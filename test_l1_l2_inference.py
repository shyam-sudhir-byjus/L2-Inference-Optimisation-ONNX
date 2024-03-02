from level1_inference import solve as compute_inference
from inference_onnx_v4 import ONNXInferenceEngine

inference_engine_onnx = ONNXInferenceEngine()
from inference_engine import InferenceEngine

inference_engine = InferenceEngine()
import requests
import json
import pandas as pd
from common import URL, QNA_AUTH


def staging_qna_api_call(query: str) -> list[dict]:
    url = URL
    headers = {"Authorization": QNA_AUTH}
    data = {"text": query}
    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        json_response = response.json()
        return json_response["questions"]

    return []


def get_data_from_csv():
    df = pd.read_csv("New Dedup 400 DoC Eval - Sheet2.csv")
    questions_list = df["question"].tolist()
    return questions_list


if __name__ == "__main__":
    questions_list = get_data_from_csv()
    qna_data = {}

    for question in questions_list:
        qna_data[question] = {"staging_qna": staging_qna_api_call(question)}

    for k, v in qna_data.items():
        try:
            level1_inference = compute_inference(
                inference_engine=inference_engine,
                question_text=k,
                suggested_questions=v["staging_qna"],
            )
            level2_inference = compute_inference(
                inference_engine=inference_engine_onnx,
                question_text=k,
                suggested_questions=v["staging_qna"],
            )

            qna_data[k]["level1_inference"] = level1_inference
            qna_data[k]["no_of_ques_l1"] = len(level1_inference)
            qna_data[k]["level2_inference"] = level2_inference
            qna_data[k]["no_of_ques_l2"] = len(level2_inference)
            qna_data[k]["l1-l2-difference"] = list(
                set(level1_inference) - set(level2_inference)
            )
            del qna_data[k]["staging_qna"]
        except:
            continue

    with open("sample_questions_level1_level2_comparison.json", "w") as f:
        json.dump(qna_data, f, indent=2)
