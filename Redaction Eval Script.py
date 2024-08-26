# Databricks notebook source
!pip install transformers==4.42.4
!pip install -U bitsandbytes
!pip install ray
!pip install pytest


# COMMAND ----------

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import ast 

def merger(anonymized_list):
    return " ".join([trans['text'] for trans in anonymized_list])
transcriptions = pd.read_parquet(r"gpu_transcriptions_2024_07_31.parquet")
redacted = pd.read_parquet(r"gpu_transcriptions_redacted_2024_07_31.parquet")


transcriptions = pd.concat([transcriptions, redacted], axis=1)[['transcription', 'anonymized']]
transcriptions['redacted_transcription'] = transcriptions['anonymized'].apply(merger)
transcriptions['transcription'] = transcriptions['transcription'].apply(merger)


# COMMAND ----------

model_path = "internlm/internlm2_5-7b-chat"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32, 
    device_map="auto",
    trust_remote_code=True, 
    load_in_4bit=True  # Use 4-bit quantization
).eval()
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code=True
)

def eval_func(unredacted_transcription, redacted_transcription):
    evaluation_prompt = """
    Given the following **unredacted transcription** and **redacted transcription**, evaluate the quality of the PII (Personally Identifiable Information) redaction process and assign scores across various dimensions:
    
    Scoring Instructions:
    
    **PII Coverage**: Does the redacted transcription accurately identify and remove all necessary PII (e.g., names, phone numbers, addresses)? Ensure no sensitive information is left in the redacted version.
    Score from 1 to 10, where 1 means very poor coverage (many PII elements missed) and 10 means perfect coverage (no PII left).

    **Over-redaction**: Does the redacted transcription remove only the required PII elements? Over-redaction should be avoided (removing information that is not PII). 
    Score from 1 to 10, where 1 means excessive over-redaction (important non-PII information removed), and 10 means no over-redaction.

    **Redaction Consistency**: Is the PII redacted consistently throughout the transcription? Ensure similar PII (e.g., the same name) is redacted uniformly across the entire transcript.
    Score from 1 to 10, where 1 means highly inconsistent (PII elements not redacted consistently), and 10 means highly consistent.

    **Legibility Post-redaction**: Does the redacted transcription remain understandable and retain its overall meaning? The redaction process should not make the transcription unreadable.
    Score from 1 to 10, where 1 means poor legibility (hard to understand) and 10 means excellent legibility (easy to understand despite redactions).

    **Redaction Speed**: Based on an assumed automated process, evaluate the efficiency and speed of the redaction process. This is based on overall execution and expected runtime.
    Score from 1 to 10, where 1 means very slow or inefficient, and 10 means fast and highly efficient.

    **Overall Redaction Quality**: Provide a holistic score that considers all the previous factors (coverage, over-redaction, consistency, legibility, speed).
    Score from 1 to 10, where 1 means poor overall redaction quality, and 10 means excellent overall redaction quality.

    The unredacted and redacted transcriptions to be evaluated are as follows:
    Unredacted Transcription:
    {unredacted_transcription}

    Redacted Transcription:
    {redacted_transcription}

    Required Output Format:
    {{
    "pii_coverage": {{
        "score": <score>
    }},
    "over_redaction": {{
        "score": <score>
    }},
    "consistency": {{
        "score": <score>
    }},
    "legibility_post_redaction": {{
        "score": <score>
    }},
    "redaction_speed": {{
        "score": <score>
    }},
    "overall_redaction_quality": {{
        "score": <score>
    }}
    }}

    Example Output:
    {{
    "pii_coverage": {{
        "score": 9
    }},
    "over_redaction": {{
        "score": 8
    }},
    "consistency": {{
        "score": 10
    }},
    "legibility_post_redaction": {{
        "score": 7
    }},
    "redaction_speed": {{
        "score": 9
    }},
    "overall_redaction_quality": {{
        "score": 9
    }}
    }}

    Only return the scoring json and no additional words in the output.

    Scoring Dictionary:
    """.format(unredacted_transcription=unredacted_transcription, redacted_transcription=redacted_transcription)

    # Call the model to generate the evaluation
    response, history = model.chat(tokenizer, evaluation_prompt)
    print(response)
    try:
        output = ast.literal_eval(response)  # Convert the response to a dictionary
        return output
    except:
        return {
            "pii_coverage": {
                "score": -1
            },
            "over_redaction": {
                "score": -1
            },
            "consistency": {
                "score": -1
            },
            "legibility_post_redaction": {
                "score": -1
            },
            "redaction_speed": {
                "score": -1
            },
            "overall_redaction_quality": {
                "score": -1
            }
        }

# Applying the eval_func to each row in the DataFrame and expanding the dictionary into separate columns
transcriptions = transcriptions.iloc[:20, :]
transcriptions['eval'] = transcriptions.apply(lambda row: eval_func(row['transcription'], row['redacted_transcription']), axis=1)

# Expanding the 'eval' dictionary into separate columns
eval_df = transcriptions['eval'].apply(pd.Series)

# Combining the evaluation scores with the original DataFrame
transcriptions = pd.concat([transcriptions, eval_df], axis=1)

# Extracting scores into individual columns
transcriptions[['pii_coverage', 'over_redaction', 'consistency', 'legibility_post_redaction', 'redaction_speed', 'overall_redaction_quality']] = transcriptions[['pii_coverage', 'over_redaction', 'consistency', 'legibility_post_redaction', 'redaction_speed', 'overall_redaction_quality']].applymap(lambda x: x['score'] if isinstance(x, dict) else x)

