# Databricks notebook source
!pip install transformers==4.42.4
!pip install -U bitsandbytes
!pip install ray
!pip install pytest


# COMMAND ----------

import pandas as pd

summaries = pd.read_parquet("gpu_transcriptions_redacted_summary_benchmarking.parquet")
summaries


# COMMAND ----------

# MAGIC %md
# MAGIC Phi-3-small-8k-instruct Scoring

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline 
import torch
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.random.manual_seed(0)
model_id = "microsoft/Phi-3-small-8k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
assert torch.cuda.is_available(), "This model needs a GPU to run ..."
device = torch.cuda.current_device()
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)

def eval_func(transcription, summary):
    evaluation_prompt = """
    ## Prompt for Evaluating Summary of Call Transcription:

    ---

    Given the following **call transcription** and **summary**, score the summary across various dimensions based on the provided transcription:

    Call Transcription:
    {transcription}

    Summary:
    {summary}

    Scoring Instructions:
    Accuracy: Does the summary capture all the key points and critical details of the call transcription? Ensure no false or misleading information is presented.

    Score from 1 to 10, where 1 means highly inaccurate, and 10 means the summary is completely accurate.
    Justify the score by comparing important points from the transcription with those in the summary.
    Relevance: Does the summary focus on the most relevant and essential parts of the conversation? Irrelevant or minor details should be omitted.

    Score from 1 to 10, where 1 means irrelevant or off-topic, and 10 means only the most relevant parts are captured.
    Provide reasoning behind why certain parts of the transcription are relevant or irrelevant to the summary.
    Conciseness: Is the summary concise while still conveying the necessary information? Avoid unnecessary details or overly verbose sentences.

    Score from 1 to 10, where 1 means the summary is too verbose, and 10 means it's perfectly concise.
    Explain how conciseness affects the readability and comprehension of the summary.
    Sentiment Alignment: Does the sentiment of the summary match the tone and emotional cues from the original call transcription? Ensure that positive, neutral, or negative sentiments are correctly represented.

    Score from 1 to 10, where 1 means the sentiment is misaligned, and 10 means the sentiment is perfectly aligned.
    Explain any misalignment or issues with how emotions or tones were interpreted.
    Clarity: Is the summary easy to read and understand? Ensure that it is free from grammatical errors, unclear language, or awkward phrasing.

    Score from 1 to 10, where 1 means the summary is difficult to understand, and 10 means the summary is exceptionally clear.
    Justify the clarity score by highlighting any confusing parts of the summary or praising good language usage.
    Overall Quality: Provide a holistic score that considers all the previous factors (accuracy, relevance, conciseness, sentiment, clarity).

    Score from 1 to 10, where 1 means poor overall quality, and 10 means excellent overall quality.

    Output Format:
    Return the score for each parameter in the following standardized JSON format:

    {{
    "accuracy": {{
        "score": <score>
    }},
    "relevance": {{
        "score": <score>
    }},
    "conciseness": {{
        "score": <score>
    }},
    "sentiment_alignment": {{
        "score": <score>
    }},
    "clarity": {{
        "score": <score>
    }},
    "overall_quality": {{
        "score": <score>
    }}
    }}

    Example Output:
    {{
    "accuracy": {{
        "score": 9
    }},
    "relevance": {{
        "score": 8
    }},
    "conciseness": {{
        "score": 10
    }},
    "sentiment_alignment": {{
        "score": 7
    }},
    "clarity": {{
        "score": 9
    }},
    "overall_quality": {{
        "score": 8
    }}
    }}
    """.format(transcription=transcription, summary=summary)

    messages = [{"role": "user", "content": evaluation_prompt}]
    generation_args = {
        "max_new_tokens": 300,
        "return_full_text": False,
        "temperature": 0.5,
        "do_sample": True,
    }
    output = pipe(messages, **generation_args)
    output_text = output[0]['generated_text']
    try:
        output = json.loads(output_text)  
        return output
    except:
        return {
            "accuracy": {
                "score": -1
            },
            "relevance": {
                "score": -1
            },
            "conciseness": {
                "score": -1
            },
            "sentiment_alignment": {
                "score": -1
            },
            "clarity": {
                "score": -1
            },
            "overall_quality": {
                "score": -1
            }
        }

# Applying the eval_func to each row in the DataFrame and expanding the dictionary into separate columns
summaries = summaries.iloc[:20, :]
summaries['eval'] = summaries.apply(lambda row: eval_func(row['transcription'], row['summary']), axis=1)

# Expanding the 'eval' dictionary into separate columns
eval_df = summaries['eval'].apply(pd.Series)

# Combining the evaluation scores with the original DataFrame
summaries = pd.concat([summaries, eval_df], axis=1)

summaries

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC InterLm Scoring

# COMMAND ----------

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import ast 

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

def eval_func(transcription, summary):
    evaluation_prompt = """
    Given the following **call transcription** and **summary**, score the summary across various dimensions based on the provided transcription:
    
    Scoring Instructions:
    Accuracy: Does the summary capture all the key points and critical details of the call transcription? Ensure no false or misleading information is presented.
    Score from 1 to 10, where 1 means highly inaccurate, and 10 means the summary is completely accurate.

    Relevance: Does the summary focus on the most relevant and essential parts of the conversation? Irrelevant or minor details should be omitted.
    Score from 1 to 10, where 1 means irrelevant or off-topic, and 10 means only the most relevant parts are captured.

    Conciseness: Is the summary concise while still conveying the necessary information? Avoid unnecessary details or overly verbose sentences.
    Score from 1 to 10, where 1 means the summary is too verbose, and 10 means it's perfectly concise.

    Sentiment Alignment: Does the sentiment of the summary match the tone and emotional cues from the original call transcription? Ensure that positive, neutral, or negative sentiments are correctly represented.
    Score from 1 to 10, where 1 means the sentiment is misaligned, and 10 means the sentiment is perfectly aligned.
    
    Clarity: Is the summary easy to read and understand? Ensure that it is free from grammatical errors, unclear language, or awkward phrasing.
    Score from 1 to 10, where 1 means the summary is difficult to understand, and 10 means the summary is exceptionally clear.
    
    Overall Quality: Provide a holistic score that considers all the previous factors (accuracy, relevance, conciseness, sentiment, clarity).
    Score from 1 to 10, where 1 means poor overall quality, and 10 means excellent overall quality.

    The call transcription ad summary that needs to be evaluated are as follows:
    Call Transcription:
    {transcription}

    Summary:
    {summary}

    Required Output Format:
    {{
    "accuracy": {{
        "score": <score>
    }},
    "relevance": {{
        "score": <score>
    }},
    "conciseness": {{
        "score": <score>
    }},
    "sentiment_alignment": {{
        "score": <score>
    }},
    "clarity": {{
        "score": <score>
    }},
    "overall_quality": {{
        "score": <score>
    }}
    }}

    Example Output:
    {{
    "accuracy": {{
        "score": 9
    }},
    "relevance": {{
        "score": 8
    }},
    "conciseness": {{
        "score": 10
    }},
    "sentiment_alignment": {{
        "score": 7
    }},
    "clarity": {{
        "score": 9
    }},
    "overall_quality": {{
        "score": 8
    }}
    }}

    Only return the scoring json and no additional words in the output.

    Scoring Dictionary: 
    """.format(transcription=transcription, summary=summary)

    # Hello! How can I help you today?
    response, history = model.chat(tokenizer, evaluation_prompt)
    print(response)
    try:
        output = ast.literal_eval(response)  
        return output
    except:
        return {
            "accuracy": {
                "score": -1
            },
            "relevance": {
                "score": -1
            },
            "conciseness": {
                "score": -1
            },
            "sentiment_alignment": {
                "score": -1
            },
            "clarity": {
                "score": -1
            },
            "overall_quality": {
                "score": -1
            }
        }

# Applying the eval_func to each row in the DataFrame and expanding the dictionary into separate columns
summaries = summaries.iloc[:20, :]
summaries['eval'] = summaries.apply(lambda row: eval_func(row['transcription'], row['summary']), axis=1)

# Expanding the 'eval' dictionary into separate columns
eval_df = summaries['eval'].apply(pd.Series)

# Combining the evaluation scores with the original DataFrame
summaries = pd.concat([summaries, eval_df], axis=1)

summaries[['accuracy', 'relevance', 'conciseness', 'sentiment_alignment', 'clarity', 'overall_quality']] = summaries[['accuracy', 'relevance', 'conciseness', 'sentiment_alignment', 'clarity', 'overall_quality']].applymap(lambda x: x['score'] if isinstance(x, dict) else x)

# COMMAND ----------



summaries.head(5)


# COMMAND ----------

summaries['accuracy'].describe()

# COMMAND ----------

summaries['relevance'].describe()

# COMMAND ----------

summaries['conciseness'].describe()

# COMMAND ----------

summaries['sentiment_alignment'].describe()

# COMMAND ----------

summaries['clarity'].describe()

# COMMAND ----------

summaries['overall_quality'].describe()

# COMMAND ----------


