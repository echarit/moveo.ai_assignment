**<h1>How to Setup and Run</h1>**
<h4>
- Create a new python enviroment or go to an existing one using pyenv, venv, conda etc. <br>
- run pip install -r requirements.txt to install the necessary python libraries <br>
- setup ollama from this link depending on your OS https://ollama.com/download/ <br>
- Download the 8 billion and 4 billion models to reproduce the results in the reports folder with 'ollama run qwen3:4b' and  'ollama run qwen3:8b' <br>
- Open a terminal and change directory to the source *.py files <br>
- run python evaluate_rag.py or python3 evaluate_rag.py depending on your virtual enviroment <br>
</h4>

**<h1>Configuration file</h1>**
<h4>
- The script uses the toml python library. Inside the config folder there is a config.toml file where savepaths and parameter values like temperature and seed are stored. 
- Changing the values in the config file will result in the evaluation script running with the updated values.
- Due to formatting issue the LLM prompts are defined in the file constants.py and are imported from there by the evaluation script.

**<h1>Input and Output</h1>**
<h4>
- The input file is inside the 'data' directory
- Output is stored in the reports folder and inside subfolders named after the LLM model name used for the evaluation.
</h4>


**<h1>Dimension Schema Description:</h1>**
The evaluator adds the following 6 columns to evaluation dataset csv file:
- <h3> Retrieved Fragments Similarity: </h3> 
  The average vector similarity between the fragment texts in  Fragment Texts column and the Current User Question concatenated with the Conversation History column.
  The vector similarities are calculated using the embeddings of a Sentence Transformers models named "all-MiniLM-L6-v2". Ranges from {0-1} real valued.
- <h3> llm_precision_at_k: </h3>
  A metric that tries to simulate the precision at k metric from Information Retrieval. 
  It refers to the Fragment Texts columns as well. The evaluator LLM judges whethere a fragment text is relevant or not using the LLM_PRECISION_AT_K_PROMPT from constants.py  Ranges from {0-1} real valued.
- <h3> True_Positives: </h3>
  An auxilary column that calculates the number of relevant texts from the Fragment Texts column that were judged as such by the evaluator LLM. Ranges from {1-n} where n the number of Fragmented Texts.
- <h3> verbosity: </h3>
  A metric that quantifies how verbose the RAG response is. Ranges from the values {1/5, 2/5, 3/5, 4/5, 5/5} for less to much verbosity respectively.
- <h3> user_intent: </h3>
  An auxilary column that classifies the intent behind the Current User Question Column. Ranges from {Malignant, Benign}
- <h3> alignment: </h3>
  A binary values metric that assigns 1 (True) to RAG responses that respond accordingly to Benign user questions or dismissively to Malignant ones and 0 otherwise.
- <h3> faithfulness: </h3>
  A metric that quantifies how factually correct the RAG response is. Ranges from the values {1/5, 2/5, 3/5, 4/5, 5/5} for minimum to maximum faithfulness respectively.
  
  

**<h1>Aggregate Statistics:</h1>**
- Having defined the evaluation metric as above then aggregate statistics can be calculated as simple average across the rows of the metrics defined above. The only exception is the aggregate precision at K metric which is calculated in the classic micro average manner. The respective json file can be found in the reports folder in the model name subfolder under the name 'aggregate_metrics.json'


