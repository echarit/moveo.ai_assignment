import pandas as pd
import torch
import json
import os
import warnings

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from ollama import chat
from ollama import ChatResponse

from constants import (
    LLM_PRECISION_AT_K_PROMPT,
    VERBOSITY_PROMPT,
    USER_INTENT_PROMPT,
    ALIGNMENT_PROMPT,
    FAITHFULNESS_PROMPT,
)


class CustomRAGEvaluator:
    """
    A custom evaluator for Retrieval-Augmented Generation (RAG) systems using LLMs and sentence similarity.
    Supports multiple evaluation dimensions such as precision, verbosity, intent detection, alignment, and faithfulness.
    """

    def __init__(self, path_to_file, llm_evaluator_name, llm_temperature, seed, similarity_model):
        """
        Initializes the evaluator.

        Args:
            path_to_file (str): Path to the evaluation dataset (CSV).
            llm_evaluator_name (str): LLM model name for evaluation.
            llm_temperature (float): Sampling temperature for LLM responses.
            seed (int): Random seed for reproducibility.
            similarity_model (str): SentenceTransformer model name for embedding generation.
        """
        self.evaulation_dataset = pd.read_csv(path_to_file)
        self.graded_dataset = pd.read_csv(path_to_file)
        self.model_name = llm_evaluator_name
        self.llm_temperature = llm_temperature
        self.random_seed = seed
        self.dataset_headers = list(self.evaulation_dataset)
        self.similarity_model = SentenceTransformer(similarity_model)
        self.save_path = os.path.join(os.getcwd(), "reports", self.model_name.replace(":", "_"))
        os.makedirs(self.save_path, exist_ok=True)

    def preprocess_prompts(self):
        """
        Injects dataset-specific header names into pre-defined evaluation prompts.
        This step adapts the prompts to the dataset being evaluated.
        """
        LLM_PRECISION_AT_K_PROMPT["content"] = LLM_PRECISION_AT_K_PROMPT["content"].replace(
            "*QUERY PLACEHOLDER*", self.dataset_headers[0]
        ).replace("*CONTENT PLACEHOLDER*", self.dataset_headers[2])

        VERBOSITY_PROMPT["content"] = VERBOSITY_PROMPT["content"].replace(
            "*QUERY PLACEHOLDER*", self.dataset_headers[0]
        ).replace("*CONTENT PLACEHOLDER*", self.dataset_headers[3])

        USER_INTENT_PROMPT["content"] = USER_INTENT_PROMPT["content"].replace(
            "*CONTENT PLACEHOLDER*", self.dataset_headers[0]
        )

        ALIGNMENT_PROMPT["content"] = ALIGNMENT_PROMPT["content"].replace(
            "*QUERY PLACEHOLDER*", self.dataset_headers[0]
        ).replace("*CONTENT PLACEHOLDER*", self.dataset_headers[3])

        FAITHFULNESS_PROMPT["content"] = FAITHFULNESS_PROMPT["content"].replace(
            "*QUERY PLACEHOLDER*", self.dataset_headers[0]
        ).replace("*CONTENT PLACEHOLDER*", self.dataset_headers[3])

        self.system_prompts = {
            "llm_precision_at_k": LLM_PRECISION_AT_K_PROMPT,
            "verbosity": VERBOSITY_PROMPT,
            "user_intent": USER_INTENT_PROMPT,
            "alignment": ALIGNMENT_PROMPT,
            "faithfulness": FAITHFULNESS_PROMPT,
        }

    def preprocess_evaluation_dataset(self):
        """
        Cleans and transforms the dataset for processing.
        Each row is converted into a dict of split fields for fragment-level access.
        """
        self.evaulation_dataset.fillna("", inplace=True)

        dataset = {}
        for i, row in self.evaulation_dataset.iterrows():
            dataset[i] = {header: row[header].split("\\n") for header in self.dataset_headers}
        self.evaulation_dataset = dataset

    def _preprocess_llm_response(self, response):
        """
        Extracts JSON content from the LLM response.

        Args:
            response (dict): Raw response from the LLM.

        Returns:
            dict: Parsed response content.
        """
        response = response["message"]["content"]
        response = response.split("</think>\n\n")[-1] if response.startswith("<think>") else response
        return json.loads(response)

    def _llm_call(self, content, metric_name):
        """
        Sends content to the LLM for evaluation.

        Args:
            content (str): The text content to evaluate.
            metric_name (str): The evaluation metric to apply.

        Returns:
            dict: LLM's response in structured format.
        """
        response: ChatResponse = chat(
            model=self.model_name,
            messages=[
                self.system_prompts[metric_name],
                {"role": "user", "content": content},
            ],
            format="json",
            options={"temperature": self.llm_temperature, "seed": self.random_seed},
        )
        return self._preprocess_llm_response(response)

    def calculate_retrieved_similarity(self, metric_name="Retrieved Fragments Similarity"):
        """
        Calculates the cosine similarity between query and retrieved text fragments.
        """
        self.graded_dataset[metric_name] = 0
        self.average_similirities = {}

        warnings.simplefilter(action="ignore", category=FutureWarning)

        for data_point in tqdm(self.evaulation_dataset, desc=f"Calculating {metric_name}"):
            conv_history = "\n".join(self.evaulation_dataset[data_point]["Conversation History"])
            question = self.evaulation_dataset[data_point]["Current User Question"]
            retrieved_texts = self.evaulation_dataset[data_point]["Fragment Texts"]

            query = f"{conv_history}\n{question[0]}"
            query_embeddings = self.similarity_model.encode(query)
            retrieved_embeddings = self.similarity_model.encode(retrieved_texts)

            similarities = self.similarity_model.similarity(
                query_embeddings + retrieved_embeddings, query_embeddings + retrieved_embeddings
            )[0, 1:]

            score = float(torch.mean(similarities)) if len(similarities) > 0 else 0
            self.average_similirities[data_point] = score
            self.graded_dataset.loc[data_point, metric_name] = score

    def calculate_precision_at_k(self, metric_name="llm_precision_at_k"):
        """
        Evaluates how many of the top-k retrieved fragments are relevant using the LLM.
        """
        self.graded_dataset[metric_name] = 0
        self.graded_dataset["True_Positives"] = 0
        self.hits_at_k_responses = {}

        for data_point in tqdm(self.evaulation_dataset, desc=f"Calculating {metric_name} metric: "):
            content = f"{self.dataset_headers[2]}: {'\n'.join(self.evaulation_dataset[data_point][self.dataset_headers[2]])}"
            response = self._llm_call(content, metric_name)
            self.hits_at_k_responses[data_point] = response

            true_positives = sum(response[fragment]["answer"].startswith("YES") for fragment in response)
            self.graded_dataset.loc[data_point, metric_name] = true_positives / len(response)
            self.graded_dataset.loc[data_point, "True_Positives"] = true_positives

        self._save_matrics_json(self.hits_at_k_responses, f"{metric_name}_metrics.json")

    def calculate_verbosity(self, metric_name="verbosity"):
        """
        Evaluates verbosity (wordiness) of the response.
        """
        self.graded_dataset[metric_name] = 0
        self.verbosity_responses = {}

        for data_point in tqdm(self.evaulation_dataset, desc=f"Calculating {metric_name} metric"):
            content = f"{self.dataset_headers[3]}: {'\n'.join(self.evaulation_dataset[data_point][self.dataset_headers[3]])}"
            response = self._llm_call(content, metric_name)
            self.verbosity_responses[data_point] = response

            self.graded_dataset.loc[data_point, metric_name] = 1 / int(response["score"])

        self._save_matrics_json(self.verbosity_responses, f"{metric_name}_metrics.json")

    def calculate_user_intent(self, metric_name="user_intent"):
        """
        Classifies whether user intent is benign or malignant.
        """
        self.graded_dataset[metric_name] = 0
        self.user_intent_responses = {}

        for data_point in tqdm(self.evaulation_dataset, desc=f"Calculating {metric_name} metric"):
            content = f"{self.dataset_headers[0]}: {'\n'.join(self.evaulation_dataset[data_point][self.dataset_headers[0]])}"
            response = self._llm_call(content, metric_name)
            self.user_intent_responses[data_point] = response

            self.graded_dataset.loc[data_point, metric_name] = response["intent"]

        self._save_matrics_json(self.user_intent_responses, f"{metric_name}_metrics.json")

    def calculate_alignment(self, metric_name="alignment"):
        """
        Checks whether the RAG response aligns with user intent appropriately.
        """
        self.graded_dataset[metric_name] = 0
        self.alignment_responses = {}

        for data_point in tqdm(self.evaulation_dataset, desc=f"Calculating {metric_name} metric"):
            content = (
                f"{self.dataset_headers[0]}: {self.evaulation_dataset[data_point][self.dataset_headers[0]][0]}\n"
                f"User Intent: {self.user_intent_responses[data_point]['intent']}\n\n"
                f"{self.dataset_headers[3]}:\n"
                f"{'\n'.join(self.evaulation_dataset[data_point][self.dataset_headers[3]])}\n"
            )

            response = self._llm_call(content, metric_name)
            self.alignment_responses[data_point] = response
            self.graded_dataset.loc[data_point, metric_name] = 0 if response["aligned"] == "FALSE" else 1

        self._save_matrics_json(self.alignment_responses, f"{metric_name}_metrics.json")

    def calculate_faithfulness(self, metric_name="faithfulness"):
        """
        Scores factual consistency of the RAG response.
        """
        self.graded_dataset[metric_name] = 0
        self.faithfulness_responses = {}
        max_score = 5

        for data_point in tqdm(self.evaulation_dataset, desc=f"Calculating {metric_name} metric"):
            content = (
                f"{self.dataset_headers[0]}: {self.evaulation_dataset[data_point][self.dataset_headers[0]][0]}\n"
                f"User Intent: {self.user_intent_responses[data_point]['intent']}\n\n"
                f"{self.dataset_headers[3]}:\n"
                f"{'\n'.join(self.evaulation_dataset[data_point][self.dataset_headers[3]])}\n"
            )

            response = self._llm_call(content, metric_name)
            self.faithfulness_responses[data_point] = response
            self.graded_dataset.loc[data_point, metric_name] = int(response["score"]) / max_score

        self._save_matrics_json(self.faithfulness_responses, f"{metric_name}_metrics.json")

    def calculate_aggregate_statistics(self):
        """
        Computes summary statistics over all metrics and saves them to disk.
        """
        self.aggregate_statistics = {
            "average_similarity": round(self.graded_dataset["Retrieved Fragments Similarity"].mean(), 3),
            "average_precision_at_k": round(
                int(self.graded_dataset["True_Positives"].sum())
                / sum(len(entry["Fragment Texts"]) for entry in self.evaulation_dataset.values()), 3),
            "average_verbosity": round(self.graded_dataset["verbosity"].mean(), 3),
            "alignment_percentage": round(
                self.graded_dataset["alignment"].sum() / len(self.graded_dataset["alignment"]), 3),
            "faithfulness": round(self.graded_dataset["faithfulness"].mean(), 3),
        }

        self._save_matrics_json(self.aggregate_statistics, "aggregate_metrics.json")

    def _save_matrics_json(self, metrics_dict, filename):
        """
        Saves a dictionary of metrics to a JSON file.

        Args:
            metrics_dict (dict): Dictionary of metrics.
            filename (str): Output file name.
        """
        filepath = os.path.join(self.save_path, filename)
        with open(filepath, "w") as f:
            json.dump(metrics_dict, f)

    def save_stats(self, savename="evaluated_dataset.tsv"):
        """
        Saves the final scored dataset to a TSV file.

        Args:
            savename (str): File name for the saved dataset.
        """

        full_path = os.path.join(self.save_path, savename)
        self.graded_dataset.to_csv(full_path, sep="\t", index=False)
