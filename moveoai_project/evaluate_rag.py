import tomli as tomllib
import argparse
from custom_rag_evaluator import CustomRAGEvaluator
import os


def load_config(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG evaluation with custom config.")
    parser.add_argument(
        "--config", type=str, default=os.path.join("config", "config.toml"), help="Path to the TOML configuration file."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    custom_rag_evaluator = CustomRAGEvaluator(
        config["path_to_file"],
        config["llm_parameters"]["llm_evaluator_name"],
        config["llm_parameters"]["llm_temperature"],
        config["llm_parameters"]["seed"],
        config["similarity_model"],
    )

    custom_rag_evaluator.preprocess_prompts()
    custom_rag_evaluator.preprocess_evaluation_dataset()
    custom_rag_evaluator.calculate_retrieved_similarity()
    custom_rag_evaluator.calculate_precision_at_k()
    custom_rag_evaluator.calculate_verbosity()
    custom_rag_evaluator.calculate_user_intent()
    custom_rag_evaluator.calculate_alignment()
    custom_rag_evaluator.calculate_faithfulness()
    custom_rag_evaluator.calculate_aggregate_statistics()
    custom_rag_evaluator.save_stats()


if __name__ == "__main__":
    main()
