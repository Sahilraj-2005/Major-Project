import os
import re
import json
import warnings
import pandas as pd
from datasets import Dataset

# Suppress warnings to keep terminal clean
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend import RAGBackend

def parse_judge_output(text):
    """Safely extracts JSON from the LLM judge's output."""
    text = re.sub(r'```json|```', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"clarity_score": 0, "practicality_score": 0, "reasoning": "Failed to parse judge output."}

def run_evaluation():
    print("Initializing Backend for Evaluation...")
    backend = RAGBackend()
    
    if not backend.is_db_populated():
        print("Error: Pinecone database is empty. Please upload documents via the frontend first.")
        return

    # Wrap for RAGAS compatibility
    ragas_llm = LangchainLLMWrapper(backend.llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(backend.embedding_func)

    # --- CUSTOM LLM-AS-A-JUDGE PROMPT ---
    judge_prompt = PromptTemplate(
        template="""You are an expert evaluator grading an educational explanation about mining laws.
        Grade the following explanation based on two metrics on a scale of 1 to 5.
        
        1. Clarity (1-5): Is it easy to understand, well-structured, and free of confusing jargon?
        2. Practicality (1-5): Does it provide a realistic, helpful example or context for a mining engineer?
        
        Question asked: {question}
        Explanation to grade: {explanation}
        
        Output ONLY a valid JSON object in this format:
        {{"clarity_score": 4, "practicality_score": 5, "reasoning": "short reason here"}}
        """,
        input_variables=["question", "explanation"]
    )
    judge_chain = judge_prompt | backend.llm | StrOutputParser()

    # --- TEST DATASET ---
    eval_data = {
    "question": [
        "What is the employment threshold that mandates a mine owner to provide a canteen?",
        "What is the maximum number of hours an adult is allowed to work above ground in a mine in any given week?",
        "Under CMR 2017, what is the minimum air velocity required at the largest span of a working face?",
        "According to the Coal Mines Regulations, who is responsible for the preparation of the Safety Management Plan (SMP)?",
        "What are the height and width restrictions for benches in an opencast metalliferous mine working in hard and compact rock?",
        "Under MMR 1961, who is authorized to prepare charges and fire explosives?",
        "Within what time frame and to whom must a fatal accident be reported under the Mines Act?",
        "In a coal mine depillaring area, up to what distance from the working face must the roof and sides be kept adequately supported?",
        "What is the required scale for providing drinking water to persons employed in a mine?",
        "What minimum clearance must be maintained between a haulage track and the side of a gallery to serve as a travelling roadway?"
    ],
    "ground_truth": [
        "Under Section 62 of the Mines Act 1952, a canteen must be provided and maintained by the owner if more than 250 persons are ordinarily employed in the mine.",
        "According to Section 30 of the Mines Act 1952, no adult shall be required or allowed to work in a mine above ground for more than forty-eight hours in any week, or more than nine hours in any day.",
        "Under the Coal Mines Regulations 2017, the velocity of air current shall not be less than 45 metres per minute at the largest span of a working face.",
        "The owner, agent, and manager of every mine are jointly responsible for the preparation and implementation of a Safety Management Plan.",
        "Under MMR 1961, in hard and compact rock, the height of the bench shall not exceed 6 metres, and the width shall not be less than the height.",
        "No person shall prepare charges, charge, or fire explosives unless they are a competent person holding a valid Blaster's certificate.",
        "Whenever a fatal accident occurs, the owner, agent, or manager shall forthwith (immediately) give notice to the prescribed authorities, including the Regional Inspector and Chief Inspector.",
        "In every depillaring area, the roof and sides of all working places and roadways within a distance of 30 metres from the working face shall be kept adequately supported.",
        "Drinking water shall be provided on a scale of at least two litres for every person employed at any one time. If 100 or more persons are employed, the water must be cooled.",
        "A clearance of not less than one metre shall be provided and maintained between the side of the gallery and the track of the haulage to serve as a travelling roadway."
    ]
}

    print(f"Querying RAG pipeline ({len(eval_data['question'])} questions)...")
    
    answers = []
    contexts = []
    
    # Storage for our custom metrics
    clarity_scores = []
    practicality_scores = []
    judge_reasons = []

    for query in eval_data["question"]:
        # 1. Retrieve Context
        retrieved_docs = backend.retrieve_docs(query)
        doc_texts = [doc.page_content for doc in retrieved_docs]
        contexts.append(doc_texts)
        
        # 2. Generate Answer
        stream = backend.get_answer_stream(query, retrieved_docs)
        full_answer = ""
        for chunk in stream:
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            full_answer += content
        
        # 3. SPLIT THE RESPONSE
        parts = full_answer.split("---EXPLANATION---")
        
        # Grab the first segment as the strict legal part
        strict_regulation = parts[0].strip()
        
        # Join any remaining segments as the explanation
        explanation = "---EXPLANATION---".join(parts[1:]).strip() if len(parts) > 1 else "No explanation provided."
        
        # Give ONLY the strict regulation to Ragas
        answers.append(strict_regulation)
        
        # 4. RUN CUSTOM EDUCATIONAL EVALUATION
        print(f"  -> Grading explanation for: '{query[:30]}...'")
        judge_result_text = judge_chain.invoke({"question": query, "explanation": explanation})
        judge_data = parse_judge_output(judge_result_text)
        
        clarity_scores.append(judge_data.get("clarity_score", 0))
        practicality_scores.append(judge_data.get("practicality_score", 0))
        judge_reasons.append(judge_data.get("reasoning", ""))

    # Compile dataset for Ragas
    data = {
        "question": eval_data["question"],
        "answer": answers, # Now contains only the strict legal text
        "contexts": contexts,
        "ground_truth": eval_data["ground_truth"]
    }
    dataset = Dataset.from_dict(data)

    print("\nStarting Ragas Evaluation (Strict Factual Check)...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    print("\n========== COMBINED EVALUATION RESULTS ==========")
    df = result.to_pandas()
    
    # Merge our custom metrics into the Ragas dataframe
    df['clarity_score_out_of_5'] = clarity_scores
    df['practicality_score_out_of_5'] = practicality_scores
    df['judge_reasoning'] = judge_reasons
    
    df.to_csv("ragas_evaluation_results.csv", index=False)
    
    # Display the most relevant columns
    cols_to_show = [
        c for c in ['question', 'user_input', 'faithfulness', 'answer_relevancy', 
                    'clarity_score_out_of_5', 'practicality_score_out_of_5'] 
        if c in df.columns
    ]
    
    print(df[cols_to_show])
    print("\n✅ Results saved to ragas_evaluation_results.csv")

if __name__ == "__main__":
    run_evaluation()