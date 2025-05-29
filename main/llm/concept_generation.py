from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import re, json
import torch
# This code does load model (huggingdace)

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
print("▶ Loading LLaMA 3-8B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16, # I think this is memmory efficient, less VRAM usage!!
    device_map="auto" #This line is to make sure model is loaded on GPU (however, in Manami's computer, it becomes CPU cuz simply, the my gpu cannot handle it. )
)

# This is the generation pipeline 
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


# Extract concepts with refined few-shot prompting
def extract_concepts(paragraph: str):
    prompt = f"""
You are an academic assistant of computer science field. Extract the most important research concepts (keywords of the from the paragraph wrapped in <TEXT> tags.
Respond only with a JSON object of the form {{"concepts": ["concept1","concept2",…]}}. Extract some unique terms rather than common words in computer science paper paragraph. 

Example 1:
<TEXT>
Transformer-based architectures, like BERT and GPT, have revolutionized NLP by enabling bidirectional attention and large-scale pretraining.
These models achieve state-of-the-art results in tasks such as question answering, machine translation, and text summarization.
</TEXT>
Expected output:
{{"concepts": ["transformer", "BERT", "GPT", "bidirectional attention", "pretraining", "question answering", "machine translation", "text summarization"]}}

Example 2:
<TEXT>
Knowledge graphs represent entities and their relations as a structured graph. They are widely used in tasks like entity linking, question answering, and recommendation systems.
</TEXT>
Expected output:
{{"concepts": ["knowledge graph","entity linking", "question answering", "recommendation systems", "semantic context"]}}

Example 3:
<TEXT>
Graph neural networks (GNNs) extend deep learning to non-Euclidean graph data by iteratively aggregating neighborhood information.  
Popular variants include Graph Convolutional Networks (GCNs), Graph Attention Networks (GATs), and Message Passing Neural Networks (MPNNs).  
They’ve been applied to node classification, link prediction, and molecular property prediction.
</TEXT>
Expected output:
{{"concepts": ["graph neural network", "neighborhood aggregation", "Graph Convolutional Network (GCN)", "Graph Attention Network (GAT)", "Message Passing Neural Network (MPNN)", "node classification", "link prediction", "molecular property prediction"]}}

Example 4:
<TEXT>
To speed up query performance, modern database systems often employ B-tree and LSM-tree indexes.  
B-trees support balanced, ordered data access with logarithmic search time, while Log-Structured Merge trees buffer writes in memory and batch them to disk for high write throughput.  
Secondary indexes like inverted lists or hash indexes accelerate lookups on non-primary key columns.
</TEXT>
Expected output:
{{"concepts": ["B-tree index", "LSM-tree index", "logarithmic search time", "write buffering", "batch disk writes", "secondary index", "inverted list", "hash index", "non-primary key lookup"]}}

Example 5:
<TEXT>
In distributed consensus, Raft and Paxos are two foundational algorithms.  
Raft divides the problem into leader election, log replication, and safety, making it more understandable.  
Paxos focuses on proposer, acceptor, and learner roles to reach agreement despite failures.  
Gossip protocols and vector clock mechanisms are also widely used for state propagation and causality tracking.
</TEXT>
Expected output:
{{"concepts": ["distributed consensus", "Raft algorithm", "leader election", "log replication", "Paxos algorithm", "proposer role", "acceptor role", "learner role", "gossip protocol", "vector clock"]}}


---
Now, without repeating the above examples, extract concepts for the following paragraph:
<TEXT>
{paragraph}
</TEXT>
Expected output (JSON only, no extra text):

"""
    
    result = generator(
        prompt,
        max_new_tokens=600,
        temperature=0.0,
        do_sample=False
    )[0]["generated_text"]

# Max new tokens are set to 600 to limit the LLM's answer (but the token is in addition to the input)
# temperature tells what kind of output. For example, 0.0 tells always same output for the same input but 1.0 has balanced randomness. so 0.0 tells conssistent and accurate answers
# Sample also tells the randomness. so if i set sample = True, then temerature tells how much randomness i want.

    # Extract valid JSON from output
    # This line try to search all strings like this (json-looking substrings) and (re is search for patterns)
    # The json file is made like this {"concepts": ["knowledge graphs", "AI systems", "data integration"]}
    matches = re.findall(r"\{[^{}]+\}", result, re.S)
    if matches:
        last = matches[-1]
        try:
            data = json.loads(last)
            if "concepts" in data and isinstance(data["concepts"], list):
                return data["concepts"]
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse model output for paragraph. Full output:\n{result}")

# Demo
if __name__ == "__main__":
    paragraph = (
        "Recently, as advanced natural language processing techniques, Large Language Models (LLMs) with billion parameters have generated large impacts on various research fields such as Natural Language Processing (NLP), Computer Vision, and Molecule Discovery. Technically, most existing LLMs are transformer-based models pre-trained on a vast amount of textual data from diverse sources, such as articles, books, websites, and other publicly available written materials. As the parameter size of LLMs continues to scale up with a larger training corpus, recent studies indicated that LLMs can lead to the emergence of remarkable capabilities. "
    )

    concepts = extract_concepts(paragraph)
    print("\n▶ Extracted Concepts:")
    for concept in concepts:
        print(f"- {concept}")
