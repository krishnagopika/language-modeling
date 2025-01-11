# Language Modeling

- LM: propability distribution over sequences of tokens/words p(x1,..,xl)

- Generative models

- Autoregressive(AR) language models
p(x1,...,xl) = p(x1)p(x2|x1)....

more time for longer text


AR Language Model:

- predict next word
1. tokenize
2. forward
3. predict probablity of next token
4. sample
5. detokenize



AR Neural Language Model

- minimizing loss is same as increasing the probility

Tokenizers:

- spelling errors
- short lengths will lead to complexity

- token: 3-4 letters
Byte Pair Encoding (BPE)

- corpus of text
- start with one token per character
- merge common pairs of tokens into a token
- repeat until dsired vocab size or all merged




Pre-tokenizers

- space and punctuations get its own ttoken.


Evaluation: Perplexity

- validation loss
- avg per token loss
- exponentiate it

between 1 and |Vocab| 
- intution: no of tokens your model is hesitating with.


Evaluation: agg. std NLP benchmarks

- HELM: Holistic evaluation of large models
- Hugging Face LLM leaderboard


Tasks:

- NarrativeAQ: passagea and books
- MMLU (Massive multitask Language undertanding)


Challenges:

- typeof evaluation - HELM, Harness and Original
- prompting/inconsities
- Train & Test contamination: (if the randomised ouput is ordered then the LLM is trained on it)


Data:

- use all clean data in internet
- thats not possible
- webcrawing common crawl (250 billion pages)
- Extract text from HTML (match is complicated, avoid boiler plate)
- filter undesirable content (NSFW, harmful content and PII)
- Deduplication
- Heuristic filtering: Rm low quality docuemnts (words, word length, outliner toks, dirty toks)
- model based filtering. predict if a page is referenced by wikipedia
- Data mix. classify dat into categories (code/books/entertainment). Rewrite domins using scaling laws to get high downstream performance.
- Lower annealing on high qualiy data, continual pretraining with longer context. (decrease the leanring rate by overfitting the high quality data).


- Collecting the Well data is important.
- challenges: synthetic data? Multi Model data for text?

- competative issues and secrecy.
- common academic datsets - C4 (105B tokens | 800GB), The Pile (280B tokens), Dolma (3T tokens) and FineWeb(15T tokens)
- closed sourced llama@ 2T tokens,  LLama 3T tokens, GPT-4 (13T tokens?)



**Scaling laws:**

- More data and langer modles => better performence
- more compute, datasets and parameters
- New pipelines: hyperparameter tuning on smaller models and extrapolate using scale.
- size vs data : chinchilla
- isoflop : same compute.

FLOPs -> parameters
FLOPs -> tokens

(20:1)  -> 20 tokens for each parameters.

inference costs: 150: 1 ( t -> p ) (optimization for inference)

- resource allocation data and algorithms.


Bitter lesson : 

- models improve with sale & Moore's law.
- dont over complicate

Training a SOTA model

Llama3 15.6 T tokens and 400 B parameters

   - FLOPS = 6 * no of parameters * data(tokns) = 6NP = 3.8 e25 FLOPs
   - sligtly less than 1e26 flops.
   - Compute: 16K H100 with average throughpt of 400TFLOPs
   - 30 M GPU hours
   - 70 days
   - maybe 52 M for training and 25 M for employess
   - carbon emmited = 4400 tCO2eq
   - Next model - 10X FLOPs

   **Post Training:**

   - Task allignment -> assist uer + moderation
   - SFT ex Alpace(LLM Generated data or synthetic data generation) : language modeling of desired answer
   - you need little data for SFT - LIMA paper
   - knowledge is available in pretrained LLM -> tune it to user
   - synthetic data with Human in the loop
   


   RLFT:

   SFT limitations: 
   
   - bound by human abilities to generate things.
   - Hallucinations even though the data is correct. (Model doesnt learn so if something new comes up that might be correct so that would cause issues)
   - Pice: collecting idel answers is expensive.


   RLHF: instad of cloning go for maximizing the preference.

   - label the prefered answes
   - algorithms to generate prefered answers.


  1. RLHF: PPO (Proximal Policy Optimization )

  - option 1: reward -> baseiing the respone and using binary
  - option 2: reward model R -> classify the outpts based on human preferences. (Bradly terry model)
    - use of logits as reward -> continious info -> information heavy!

  - reward - regularization to not to reach infinity 
  - good answers but not likelyhood of different texts.

limitations:

- complicated: cliping

DPO (Direct Preference Optimization) -> simplification of PPO:

- maximize the prefered over less prefered.
- global minima is equivalent to RLHF/PPO

Why PPO over DPO in intial days? Open AI staff were teh people who wrote PPO.

- PPO: train RM with labeled and tune the model later using unlabeled data. In DPO labele data is required.
- DPO is much simpler than PPO and performs better as well. 



How to collect data ?

- humans - laws, instructions and a lot of complexities.
- human erros? avoid correcteness like form
slow and expensive. 
- more RLHF -> increase in answer length
- annotator distributed shift
- crowd souricng ethics (pay and going through toxic data)



- maybe use LLM preferenes over Human Preferences. (LLMs that are aligned with Human preferences with more accuracy)


**LLM evaluation after post training**

evaluating unbouded answers (ChatGPT). 
- validation loss wont work.
- cant use perplexity: not coloborated
- LArge diversity
- open-ended tasks


solution: what would users ask?

- Generation, Open QA, Brainstroming, Chat, Rewrite, class, summ, closed QA and Extract.


- ChatBot Arena


- LLM for LLM evaluation. average win-probablity (AlpacaEval Leaderboard)
- cheap
- LLM evaluation limitations: sprious correlation. (bias).
- solution: regression analysis/ casual innference to "control" length.


**Systems**

- expensive and scarse
- resource allocations and optimized pipelines.
- GPU: optiised for throughput. parllel processing.
- compute is improving when compared to storage and networkng.
- memory heirarchy (closer to chores -> past but less memory, further from cores ->more memory but slower)

MLU (model flop unitization)

- Observed throughput/ theoretical best for thet GPU.
- 50% is great. LLAMA(~45%)


solutions:

- low precesion -> few bits -? fast communication &  lower memory consuption

- for DL: decimal pression doest matter.
   - matrix multiplication.

- AMP (Automatic Mixed Precision) for training. (weights tsired in 32b but  sent as 16b)

1. Operation fusion: communicate once.

- communication is slow.
- pytorch line moves variabiles closer to global memory 
- torch.compile (2X faster) -> rewrites the code in C++ n cuda so that the communication is only done once.

others:

2. tiling

- Flash attention

3. parallelization

4. Architecture Sparsity



Other topics to look into:

1. Architectire: MoE & SSM
2. Decoding and Inferece
3. UI & tols
4. Multimodality
5. Misuse
6. Context size
7. Data wall
8. Legalty of data collection.

Stanford classes:


CS224N -> historical context - dont bother 
CS324 -> in-depth reading - maybe
CS336 -> build an LLM - Brilliant! -> https://stanford-cs336.github.io/spring2024/































